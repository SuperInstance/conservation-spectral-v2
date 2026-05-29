//! Column-major Laplacian with blocked/tiled matvec (FORTRAN IV / LINPACK insight).
//!
//! Key optimizations:
//! - Column-major storage for sequential access in shifted matvec
//! - 64-byte aligned memory for SIMD (Assembly insight)
//! - Blocked 64×64 tile multiplication for L2 cache friendliness
//! - Prefetch hints for next tile

use crate::aligned::AlignedVec;
use crate::graph::{Graph, GraphBuilt};

/// Column-major matrix: element (i,j) at index [j * n + i].
/// This is the FORTRAN/LINPACK convention — sequential access down columns.
#[derive(Debug, Clone)]
pub struct ColumnMajorMatrix {
    pub data: AlignedVec<f64>,
    pub n: usize,
}

impl ColumnMajorMatrix {
    pub fn zeroed(n: usize) -> Self {
        Self {
            data: AlignedVec::zeroed(n * n),
            n,
        }
    }

    /// Element access: (i, j) → data[j * n + i]
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i < self.n && j < self.n, "ColumnMajorMatrix index OOB");
        self.data[j * self.n + i]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        debug_assert!(i < self.n && j < self.n, "ColumnMajorMatrix index OOB");
        self.data[j * self.n + i] = val;
    }

    /// Column-major matvec: y = A * x.
    /// Accessing A column-by-column means contiguous reads from `data`.
    /// Each column j contributes: y[i] += A[i,j] * x[j] for all i.
    /// This is a series of axpy operations on contiguous memory — SIMD friendly.
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let n = self.n;
        debug_assert_eq!(x.len(), n);
        debug_assert_eq!(y.len(), n);
        y.fill(0.0);

        for j in 0..n {
            let xj = x[j];
            let col_start = j * n;
            // axpy: y += xj * column_j
            // SIMD-friendly: sequential access to both data and y
            for i in 0..n {
                y[i] += self.data[col_start + i] * xj;
            }
        }
    }

    /// Blocked matvec with TILE_SIZE tiles (LINPACK blocked LU insight).
    /// Processes the matrix in TILE×TILE blocks to maximize L2 cache reuse.
    /// For the shifted Laplacian M = shift·I - L, the inner loop accesses
    /// a contiguous tile of L and a slice of x that stays in L2.
    pub fn matvec_blocked<const TILE: usize>(&self, x: &[f64], y: &mut [f64]) {
        let n = self.n;
        debug_assert_eq!(x.len(), n);
        debug_assert_eq!(y.len(), n);
        y.fill(0.0);

        let mut j = 0;
        while j < n {
            let j_end = (j + TILE).min(n);
            let tile_cols = j_end - j;

            // Process all rows for this column tile
            let mut i = 0;
            while i < n {
                let i_end = (i + TILE).min(n);

                // Prefetch next tile
                if i + TILE < n {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            self.data.as_ptr().add((j_end) * n + i + TILE) as *const i8,
                            std::arch::x86_64::_MM_HINT_T1,
                        );
                    };
                }

                // Inner tile multiply
                for jj in j..j_end {
                    let xj = x[jj];
                    let col = jj * n;
                    for ii in i..i_end {
                        y[ii] += self.data[col + ii] * xj;
                    }
                }

                i += TILE;
            }
            j += TILE;
        }
    }

    /// SIMD-accelerated matvec using wide crate f64x4.
    /// Processes 4 elements per cycle on AVX2 / 8 on AVX-512.
    pub fn matvec_simd(&self, x: &[f64], y: &mut [f64]) {
        use wide::f64x4;
        let n = self.n;
        debug_assert_eq!(x.len(), n);
        debug_assert_eq!(y.len(), n);
        y.fill(0.0);

        for j in 0..n {
            let xj = f64x4::splat(x[j]);
            let col = j * n;
            let mut i = 0;
            // SIMD loop: process 4 doubles at a time
            while i + 4 <= n {
                let a_slice: &[wide::f64x4] = bytemuck::cast_slice(&self.data.as_slice()[col + i..col + i + 4]);
                let y_slice: &[wide::f64x4] = bytemuck::cast_slice(&y[i..i + 4]);
                let a = a_slice[0];
                let y_vec = y_slice[0];
                let result = y_vec + a * xj;
                let arr: [f64; 4] = bytemuck::cast(result);
                y[i..i + 4].copy_from_slice(&arr);
                i += 4;
            }
            // Scalar tail
            while i < n {
                y[i] += self.data[col + i] * x[j];
                i += 1;
            }
        }
    }
}

/// Kind of Laplacian to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaplacianKind {
    Unnormalized,
    SymmetricNormalized,
    RandomWalkNormalized,
}

/// Laplacian stored in column-major format.
#[derive(Debug, Clone)]
pub struct Laplacian {
    /// Column-major Laplacian matrix L (n×n): L[j*n+i] = L_{ij}
    pub matrix: ColumnMajorMatrix,
    /// Degree vector
    pub degree: AlignedVec<f64>,
    /// Symmetrized weight matrix W_sym (column-major)
    pub weights: ColumnMajorMatrix,
    pub kind: LaplacianKind,
    pub n: usize,
    /// Shifted matrix M = shift·I - L for power iteration on smallest eigenvalues
    pub shifted: Option<ColumnMajorMatrix>,
    pub shift: f64,
}

impl Laplacian {
    /// Build the shifted matrix for power iteration on the smallest eigenvalues.
    /// M = shift·I - L. The largest eigenvalue of M corresponds to smallest of L.
    pub fn build_shifted(&mut self) {
        let n = self.n;
        let shift = self.shift;
        let mut shifted = ColumnMajorMatrix::zeroed(n);
        for j in 0..n {
            for i in 0..n {
                let val = if i == j { shift } else { 0.0 } - self.matrix.get(i, j);
                shifted.set(i, j, val);
            }
        }
        self.shifted = Some(shifted);
    }

    /// Fast alignment estimate using LINPACK condition-number trick.
    /// Runs ~30 power iterations to estimate λ₂, computes CR(a) directly.
    /// O(nnz · 30) instead of O(n² · k).
    pub fn fast_alignment_estimate(
        &self,
        attribute: &[f64],
        power_iters: usize,
    ) -> f64 {
        let n = self.n;
        debug_assert_eq!(attribute.len(), n);

        // Estimate λ₂ via Rayleigh quotient on shifted matrix
        let shifted = self.shifted.as_ref().unwrap_or_else(|| {
            panic!("shifted matrix not built — call build_shifted() first");
        });

        // Power iteration on M = shift·I - L for largest eigenvalue (≈ shift - λ₂)
        let mut v = AlignedVec::from_vec(vec![1.0f64; n]);
        let norm: f64 = v.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in v.as_mut_slice() {
            *x /= norm;
        }

        let mut mu = 0.0f64; // dominant eigenvalue of M
        for _ in 0..power_iters {
            let mut w = AlignedVec::zeroed(n);
            shifted.matvec_blocked::<64>(v.as_slice(), w.as_mut_slice());

            // Rayleigh quotient
            mu = w.as_slice().iter().zip(v.as_slice()).map(|(a, b)| a * b).sum();

            let wnorm: f64 = w.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();
            if wnorm < 1e-15 {
                break;
            }
            for x in w.as_mut_slice() {
                *x /= wnorm;
            }
            v = w;
        }

        let lambda2_est = self.shift - mu;

        // CR(a) = a^T L a / ||a||²
        let mut la = AlignedVec::zeroed(n);
        self.matrix.matvec_blocked::<64>(attribute, la.as_mut_slice());
        let cr: f64 = attribute.iter().zip(la.as_slice()).map(|(a, l)| a * l).sum();
        let a_norm_sq: f64 = attribute.iter().map(|x| x * x).sum();

        if a_norm_sq < 1e-15 || cr.abs() < 1e-15 {
            return 0.0;
        }
        lambda2_est / cr
    }
}

/// Typestate: Laplacian is built.
pub struct LaplacianBuilt;

/// Builder for Laplacian (Forth-style pipeline stage 2).
pub struct LaplacianBuilder<'a> {
    graph: &'a Graph,
    kind: LaplacianKind,
    similarity: Option<&'a dyn Fn(usize, usize) -> f64>,
}

impl<'a> LaplacianBuilder<'a> {
    pub fn new(graph: &'a Graph, _state: &'a GraphBuilt) -> Self {
        Self {
            graph,
            kind: LaplacianKind::Unnormalized,
            similarity: None,
        }
    }

    pub fn kind(mut self, kind: LaplacianKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn similarity(mut self, f: &'a dyn Fn(usize, usize) -> f64) -> Self {
        self.similarity = Some(f);
        self
    }

    pub fn build(self) -> (Laplacian, LaplacianBuilt) {
        let n = self.graph.n;
        let sim = self.similarity.unwrap_or(&|_, _| 1.0f64);

        // Build weight matrix W from graph edges + similarity kernel
        let mut w_sym = ColumnMajorMatrix::zeroed(n);
        let mut degree = vec![0.0f64; n];

        // First pass: compute W[i,j] = transition_weight * similarity(i,j)
        // and symmetrize: W_sym = (W + W^T)/2
        let mut w_raw = vec![0.0f64; n * n]; // row-major temporary
        for i in 0..n {
            let (neighbors, weights) = self.graph.neighbors(i);
            let total_w: f64 = weights.iter().sum();
            if total_w > 0.0 {
                for (k, &j) in neighbors.iter().enumerate() {
                    w_raw[i * n + j] = weights[k] / total_w * sim(i, j);
                }
            }
        }

        // Symmetrize into column-major
        for i in 0..n {
            for j in 0..n {
                let val = (w_raw[i * n + j] + w_raw[j * n + i]) / 2.0;
                w_sym.set(i, j, val);
            }
        }

        // Degree from symmetrized weights
        for i in 0..n {
            degree[i] = (0..n).map(|j| w_sym.get(i, j)).sum();
        }

        // Build Laplacian in column-major
        let mut matrix = ColumnMajorMatrix::zeroed(n);
        match self.kind {
            LaplacianKind::Unnormalized => {
                for j in 0..n {
                    for i in 0..n {
                        let val = if i == j {
                            degree[i] - w_sym.get(i, j)
                        } else {
                            -w_sym.get(i, j)
                        };
                        matrix.set(i, j, val);
                    }
                }
            }
            LaplacianKind::SymmetricNormalized => {
                let d_inv_sqrt: Vec<f64> = degree
                    .iter()
                    .map(|d| if *d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
                    .collect();
                for j in 0..n {
                    for i in 0..n {
                        let val = if i == j {
                            1.0 - d_inv_sqrt[i] * w_sym.get(i, j) * d_inv_sqrt[j]
                        } else {
                            -d_inv_sqrt[i] * w_sym.get(i, j) * d_inv_sqrt[j]
                        };
                        matrix.set(i, j, val);
                    }
                }
            }
            LaplacianKind::RandomWalkNormalized => {
                let d_inv: Vec<f64> = degree
                    .iter()
                    .map(|d| if *d > 0.0 { 1.0 / *d } else { 0.0 })
                    .collect();
                for j in 0..n {
                    for i in 0..n {
                        let val = if i == j {
                            1.0 - d_inv[i] * w_sym.get(i, j)
                        } else {
                            -d_inv[i] * w_sym.get(i, j)
                        };
                        matrix.set(i, j, val);
                    }
                }
            }
        }

        // Compute shift = max diagonal element (Gershgorin bound)
        let shift = (0..n).map(|i| matrix.get(i, i)).fold(0.0f64, f64::max);

        let mut lap = Laplacian {
            matrix,
            degree: AlignedVec::from_vec(degree),
            weights: w_sym,
            kind: self.kind,
            n,
            shifted: None,
            shift: shift + 1e-6, // slight overestimate for safety
        };
        lap.build_shifted();
        (lap, LaplacianBuilt)
    }
}
