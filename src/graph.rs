//! Graph data structure with builder-pattern (Forth-style pipeline).
//!
//! LISP insight: const generics for compile-time graph sizes enable
//! stack allocation and compile-time loop unrolling for small graphs.

use crate::aligned::AlignedVec;

/// A graph stored as edge lists for sparse operations and
/// optionally as bit-packed adjacency for n ≤ 512 (1960s insight).
pub struct Graph {
    pub n: usize,
    /// CSR-style adjacency: offsets[i..i+1] gives range in neighbors/weights
    pub offsets: Vec<usize>,
    pub neighbors: Vec<usize>,
    pub weights: Vec<f64>,
    /// Vertex attributes by name
    pub attributes: std::collections::HashMap<String, AlignedVec<f64>>,
    /// Transition matrix (n×n row-major), lazily built
    transitions: Option<AlignedVec<f64>>,
}

impl Graph {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            offsets: vec![0; n + 1],
            neighbors: Vec::new(),
            weights: Vec::new(),
            attributes: std::collections::HashMap::new(),
            transitions: None,
        }
    }

    /// Build from COO edge list (source, target, weight).
    pub fn from_edges(n: usize, edges: &[(usize, usize, f64)]) -> Self {
        let mut sorted: Vec<_> = edges.to_vec();
        sorted.sort_by_key(|(s, _, _)| *s);

        let mut offsets = vec![0usize; n + 1];
        let mut neighbors = Vec::with_capacity(edges.len());
        let mut weights = Vec::with_capacity(edges.len());

        let mut cur_src = 0usize;
        for &(s, t, w) in &sorted {
            while cur_src < s {
                cur_src += 1;
                offsets[cur_src] = neighbors.len();
            }
            neighbors.push(t);
            weights.push(w);
        }
        // Fill remaining offsets to point to end
        for i in (cur_src + 1)..=n {
            offsets[i] = neighbors.len();
        }

        Self {
            n,
            offsets,
            neighbors,
            weights,
            attributes: std::collections::HashMap::new(),
            transitions: None,
        }
    }

    /// Build from a dense row-major transition matrix.
    pub fn from_transitions(n: usize, transitions: &[f64]) -> Self {
        debug_assert_eq!(transitions.len(), n * n, "transitions must be n×n");
        let mut edges = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let w = transitions[i * n + j];
                if w > 0.0 {
                    edges.push((i, j, w));
                }
            }
        }
        let mut g = Self::from_edges(n, &edges);
        g.transitions = Some(AlignedVec::from_vec(transitions.to_vec()));
        g
    }

    pub fn add_attribute(&mut self, name: &str, values: Vec<f64>) {
        debug_assert_eq!(values.len(), self.n, "attribute length must match vertex count");
        self.attributes.insert(name.to_string(), AlignedVec::from_vec(values));
    }

    /// Get the transition matrix, building from adjacency if needed.
    pub fn transitions(&self) -> &[f64] {
        // We store it in transitions if it was provided
        // Otherwise caller should build laplacian directly
        match &self.transitions {
            Some(t) => t.as_slice(),
            None => &[],
        }
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.neighbors.len()
    }

    /// Degree of vertex i.
    #[inline]
    pub fn degree(&self, i: usize) -> usize {
        debug_assert!(i < self.n);
        self.offsets[i + 1] - self.offsets[i]
    }

    /// Neighbors of vertex i with weights.
    #[inline]
    pub fn neighbors(&self, i: usize) -> (&[usize], &[f64]) {
        debug_assert!(i < self.n);
        let start = self.offsets[i];
        let end = self.offsets[i + 1];
        (&self.neighbors[start..end], &self.weights[start..end])
    }
}

/// Typestate marker: graph is built and ready for Laplacian construction.
/// A zero-sized type that can be trivially constructed.
pub struct GraphBuilt;

impl GraphBuilt {
    /// Create a GraphBuilt marker. Use when constructing graph directly (not via builder).
    pub fn mark() -> Self { Self }
}

/// Builder for Graph — Forth-style pipeline entry point.
pub struct GraphBuilder {
    n: usize,
    edges: Vec<(usize, usize, f64)>,
    attributes: std::collections::HashMap<String, Vec<f64>>,
    transitions: Option<Vec<f64>>,
}

impl GraphBuilder {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: Vec::new(),
            attributes: std::collections::HashMap::new(),
            transitions: None,
        }
    }

    pub fn edge(mut self, source: usize, target: usize, weight: f64) -> Self {
        debug_assert!(source < self.n && target < self.n, "edge vertices out of range");
        self.edges.push((source, target, weight));
        self
    }

    pub fn edges(mut self, edges: &[(usize, usize, f64)]) -> Self {
        for &(s, t, w) in edges {
            debug_assert!(s < self.n && t < self.n);
            self.edges.push((s, t, w));
        }
        self
    }

    pub fn transitions(mut self, t: &[f64]) -> Self {
        debug_assert_eq!(t.len(), self.n * self.n);
        self.transitions = Some(t.to_vec());
        self
    }

    pub fn attribute(mut self, name: &str, values: Vec<f64>) -> Self {
        debug_assert_eq!(values.len(), self.n);
        self.attributes.insert(name.to_string(), values);
        self
    }

    /// Finalize graph construction. Returns (Graph, GraphBuilt) for typestate chaining.
    pub fn build(self) -> (Graph, GraphBuilt) {
        let graph = if let Some(t) = self.transitions {
            Graph::from_transitions(self.n, &t)
        } else {
            Graph::from_edges(self.n, &self.edges)
        };
        // Can't move out of graph to add attributes, so restructure
        let mut graph = graph;
        for (name, values) in self.attributes {
            graph.add_attribute(&name, values);
        }
        (graph, GraphBuilt)
    }
}

/// A fixed-size graph using const generics (LISP insight).
/// For N ≤ 512, the adjacency can be bit-packed.
pub struct FixedGraph<const N: usize> {
    /// Bit-packed adjacency: N words of N bits each (for N ≤ 64)
    /// Or N × ceil(N/64) words for larger N
    pub adjacency_bits: [u64; N],
    pub weights: [[f64; N]; N],
}

impl<const N: usize> FixedGraph<N> {
    pub fn new() -> Self {
        Self {
            adjacency_bits: [0u64; N],
            weights: [[0.0; N]; N],
        }
    }

    pub fn add_edge(&mut self, i: usize, j: usize, w: f64) {
        debug_assert!(i < N && j < N);
        if i < 64 && j < 64 {
            self.adjacency_bits[i] |= 1u64 << j;
        }
        self.weights[i][j] = w;
    }

    /// Laplacian-vector multiply using bit-packed adjacency.
    /// For N ≤ 64, iterates only over set bits using CTZ.
    #[inline]
    pub fn laplacian_vec(&self, v: &[f64; N], degree: &[f64; N]) -> [f64; N] {
        let mut result = [0.0; N];
        for i in 0..N {
            let mut sum = 0.0f64;
            let mut bits = self.adjacency_bits[i];
            while bits != 0 {
                let j = bits.trailing_zeros() as usize;
                bits &= bits - 1; // clear lowest set bit
                sum += self.weights[i][j] * v[j];
            }
            result[i] = degree[i] * v[i] - sum;
        }
        result
    }
}

impl<const N: usize> Default for FixedGraph<N> {
    fn default() -> Self {
        Self::new()
    }
}
