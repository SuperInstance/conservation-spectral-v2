//! Conservation analysis — alignment coefficients, anomaly detection, spectral fingerprinting.
//!
//! Ada insight: debug_assert! on every invariant, property-based tests in benches/.
//! Forth insight: each function is a composable local section of the sheaf.

use crate::aligned::AlignedVec;
use crate::eigen::{EigenDone, EigenResult};
use crate::graph::Graph;
use crate::laplacian::{Laplacian, LaplacianBuilt};

/// Alignment coefficient α = λ₂/CR(a).
/// Measures how well attribute a aligns with the Fiedler vector direction.
#[derive(Debug, Clone)]
pub struct AlignmentCoefficient {
    pub alpha: f64,
    pub lambda2: f64,
    pub cr: f64,
    pub attribute_name: String,
}

/// Full conservation report.
#[derive(Debug, Clone)]
pub struct ConservationReport {
    pub alignments: Vec<AlignmentCoefficient>,
    pub spectral_gap: f64,
    pub cheeger_constant: f64,
    pub spectral_entropy: f64,
    pub effective_dimension: f64,
    pub anomalies: Vec<Anomaly>,
    pub fiedler_vector: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyType {
    ConservationViolation,
    SpectralOutlier,
    StructuralBreak,
}

#[derive(Debug, Clone)]
pub struct Anomaly {
    pub vertex: usize,
    pub z_score: f64,
    pub kind: AnomalyType,
    pub description: String,
}

/// Typestate: analysis complete.
pub struct AnalysisDone;

/// Builder for conservation analysis (Forth-style pipeline stage 4).
pub struct AnalysisBuilder<'a> {
    eigen: &'a EigenResult,
    laplacian: &'a Laplacian,
    graph: Option<&'a Graph>,
    attributes: Vec<(&'a str, &'a [f64])>,
    anomaly_threshold: f64,
}

impl<'a> AnalysisBuilder<'a> {
    pub fn new(eigen: &'a EigenResult, laplacian: &'a Laplacian, _state: &'a EigenDone) -> Self {
        Self {
            eigen,
            laplacian,
            graph: None,
            attributes: Vec::new(),
            anomaly_threshold: 2.0,
        }
    }

    pub fn graph(mut self, g: &'a Graph) -> Self {
        self.graph = Some(g);
        self
    }

    pub fn attribute(mut self, name: &'a str, values: &'a [f64]) -> Self {
        debug_assert_eq!(values.len(), self.eigen.n, "attribute length must match graph size");
        self.attributes.push((name, values));
        self
    }

    pub fn anomaly_threshold(mut self, t: f64) -> Self {
        self.anomaly_threshold = t;
        self
    }

    pub fn build(self) -> (ConservationReport, AnalysisDone) {
        let n = self.eigen.n;

        // If no attributes specified, use uniform
        let attrs: Vec<(&str, &[f64])> = if self.attributes.is_empty() {
            // Use a synthetic uniform attribute
            // We can't create a temporary here easily, so just compute alignment
            // for a default attribute
            vec![]
        } else {
            self.attributes
        };

        let mut alignments = Vec::new();
        for (name, attr) in &attrs {
            let ac = compute_alignment(self.eigen, self.laplacian, attr, name);
            alignments.push(ac);
        }

        // If no attributes provided, compute a default one
        if attrs.is_empty() {
            let uniform: Vec<f64> = vec![1.0; n];
            let ac = compute_alignment(self.eigen, self.laplacian, &uniform, "uniform");
            alignments.push(ac);
        }

        // Spectral gap
        let spectral_gap = self.eigen.spectral_gap();

        // Cheeger constant: λ₂/2
        let cheeger = if self.eigen.k >= 2 {
            self.eigen.eigenvalues[1] / 2.0
        } else {
            0.0
        };

        // Spectral entropy
        let evals = self.eigen.eigenvalues.as_slice();
        let total: f64 = evals.iter().map(|e| e.abs()).sum();
        let spectral_entropy = if total > 1e-15 {
            evals
                .iter()
                .map(|&e| {
                    let p = e.abs() / total;
                    if p > 1e-15 { -p * p.ln() } else { 0.0 }
                })
                .sum()
        } else {
            0.0
        };

        let effective_dimension = spectral_entropy.exp();

        // Anomaly detection
        let anomalies = detect_anomalies(self.eigen, self.anomaly_threshold);

        // Fiedler vector
        let fiedler = if self.eigen.k >= 2 {
            self.eigen.fiedler_vector().to_vec()
        } else {
            vec![]
        };

        let report = ConservationReport {
            alignments,
            spectral_gap,
            cheeger_constant: cheeger,
            spectral_entropy,
            effective_dimension,
            anomalies,
            fiedler_vector: fiedler,
        };

        (report, AnalysisDone)
    }
}

/// Compute alignment coefficient α = λ₂/CR(a).
fn compute_alignment(eigen: &EigenResult, lap: &Laplacian, attr: &[f64], name: &str) -> AlignmentCoefficient {
    let n = eigen.n;
    debug_assert_eq!(attr.len(), n);

    // λ₂
    let lambda2 = eigen.fiedler_value();

    // CR(a) = a^T L a / ||a||² using blocked matvec
    let mut la = AlignedVec::zeroed(n);
    lap.matrix.matvec_blocked::<64>(attr, la.as_mut_slice());
    let cr: f64 = attr.iter().zip(la.as_slice()).map(|(a, l)| a * l).sum();
    let a_norm_sq: f64 = attr.iter().map(|x| x * x).sum();

    let cr_normalized = if a_norm_sq > 1e-15 { cr / a_norm_sq } else { 0.0 };
    let alpha = if cr_normalized.abs() > 1e-15 {
        lambda2 / cr_normalized
    } else {
        0.0
    };

    AlignmentCoefficient {
        alpha,
        lambda2,
        cr: cr_normalized,
        attribute_name: name.to_string(),
    }
}

/// Z-score based anomaly detection on Fiedler vector.
fn detect_anomalies(eigen: &EigenResult, threshold: f64) -> Vec<Anomaly> {
    if eigen.k < 2 {
        return vec![];
    }
    let fiedler = eigen.fiedler_vector();
    let n = fiedler.len();

    let mean: f64 = fiedler.iter().sum::<f64>() / n as f64;
    let var: f64 = fiedler.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std = var.sqrt();
    if std < 1e-15 {
        return vec![];
    }

    fiedler
        .iter()
        .enumerate()
        .filter_map(|(i, &val)| {
            let z = (val - mean) / std;
            if z.abs() > threshold {
                Some(Anomaly {
                    vertex: i,
                    z_score: z,
                    kind: AnomalyType::ConservationViolation,
                    description: format!(
                        "Vertex {} has Fiedler z-score {:.3}",
                        i, z
                    ),
                })
            } else {
                None
            }
        })
        .collect()
}

/// Quick one-shot analysis combining the full pipeline.
/// Forth-style: compose graph → laplacian → eigen → analysis in one call.
pub fn quick_analyze(
    n: usize,
    transitions: &[f64],
    attribute: &[f64],
    attr_name: &str,
) -> ConservationReport {
    debug_assert_eq!(transitions.len(), n * n);
    debug_assert_eq!(attribute.len(), n);

    let graph = crate::graph::Graph::from_transitions(n, transitions);
    let gb = crate::graph::GraphBuilder::new(n);
    // Since we already have the graph, use direct construction
    let (_graph, graph_built) = (graph, crate::graph::GraphBuilt::mark());

    let lap_builder = crate::laplacian::LaplacianBuilder::new(&_graph, &graph_built);
    let (lap, lap_built) = lap_builder.build();

    let eigen_builder = crate::eigen::EigenBuilder::new(&lap, &lap_built)
        .k(5.min(n))
        .method(crate::eigen::EigenMethod::Lanczos);
    let (eigen, eigen_done) = eigen_builder.build();

    let analysis = crate::conservation::AnalysisBuilder::new(&eigen, &lap, &eigen_done)
        .attribute(attr_name, attribute)
        .build();

    analysis.0
}
