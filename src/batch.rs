//! Batch analysis API — APL insight: process N graphs simultaneously.
//!
//! In APL, operations apply to entire arrays. This module provides
//! vectorized conservation analysis across multiple graphs or attributes.

use crate::aligned::AlignedVec;
use crate::conservation::ConservationReport;
use crate::eigen::EigenMethod;
use crate::graph::Graph;
use crate::laplacian::LaplacianKind;

/// Result of batch analysis for multiple graphs.
#[derive(Debug)]
pub struct BatchResult {
    pub reports: Vec<ConservationReport>,
    pub alignment_coefficients: Vec<f64>,
}

/// Batch analyzer for computing alignment across multiple graphs.
pub struct BatchAnalyzer {
    pub n_graphs: usize,
    k: usize,
    method: EigenMethod,
    laplacian_kind: LaplacianKind,
}

impl BatchAnalyzer {
    pub fn new() -> Self {
        Self {
            n_graphs: 0,
            k: 5,
            method: EigenMethod::Lanczos,
            laplacian_kind: LaplacianKind::Unnormalized,
        }
    }

    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    pub fn method(mut self, m: EigenMethod) -> Self {
        self.method = m;
        self
    }

    /// Analyze multiple graphs with the same attribute.
    /// APL-style: one operation applies across all graphs.
    pub fn analyze_batch(
        &mut self,
        graphs: &[&Graph],
        attribute: &[f64],
        attr_name: &str,
    ) -> BatchResult {
        self.n_graphs = graphs.len();
        let mut reports = Vec::with_capacity(graphs.len());
        let mut alphas = Vec::with_capacity(graphs.len());

        for graph in graphs {
            let n = graph.n;
            debug_assert_eq!(attribute.len(), n, "attribute length must match each graph");

            let (_g, gb) = ((), crate::graph::GraphBuilt::mark());
            let lap_builder = crate::laplacian::LaplacianBuilder::new(graph, &gb);
            let (lap, lap_built) = lap_builder.build();

            let k = self.k.min(n);
            let eigen_builder = crate::eigen::EigenBuilder::new(&lap, &lap_built)
                .k(k)
                .method(self.method);
            let (eigen, eigen_done) = eigen_builder.build();

            let analysis = crate::conservation::AnalysisBuilder::new(&eigen, &lap, &eigen_done)
                .attribute(attr_name, attribute)
                .build();

            let alpha = analysis.0.alignments.first().map(|a| a.alpha).unwrap_or(0.0);
            alphas.push(alpha);
            reports.push(analysis.0);
        }

        BatchResult {
            reports,
            alignment_coefficients: alphas,
        }
    }

    /// Batch fast alignment estimates using LINPACK condition-number trick.
    /// O(nnz · 30) per graph instead of full eigendecomposition.
    pub fn fast_alignment_batch(
        &self,
        transitions_batch: &[&[f64]], // each is n×n row-major
        n: usize,
        attribute: &[f64],
        power_iters: usize,
    ) -> Vec<f64> {
        transitions_batch
            .iter()
            .map(|trans| {
                let graph = Graph::from_transitions(n, trans);
                let (_g, gb) = ((), crate::graph::GraphBuilt::mark());
                let mut lap_builder = crate::laplacian::LaplacianBuilder::new(&graph, &gb);
                let (mut lap, _) = lap_builder.build();

                // Build shifted matrix for power iteration
                lap.build_shifted();
                lap.fast_alignment_estimate(attribute, power_iters)
            })
            .collect()
    }

    /// Batch alignment for multiple attributes on the same graph.
    pub fn analyze_attributes(
        &mut self,
        graph: &Graph,
        attributes: &[(&str, &[f64])],
    ) -> BatchResult {
        let (_g, gb) = ((), crate::graph::GraphBuilt::mark());
        let lap_builder = crate::laplacian::LaplacianBuilder::new(graph, &gb);
        let (lap, lap_built) = lap_builder.build();

        let k = self.k.min(graph.n);
        let eigen_builder = crate::eigen::EigenBuilder::new(&lap, &lap_built)
            .k(k)
            .method(self.method);
        let (eigen, eigen_done) = eigen_builder.build();

        let mut reports = Vec::with_capacity(attributes.len());
        let mut alphas = Vec::with_capacity(attributes.len());

        for (name, attr) in attributes {
            let analysis = crate::conservation::AnalysisBuilder::new(&eigen, &lap, &eigen_done)
                .attribute(name, attr)
                .build();

            let alpha = analysis.0.alignments.first().map(|a| a.alpha).unwrap_or(0.0);
            alphas.push(alpha);
            reports.push(analysis.0);
        }

        BatchResult {
            reports,
            alignment_coefficients: alphas,
        }
    }
}

impl Default for BatchAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
