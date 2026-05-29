//! Real-time conservation tracking with sliding window.
//! Apollo insight: deterministic error tracking for anomaly detection.

use crate::conservation::ConservationReport;
use crate::eigen::EigenMethod;
use crate::graph::Graph;
use crate::laplacian::LaplacianKind;

/// Alert when conservation drops below threshold.
#[derive(Debug, Clone)]
pub struct ConservationAlert {
    pub message: String,
    pub current_ratio: f64,
    pub historical_mean: f64,
    pub percent_change: f64,
}

/// Real-time conservation tracker with sliding window.
pub struct ConservationTracker {
    window_size: usize,
    threshold: f64,
    method: EigenMethod,
    history: Vec<f64>,
    baseline: Option<f64>,
}

impl ConservationTracker {
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window_size,
            threshold,
            method: EigenMethod::Lanczos,
            history: Vec::with_capacity(window_size),
            baseline: None,
        }
    }

    pub fn method(mut self, m: EigenMethod) -> Self {
        self.method = m;
        self
    }

    /// Feed a new observation (transition matrix) and return an alert if conservation degraded.
    pub fn feed(&mut self, transitions: &[f64], n: usize) -> Option<ConservationAlert> {
        if n == 0 {
            return None;
        }

        let graph = Graph::from_transitions(n, transitions);
        let (_g, gb) = ((), crate::graph::GraphBuilt::mark());
        let lap_builder = crate::laplacian::LaplacianBuilder::new(&graph, &gb);
        let (lap, lap_built) = lap_builder.build();

        let k = 3.min(n);
        let eigen_builder = crate::eigen::EigenBuilder::new(&lap, &lap_built)
            .k(k)
            .method(self.method);
        let (eigen, eigen_done) = eigen_builder.build();

        let attr = vec![1.0f64; n];
        let analysis = crate::conservation::AnalysisBuilder::new(&eigen, &lap, &eigen_done)
            .attribute("uniform", &attr)
            .build();

        let alpha = analysis.0.alignments.first().map(|a| a.alpha).unwrap_or(0.0);
        self.history.push(alpha);

        // Establish baseline after 3 observations
        if self.baseline.is_none() && self.history.len() >= 3 {
            self.baseline = Some(self.history[..3].iter().sum::<f64>() / 3.0);
        }

        // Trim window
        while self.history.len() > self.window_size {
            self.history.remove(0);
        }

        // Check for alert
        self.check()
    }

    fn check(&self) -> Option<ConservationAlert> {
        if self.history.len() < 3 || self.baseline.is_none() {
            return None;
        }

        let baseline = self.baseline.unwrap();
        let current = *self.history.last().unwrap();
        let change = (current - baseline) / baseline.abs().max(1e-10);

        if change.abs() > self.threshold {
            Some(ConservationAlert {
                message: format!(
                    "Conservation shift: α from {:.4} to {:.4} ({:.1}% change)",
                    baseline, current, change * 100.0
                ),
                current_ratio: current,
                historical_mean: baseline,
                percent_change: change * 100.0,
            })
        } else {
            None
        }
    }

    pub fn history(&self) -> &[f64] {
        &self.history
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.baseline = None;
    }
}
