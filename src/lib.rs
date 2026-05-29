//! Conservation Spectral SDK v2 — Hyper-Optimized
//!
//! Retro-language insights baked in:
//! - FORTRAN IV: column-major Laplacian, blocked/tiled matvec for L2 cache
//! - Assembly: SIMD (wide crate), 64-byte alignment, prefetch hints
//! - APL: batch analysis API — process N graphs simultaneously
//! - LISP: const generics for compile-time sizes, typestate pipeline safety
//! - Ada: debug assertions on all invariants, proptest suite
//! - Forth: builder-pattern pipeline — Graph::new().build_laplacian().eigendecompose().analyze()

#![warn(clippy::all)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod aligned;
pub mod graph;
pub mod laplacian;
pub mod eigen;
pub mod conservation;
pub mod batch;
pub mod tracker;

pub use aligned::{AlignedVec, ALIGN_64};
pub use graph::Graph;
pub use laplacian::{Laplacian, LaplacianKind, ColumnMajorMatrix};
pub use eigen::{EigenResult, EigenMethod};
pub use conservation::{ConservationReport, AlignmentCoefficient};
pub use batch::BatchAnalyzer;
pub use tracker::ConservationTracker;

// Re-export pipeline stages for Forth-style builder composition
pub use graph::GraphBuilder;
pub use laplacian::LaplacianBuilder;
pub use eigen::EigenBuilder;
pub use conservation::AnalysisBuilder;
