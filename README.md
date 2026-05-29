# Conservation Spectral SDK v2

Hyper-optimized spectral graph analysis for anomaly detection, conservation fingerprinting, and structural health monitoring.

## What's New in v2

This is a ground-up rewrite incorporating insights from retro computing languages:

| Retro Source | Insight Applied | Speedup Mechanism |
|---|---|---|
| **FORTRAN IV / LINPACK** | Column-major Laplacian storage, 64×64 blocked matvec | L2 cache reuse ratio 64:1 vs 1:1 |
| **Assembly / SIMD** | `wide` crate f64x4, 64-byte aligned memory, prefetch hints | 4× throughput per cycle |
| **APL (1966)** | Batch API — process N graphs simultaneously | Amortized kernel overhead |
| **LISP / ML** | Const generics for compile-time graph sizes | Stack allocation, loop unrolling |
| **Ada** | Debug assertions on all invariants, proptest suite | Correctness guaranteed |
| **Forth** | Builder-pattern pipeline: `.build_laplacian().eigendecompose().analyze()` | Typestate safety |

**Target: 10× faster than v1 on 1000+ node graphs.**

## Quick Start

```rust
use conservation_spectral_v2::*;

// Forth-style builder pipeline
let n = 1000;
let transitions = /* n×n row-major transition matrix */;
let attribute = /* f64 vector of length n */;

let report = conservation::quick_analyze(n, &transitions, &attribute, "my_attr");

println!("Alignment α = {:.4}", report.alignments[0].alpha);
println!("Spectral gap = {:.4}", report.spectral_gap);
println!("Cheeger constant = {:.4}", report.cheeger_constant);
```

### Step-by-step pipeline

```rust
// 1. Build graph
let graph = Graph::from_transitions(n, &transitions);
let gb = GraphBuilt::mark();

// 2. Build Laplacian (column-major, blocked)
let (lap, lb) = LaplacianBuilder::new(&graph, &gb)
    .kind(LaplacianKind::Unnormalized)
    .build();

// 3. Eigendecompose (Lanczos for large graphs)
let (eigen, ed) = EigenBuilder::new(&lap, &lb)
    .k(5)
    .method(EigenMethod::Lanczos)
    .build();

// 4. Analyze conservation
let (report, _) = AnalysisBuilder::new(&eigen, &lap, &ed)
    .attribute("temperature", &temperature_data)
    .anomaly_threshold(2.0)
    .build();
```

### Batch Analysis (APL-style)

```rust
let mut analyzer = BatchAnalyzer::new().k(3);
let result = analyzer.analyze_batch(&graph_refs, &attribute, "sensor");

// α for each graph
for (i, alpha) in result.alignment_coefficients.iter().enumerate() {
    println!("Graph {}: α = {:.4}", i, alpha);
}
```

### Fast Alignment Estimate (LINPACK condition-number trick)

```rust
// O(nnz · 30) instead of full O(n² · k) eigendecomposition
let alpha = lap.fast_alignment_estimate(&attribute, 30);
```

## Architecture

```
src/
├── lib.rs           # Re-exports, module structure
├── aligned.rs       # 64-byte aligned Vec<T> for SIMD
├── graph.rs         # Graph (CSR), FixedGraph<const N>, builder
├── laplacian.rs     # Column-major Laplacian, blocked/SIMD matvec
├── eigen.rs         # Lanczos + power iteration with deflation
├── conservation.rs  # Alignment α, anomaly detection, reports
├── batch.rs         # APL-style batch analysis
└── tracker.rs       # Sliding-window real-time tracker
```

## Benchmarks

```bash
cargo bench
```

Benchmarks cover:
- Laplacian construction (100–2000 nodes)
- Eigendecomposition (Lanczos vs power iteration)
- Full pipeline end-to-end
- Matvec: naive vs blocked vs SIMD
- Batch analysis (50 graphs)

## License

MIT
