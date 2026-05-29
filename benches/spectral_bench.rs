//! Benchmarks comparing v1 patterns vs v2 optimizations.
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use conservation_spectral_v2::*;

/// Generate a random n×n sparse transition matrix (band structure + noise).
fn random_transitions(n: usize, density: f64) -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut mat = vec![0.0f64; n * n];

    for i in 0..n {
        // Band: connect to nearby vertices
        for d in -3i32..=3 {
            let j = (i as i32 + d) as usize;
            if j < n {
                mat[i * n + j] = rng.gen_range(0.1..1.0);
            }
        }
        // Random sparse connections
        for _ in 0..((n as f64 * density) as usize) {
            let j = rng.gen_range(0..n);
            mat[i * n + j] += rng.gen_range(0.1..0.5);
        }
        // Normalize row
        let row_sum: f64 = mat[i * n..(i + 1) * n].iter().sum();
        if row_sum > 0.0 {
            for j in 0..n {
                mat[i * n + j] /= row_sum;
            }
        }
    }
    mat
}

fn bench_laplacian_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("laplacian_build");
    for n in [100, 500, 1000, 2000] {
        let trans = random_transitions(n, 0.05);
        let graph = graph::Graph::from_transitions(n, &trans);
        let gb = graph::GraphBuilt::mark();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            b.iter(|| {
                let builder = laplacian::LaplacianBuilder::new(&graph, &gb);
                let (_lap, _built) = builder.build();
                black_box(&_lap);
            });
        });
    }
    group.finish();
}

fn bench_eigendecompose(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigendecompose");
    for n in [100, 500, 1000] {
        let trans = random_transitions(n, 0.05);
        let graph = graph::Graph::from_transitions(n, &trans);
        let gb = graph::GraphBuilt::mark();
        let (lap, lb) = laplacian::LaplacianBuilder::new(&graph, &gb).build();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("lanczos_n", n),
            &n,
            |b, _| {
                b.iter(|| {
                    let builder = eigen::EigenBuilder::new(&lap, &lb)
                        .k(5)
                        .method(eigen::EigenMethod::Lanczos);
                    let (_result, _) = builder.build();
                    black_box(&_result);
                });
            },
        );
    }
    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    for n in [100, 500, 1000] {
        let trans = random_transitions(n, 0.05);
        let attr: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            b.iter(|| {
                let report = conservation::quick_analyze(n, &trans, &attr, "test");
                black_box(&report);
            });
        });
    }
    group.finish();
}

fn bench_matvec_methods(c: &mut Criterion) {
    let n = 1000;
    let mut group = c.benchmark_group("matvec");

    let trans = random_transitions(n, 0.05);
    let graph = graph::Graph::from_transitions(n, &trans);
    let gb = graph::GraphBuilt::mark();
    let (lap, _) = laplacian::LaplacianBuilder::new(&graph, &gb).build();
    let shifted = lap.shifted.as_ref().unwrap();
    let x = vec![1.0f64; n];
    let mut y = vec![0.0f64; n];

    group.throughput(Throughput::Elements((n * n) as u64));
    group.bench_function("naive_column_major", |b| {
        b.iter(|| {
            shifted.matvec(&x, &mut y);
            black_box(&y);
        });
    });

    group.bench_function("blocked_64", |b| {
        b.iter(|| {
            shifted.matvec_blocked::<64>(&x, &mut y);
            black_box(&y);
        });
    });

    group.bench_function("simd_f64x4", |b| {
        b.iter(|| {
            shifted.matvec_simd(&x, &mut y);
            black_box(&y);
        });
    });

    group.finish();
}

fn bench_batch_analysis(c: &mut Criterion) {
    let n = 100;
    let n_graphs = 50;
    let trans: Vec<Vec<f64>> = (0..n_graphs)
        .map(|_| random_transitions(n, 0.05))
        .collect();
    let attr: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let graphs: Vec<graph::Graph> = trans.iter().map(|t| graph::Graph::from_transitions(n, t)).collect();
    let refs: Vec<&graph::Graph> = graphs.iter().collect();

    let mut group = c.benchmark_group("batch");
    group.throughput(Throughput::Elements((n_graphs * n) as u64));
    group.bench_function("50_graphs", |b| {
        b.iter(|| {
            let mut analyzer = batch::BatchAnalyzer::new().k(3);
            let result = analyzer.analyze_batch(&refs, &attr, "test");
            black_box(&result);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_laplacian_build,
    bench_eigendecompose,
    bench_full_pipeline,
    bench_matvec_methods,
    bench_batch_analysis,
);
criterion_main!(benches);
