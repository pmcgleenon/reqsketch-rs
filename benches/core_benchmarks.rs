use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use reqsketch::{ReqSketch, SearchCriteria};
use rand::prelude::*;

/// Core benchmark suite for REQ sketch - essential measurements only
///
// ========== 1. INSERTION THROUGHPUT ==========

fn bench_insertion_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion_throughput");

    for &batch_size in &[1_000, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(batch_size));
        group.bench_with_input(
            BenchmarkId::new("uniform_data", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let mut sketch = ReqSketch::new();
                    let mut rng = StdRng::seed_from_u64(42);

                    for _ in 0..batch_size {
                        let value: f64 = rng.random_range(0.0..1000000.0);
                        sketch.update(black_box(value));
                    }

                    black_box(sketch)
                });
            }
        );
    }

    group.finish();
}

// ========== 2. QUERY LATENCY ==========

fn bench_query_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_latency");

    // Pre-populate sketch with 100K items
    let mut sketch = ReqSketch::new();
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..100_000 {
        sketch.update(rng.random_range(0.0..1000000.0));
    }

    // Test key quantiles
    let quantiles = vec![0.5, 0.9, 0.95, 0.99];

    for &quantile in &quantiles {
        group.bench_with_input(
            BenchmarkId::new("quantile", format!("p{}", (quantile * 100.0) as u32)),
            &quantile,
            |b, &quantile| {
                b.iter(|| {
                    let result = sketch.quantile(black_box(quantile), SearchCriteria::Inclusive);
                    black_box(result)
                });
            }
        );
    }

    group.finish();
}

// ========== 3. MEMORY EFFICIENCY ==========

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    for &n in &[10_000, 100_000, 1_000_000] {
        group.bench_with_input(
            BenchmarkId::new("compression_ratio", n),
            &n,
            |b, &n| {
                b.iter(|| {
                    let mut sketch = ReqSketch::new();
                    let mut rng = StdRng::seed_from_u64(42);

                    for _ in 0..n {
                        sketch.update(rng.random_range(0.0..1000000.0));
                    }

                    let compression_ratio = n as f64 / sketch.num_retained() as f64;
                    black_box((sketch.num_retained(), compression_ratio))
                });
            }
        );
    }

    group.finish();
}

// ========== 4. SCALING PERFORMANCE ==========

fn bench_scaling_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_performance");
    group.sample_size(10); // Fewer samples for larger tests

    let scales = vec![
        ("1K", 1_000),
        ("10K", 10_000),
        ("100K", 100_000),
        ("1M", 1_000_000),
    ];

    for (name, n) in scales {
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(
            BenchmarkId::new("throughput", name),
            &n,
            |b, &n| {
                b.iter(|| {
                    let mut sketch = ReqSketch::new();
                    let mut rng = StdRng::seed_from_u64(42);

                    for _ in 0..n {
                        let value: f64 = rng.random_range(0.0..1000000.0);
                        sketch.update(black_box(value));
                    }

                    black_box(sketch)
                });
            }
        );
    }

    group.finish();
}

// ========== 5. STATISTICAL ACCURACY ==========

fn bench_statistical_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistical_accuracy");
    group.sample_size(20);

    for &n in &[100_000, 1_000_000] {
        group.bench_with_input(
            BenchmarkId::new("sigma_compliance", n),
            &n,
            |b, &n| {
                b.iter(|| {
                    let mut sketch = ReqSketch::new();

                    // Insert sequential data for predictable accuracy testing
                    for i in 0..n {
                        sketch.update(i as f64);
                    }

                    // Test 3-sigma compliance at critical quantiles
                    let test_ranks = vec![0.9, 0.95, 0.99];
                    let mut compliance_results = Vec::new();

                    for &rank in &test_ranks {
                        let true_quantile = rank * (n - 1) as f64;

                        if let Ok(estimated_rank) = sketch.rank(&true_quantile, SearchCriteria::Inclusive) {
                            // Calculate 3-sigma bounds
                            let k = sketch.k() as f64;
                            let n_f = n as f64;
                            let sigma = (k / (2.0 * n_f)).sqrt();
                            let lower_bound = rank - 3.0 * sigma;
                            let upper_bound = rank + 3.0 * sigma;

                            let in_bounds = estimated_rank >= lower_bound && estimated_rank <= upper_bound;
                            let error = estimated_rank - rank;
                            compliance_results.push((rank, in_bounds, error));
                        }
                    }

                    black_box(compliance_results)
                });
            }
        );
    }

    group.finish();
}

// ========== 6. CONFIGURATION TUNING ==========

fn bench_k_parameter_tuning(c: &mut Criterion) {
    let mut group = c.benchmark_group("k_parameter_tuning");

    let n = 50_000;
    group.throughput(Throughput::Elements(n));

    for &k in &[8, 12, 16, 32] {
        group.bench_with_input(
            BenchmarkId::new("performance_vs_accuracy", k),
            &k,
            |b, &k| {
                b.iter(|| {
                    let mut sketch = ReqSketch::builder().k(k).expect("Operation should succeed").build().expect("Operation should succeed");
                    let mut rng = StdRng::seed_from_u64(42);

                    // Measure insertion performance
                    for _ in 0..n {
                        sketch.update(rng.random_range(0.0..1000000.0));
                    }

                    // Measure memory efficiency
                    let compression_ratio = n as f64 / sketch.num_retained() as f64;

                    // Measure query accuracy (simplified)
                    let mut accuracy_score = 0.0;
                    if let Ok(estimated) = sketch.quantile(0.95, SearchCriteria::Inclusive) {
                        let true_value = 0.95 * (n - 1) as f64;
                        accuracy_score = 1.0 - (estimated - true_value).abs() / true_value;
                    }

                    black_box((k, sketch.num_retained(), compression_ratio, accuracy_score))
                });
            }
        );
    }

    group.finish();
}

// ========== BENCHMARK GROUPS ==========

criterion_group!(
    core_benches,
    bench_insertion_throughput,
    bench_query_latency,
    bench_memory_efficiency,
    bench_scaling_performance,
    bench_statistical_accuracy,
    bench_k_parameter_tuning
);

criterion_main!(core_benches);
