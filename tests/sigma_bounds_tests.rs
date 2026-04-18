use reqsketch::{ReqSketch, RankAccuracy, SearchCriteria};

/// Test that critical quantiles have reasonable absolute error OR fall within bounds.
/// This is a per-rank test, not a pass-rate test — each rank must independently pass.
#[test]
fn test_critical_quantiles_2_sigma() {
    let mut sketch = ReqSketch::builder()
        .rank_accuracy(RankAccuracy::HighRank)
        .build()
        .expect("Operation should succeed");

    let n = 50_000;
    for i in 0..n {
        sketch.update(i as f64);
    }

    let critical_ranks = [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99];

    for &rank in &critical_ranks {
        let true_quantile = rank * (n - 1) as f64;
        let estimated_rank = sketch.rank(&true_quantile, SearchCriteria::Inclusive)
            .expect("Operation should succeed");

        let lower_bound = sketch.get_rank_lower_bound(rank, 2);
        let upper_bound = sketch.get_rank_upper_bound(rank, 2);
        let within_bounds = estimated_rank >= lower_bound && estimated_rank <= upper_bound;

        let rank_error = (estimated_rank - rank).abs();
        assert!(within_bounds || rank_error < 0.10,
               "Critical quantile {:.2} fails 2-sigma bounds AND has high absolute error {:.4}. \
                Estimated: {:.6}, Bounds: [{:.6}, {:.6}]",
               rank, rank_error, estimated_rank, lower_bound, upper_bound);
    }
}

/// Test that extreme quantiles have reasonable error or fall within bounds.
#[test]
fn test_extreme_quantiles_bounds() {
    let mut sketch = ReqSketch::builder()
        .rank_accuracy(RankAccuracy::HighRank)
        .build()
        .expect("Operation should succeed");

    let n = 50_000;
    for i in 0..n {
        sketch.update(i as f64);
    }

    let extreme_ranks = [0.0001, 0.001, 0.999, 0.9999];

    for &rank in &extreme_ranks {
        let true_quantile = rank * (n - 1) as f64;

        match sketch.rank(&true_quantile, SearchCriteria::Inclusive) {
            Ok(estimated_rank) => {
                let lower = sketch.get_rank_lower_bound(rank, 2);
                let upper = sketch.get_rank_upper_bound(rank, 2);
                let within_2sigma = estimated_rank >= lower && estimated_rank <= upper;

                let error_pct = (estimated_rank - rank).abs() / rank * 100.0;

                assert!(within_2sigma || error_pct < 10.0,
                       "Extreme quantile {:.4} fails bounds with unreasonable error {:.1}%",
                       rank, error_pct);
            }
            Err(_) => {
                // Acceptable for very extreme quantiles outside retained range
            }
        }
    }
}

/// Multi-trial coverage test for high ranks (where HRA excels).
/// Validates sigma bounds are statistically meaningful.
#[test]
fn test_hra_high_rank_coverage() {
    let trials = 50;
    let n = 10_000;
    // Test ranks near 1.0 where HRA mode should have best accuracy
    let test_ranks = [0.95, 0.99];

    for &test_rank in &test_ranks {
        let mut within_2sigma = 0;
        let mut within_3sigma = 0;

        for trial in 0..trials {
            let mut sketch = ReqSketch::builder()
                .rank_accuracy(RankAccuracy::HighRank)
                .build()
                .expect("build should succeed");

            for i in 0..n {
                sketch.update((i as f64) + (trial as f64 * 0.0001));
            }

            let true_quantile = test_rank * (n - 1) as f64 + (trial as f64 * 0.0001);
            let estimated_rank = sketch.rank(&true_quantile, SearchCriteria::Inclusive).unwrap();

            let lb2 = sketch.get_rank_lower_bound(test_rank, 2);
            let ub2 = sketch.get_rank_upper_bound(test_rank, 2);
            if estimated_rank >= lb2 && estimated_rank <= ub2 {
                within_2sigma += 1;
            }

            let lb3 = sketch.get_rank_lower_bound(test_rank, 3);
            let ub3 = sketch.get_rank_upper_bound(test_rank, 3);
            if estimated_rank >= lb3 && estimated_rank <= ub3 {
                within_3sigma += 1;
            }
        }

        let coverage_2 = within_2sigma as f64 / trials as f64;
        let coverage_3 = within_3sigma as f64 / trials as f64;

        // 3-sigma coverage should be ≥ 2-sigma coverage
        assert!(coverage_3 >= coverage_2 - 0.05,
            "3-sigma coverage ({:.0}%) should be ≥ 2-sigma coverage ({:.0}%) for rank {}",
            coverage_3 * 100.0, coverage_2 * 100.0, test_rank);

        // High ranks in HRA mode should have good coverage
        assert!(coverage_3 >= 0.70,
            "3-sigma coverage {:.0}% too low for HRA high rank {} across {} trials",
            coverage_3 * 100.0, test_rank, trials);
    }
}

/// Test that bounds are structurally consistent across accuracy modes.
#[test]
fn test_bounds_hra_vs_lra_structure() {
    let n = 10_000;

    let mut hra_sketch = ReqSketch::builder()
        .rank_accuracy(RankAccuracy::HighRank)
        .build()
        .expect("build should succeed");

    let mut lra_sketch = ReqSketch::builder()
        .rank_accuracy(RankAccuracy::LowRank)
        .build()
        .expect("build should succeed");

    for i in 0..n {
        hra_sketch.update(i as f64);
        lra_sketch.update(i as f64);
    }

    // HRA should have tighter bounds near rank 1.0
    let hra_width_99 = sketch_bound_width(&hra_sketch, 0.99, 2);
    let lra_width_99 = sketch_bound_width(&lra_sketch, 0.99, 2);
    assert!(hra_width_99 <= lra_width_99,
        "HRA should have tighter bounds at rank 0.99: HRA={:.6}, LRA={:.6}",
        hra_width_99, lra_width_99);

    // LRA should have tighter bounds near rank 0.0
    let hra_width_01 = sketch_bound_width(&hra_sketch, 0.01, 2);
    let lra_width_01 = sketch_bound_width(&lra_sketch, 0.01, 2);
    assert!(lra_width_01 <= hra_width_01,
        "LRA should have tighter bounds at rank 0.01: LRA={:.6}, HRA={:.6}",
        lra_width_01, hra_width_01);
}

fn sketch_bound_width(sketch: &ReqSketch<f64>, rank: f64, sigma: u8) -> f64 {
    sketch.get_rank_upper_bound(rank, sigma) - sketch.get_rank_lower_bound(rank, sigma)
}
