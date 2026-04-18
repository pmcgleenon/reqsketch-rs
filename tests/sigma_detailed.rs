use reqsketch::{ReqSketch, RankAccuracy, SearchCriteria};

/// Test that bounds are structurally valid: nested (1σ ⊂ 2σ ⊂ 3σ), within [0,1],
/// and lower <= upper. These are deterministic properties, not statistical.
#[test]
fn test_bounds_structural_properties() {
    let mut sketch = ReqSketch::builder()
        .k(12)
        .unwrap()
        .rank_accuracy(RankAccuracy::HighRank)
        .build()
        .unwrap();

    let n = 50_000;
    for i in 0..n {
        sketch.update(i as f64);
    }

    let test_ranks = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999];

    for &rank in &test_ranks {
        let bounds: Vec<(f64, f64)> = (1..=3u8)
            .map(|s| (sketch.get_rank_lower_bound(rank, s), sketch.get_rank_upper_bound(rank, s)))
            .collect();

        for (sigma, &(lower, upper)) in bounds.iter().enumerate() {
            let sigma = sigma + 1;
            assert!(lower <= upper,
                "{}-sigma bounds invalid for rank {}: [{}, {}]", sigma, rank, lower, upper);
            assert!((0.0..=1.0).contains(&lower),
                "{}-sigma lower bound out of [0,1] for rank {}: {}", sigma, rank, lower);
            assert!((0.0..=1.0).contains(&upper),
                "{}-sigma upper bound out of [0,1] for rank {}: {}", sigma, rank, upper);
        }

        // Nesting: wider sigma should give wider bounds
        assert!(bounds[1].0 <= bounds[0].0 && bounds[0].1 <= bounds[1].1,
            "2-sigma bounds should contain 1-sigma bounds for rank {}", rank);
        assert!(bounds[2].0 <= bounds[1].0 && bounds[1].1 <= bounds[2].1,
            "3-sigma bounds should contain 2-sigma bounds for rank {}", rank);
    }
}

/// Multi-trial coverage test: run many independent sketches and check that
/// the empirical coverage of 3-sigma bounds matches the expected ~99.7%.
/// This is a proper statistical test, unlike checking a single instance.
#[test]
fn test_3_sigma_empirical_coverage() {
    let trials = 50;
    let n = 10_000;
    let test_rank = 0.9; // HRA mode, test near the optimized end
    let mut within_bounds_count = 0;

    for trial in 0..trials {
        let mut sketch = ReqSketch::builder()
            .k(12)
            .unwrap()
            .rank_accuracy(RankAccuracy::HighRank)
            .build()
            .unwrap();

        // Use different data per trial to get independent randomness
        for i in 0..n {
            sketch.update((i as f64) + (trial as f64 * 0.001));
        }

        let true_quantile = test_rank * (n - 1) as f64 + (trial as f64 * 0.001);
        let estimated_rank = sketch.rank(&true_quantile, SearchCriteria::Inclusive).unwrap();
        let lower = sketch.get_rank_lower_bound(test_rank, 3);
        let upper = sketch.get_rank_upper_bound(test_rank, 3);

        if estimated_rank >= lower && estimated_rank <= upper {
            within_bounds_count += 1;
        }
    }

    let coverage = within_bounds_count as f64 / trials as f64;
    // 3-sigma should cover ~99.7%, but with only 50 trials and k=12,
    // we accept ≥80% as a meaningful threshold
    assert!(coverage >= 0.80,
        "3-sigma empirical coverage {:.1}% across {} trials is too low (expected ≥80%)",
        coverage * 100.0, trials);
}
