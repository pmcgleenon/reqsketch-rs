use reqsketch::{ReqSketch, RankAccuracy, SearchCriteria};

/// Test 3-Sigma compliance for rank error bounds
/// Validates that rank estimates fall within expected statistical bounds
#[test]
fn test_3_sigma_compliance() {
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

    // Test comprehensive range of quantiles
    let test_ranks = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999];

    let mut passed_3sigma = 0;
    let mut passed_2sigma = 0;
    let total_tests = test_ranks.len();

    for &rank in &test_ranks {
        let true_quantile = rank * (n - 1) as f64;
        let estimated_rank = sketch.rank(&true_quantile, SearchCriteria::Inclusive).unwrap();

        // Test 3-sigma bounds
        let lower_bound_3 = sketch.get_rank_lower_bound(rank, 3);
        let upper_bound_3 = sketch.get_rank_upper_bound(rank, 3);
        let within_3sigma = estimated_rank >= lower_bound_3 && estimated_rank <= upper_bound_3;

        // Test 2-sigma bounds
        let lower_bound_2 = sketch.get_rank_lower_bound(rank, 2);
        let upper_bound_2 = sketch.get_rank_upper_bound(rank, 2);
        let within_2sigma = estimated_rank >= lower_bound_2 && estimated_rank <= upper_bound_2;

        if within_3sigma {
            passed_3sigma += 1;
        }
        if within_2sigma {
            passed_2sigma += 1;
        }

        // Bounds should be valid (lower <= upper)
        assert!(lower_bound_3 <= upper_bound_3,
            "3-sigma bounds invalid for rank {}: lower={}, upper={}",
            rank, lower_bound_3, upper_bound_3);
        assert!(lower_bound_2 <= upper_bound_2,
            "2-sigma bounds invalid for rank {}: lower={}, upper={}",
            rank, lower_bound_2, upper_bound_2);

        // Error should be reasonable (less than 50% for most quantiles)
        let error_pct = (estimated_rank - rank).abs() / rank * 100.0;
        if (0.1..=0.9).contains(&rank) {
            assert!(error_pct < 50.0,
                "Excessive error for rank {}: {:.2}%", rank, error_pct);
        }
    }

    let pass_rate_3sigma = passed_3sigma as f64 / total_tests as f64 * 100.0;
    let pass_rate_2sigma = passed_2sigma as f64 / total_tests as f64 * 100.0;

    // Statistical bounds should generally work
    assert!(pass_rate_2sigma >= 70.0,
        "2-sigma pass rate too low: {:.1}% (expected ≥70%)", pass_rate_2sigma);
    assert!(pass_rate_3sigma >= 80.0,
        "3-sigma pass rate too low: {:.1}% (expected ≥80%)", pass_rate_3sigma);

    // 3-sigma should perform better than 2-sigma
    assert!(pass_rate_3sigma >= pass_rate_2sigma,
        "3-sigma pass rate ({:.1}%) should be ≥ 2-sigma pass rate ({:.1}%)",
        pass_rate_3sigma, pass_rate_2sigma);
}

/// Test bounds consistency across different sigma levels
#[test]
fn test_bounds_consistency() {
    let mut sketch = ReqSketch::builder()
        .k(12)
        .unwrap()
        .rank_accuracy(RankAccuracy::HighRank)
        .build()
        .unwrap();

    for i in 0..1000 {
        sketch.update(i as f64);
    }

    let test_ranks = [0.01, 0.1, 0.5, 0.9, 0.99];

    for &rank in &test_ranks {
        let bounds_1 = (sketch.get_rank_lower_bound(rank, 1), sketch.get_rank_upper_bound(rank, 1));
        let bounds_2 = (sketch.get_rank_lower_bound(rank, 2), sketch.get_rank_upper_bound(rank, 2));
        let bounds_3 = (sketch.get_rank_lower_bound(rank, 3), sketch.get_rank_upper_bound(rank, 3));

        // Higher sigma should give wider bounds
        assert!(bounds_2.0 <= bounds_1.0 && bounds_1.1 <= bounds_2.1,
            "2-sigma bounds should contain 1-sigma bounds for rank {}", rank);
        assert!(bounds_3.0 <= bounds_2.0 && bounds_2.1 <= bounds_3.1,
            "3-sigma bounds should contain 2-sigma bounds for rank {}", rank);

        // All bounds should be within [0,1]
        for sigma in 1..=3 {
            let lower = sketch.get_rank_lower_bound(rank, sigma);
            let upper = sketch.get_rank_upper_bound(rank, sigma);
            assert!((0.0..=1.0).contains(&lower),
                "{}-sigma lower bound out of range for rank {}: {}", sigma, rank, lower);
            assert!((0.0..=1.0).contains(&upper),
                "{}-sigma upper bound out of range for rank {}: {}", sigma, rank, upper);
        }
    }
}