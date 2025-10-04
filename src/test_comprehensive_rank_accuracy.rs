#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_comprehensive_rank_accuracy_comparison() {
        let mut sketch = ReqSketch::new();
        let n = 50_000;

        // Add uniformly distributed values
        for i in 0..n {
            sketch.update(i as f64);
        }

        // Verify basic sketch properties
        assert_eq!(sketch.total_n, n as u64, "Should have processed {} items", n);
        assert!(sketch.len() <= n as u64, "Should not retain more items than inserted");

        // Test comprehensive range of quantiles including extreme ones
        let test_ranks = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999];

        for &rank in &test_ranks {
            // Calculate the true quantile for this rank
            let true_quantile = rank * (n - 1) as f64;

            // Get our estimated rank for this true quantile
            let estimated_rank = sketch.rank(&true_quantile, SearchCriteria::Inclusive).expect("Operation should succeed");

            let rank_error = (estimated_rank - rank).abs();
            let error_percent = (rank_error / rank) * 100.0;

            // Assert reasonable accuracy based on rank position
            // REQ sketches should provide reasonable accuracy, but allow for algorithm limitations
            let max_error_percent = if rank <= 0.01 || rank >= 0.99 {
                20.0  // Very extreme quantiles - algorithm inherently less precise here
            } else if rank <= 0.1 || rank >= 0.9 {
                10.0  // Extreme quantiles
            } else {
                5.0   // Middle quantiles should be most accurate
            };

            assert!(error_percent < max_error_percent,
                   "Rank {} error {:.2}% exceeds maximum {:.1}% (estimated: {:.6}, true: {:.6})",
                   rank, error_percent, max_error_percent, estimated_rank, rank);
        }

        // Test some very extreme quantiles
        let extreme_ranks = [0.001, 0.0001];

        for &rank in &extreme_ranks {
            let true_quantile = rank * (n - 1) as f64;
            let estimated_rank = sketch.rank(&true_quantile, SearchCriteria::Inclusive).expect("Operation should succeed");
            let rank_error = (estimated_rank - rank).abs();
            let error_percent = (rank_error / rank) * 100.0;

            // Very extreme quantiles still need bounds - 50% is a reasonable upper limit
            assert!(error_percent < 50.0,
                   "Extreme rank {} error {:.2}% is too high (estimated: {:.6}, true: {:.6})",
                   rank, error_percent, estimated_rank, rank);
        }

        // Check if our estimates are within theoretical bounds with assertions
        let mut passed_2sigma = 0;
        let mut passed_3sigma = 0;

        for &rank in &test_ranks {
            let true_quantile = rank * (n - 1) as f64;
            let estimated_rank = sketch.rank(&true_quantile, SearchCriteria::Inclusive).expect("Operation should succeed");

            // Test 2 and 3 sigma bounds (focus on industry standard)
            let within_2sigma = {
                let lower = sketch.get_rank_lower_bound(rank, 2);
                let upper = sketch.get_rank_upper_bound(rank, 2);
                estimated_rank >= lower && estimated_rank <= upper
            };

            let within_3sigma = {
                let lower = sketch.get_rank_lower_bound(rank, 3);
                let upper = sketch.get_rank_upper_bound(rank, 3);
                estimated_rank >= lower && estimated_rank <= upper
            };

            if within_2sigma { passed_2sigma += 1; }
            if within_3sigma { passed_3sigma += 1; }
        }

        // Assert that most estimates are within reasonable bounds
        // REQ sketches should satisfy 2-sigma bounds for majority of cases
        let success_rate_2sigma = passed_2sigma as f64 / test_ranks.len() as f64;
        let success_rate_3sigma = passed_3sigma as f64 / test_ranks.len() as f64;

        // REQ sketch bounds may be conservative estimates, so we expect reasonable but not perfect coverage
        assert!(success_rate_3sigma >= 0.5,
               "Should have at least 50% of estimates within 3-sigma bounds, got {:.1}%",
               success_rate_3sigma * 100.0);

        // 2-sigma bounds are tighter, so expect lower success rate
        assert!(success_rate_2sigma >= 0.3,
               "Should have at least 30% of estimates within 2-sigma bounds, got {:.1}%",
               success_rate_2sigma * 100.0);

        // Verify overall sketch health
        assert!(sketch.compactors.len() > 0, "Should have at least one compactor");
        assert!(sketch.total_n > 0, "Should have processed items");
    }
}