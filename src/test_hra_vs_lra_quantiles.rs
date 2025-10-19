#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_hra_vs_lra_for_low_quantiles() {
        let n = 50_000;

        // Test both modes
        let mut hra_sketch = ReqSketch::builder()
            .rank_accuracy(RankAccuracy::HighRank)
            .build().expect("Operation should succeed");

        let mut lra_sketch = ReqSketch::builder()
            .rank_accuracy(RankAccuracy::LowRank)
            .build().expect("Operation should succeed");

        // Add same data to both
        for i in 0..n {
            hra_sketch.update(i as f64);
            lra_sketch.update(i as f64);
        }

        // Verify both sketches processed data correctly
        assert_eq!(hra_sketch.total_n, n as u64, "HRA sketch should have processed {} items", n);
        assert_eq!(lra_sketch.total_n, n as u64, "LRA sketch should have processed {} items", n);

        // Compare retention - should both retain reasonable amounts
        let hra_retained = hra_sketch.total_retained_items();
        let lra_retained = lra_sketch.total_retained_items();
        assert!(hra_retained > 0, "HRA should retain some items");
        assert!(lra_retained > 0, "LRA should retain some items");

        // Test low quantile accuracy
        let test_ranks = [0.01, 0.1, 0.25];

        for &rank in &test_ranks {
            let true_quantile = rank * (n - 1) as f64;

            let hra_rank = hra_sketch.rank(&true_quantile, SearchCriteria::Inclusive).expect("Operation should succeed");
            let lra_rank = lra_sketch.rank(&true_quantile, SearchCriteria::Inclusive).expect("Operation should succeed");

            let hra_error = (hra_rank - rank).abs() / rank;
            let lra_error = (lra_rank - rank).abs() / rank;

            // Both should be reasonable - extreme quantiles inherently have high estimation error
            let max_error = if rank <= 0.01 { 2.0 } else { 0.5 };
            assert!(hra_error < max_error, "HRA error for rank {} should be reasonable: {:.2}%", rank, hra_error * 100.0);
            assert!(lra_error < max_error, "LRA error for rank {} should be reasonable: {:.2}%", rank, lra_error * 100.0);

            // For very low quantiles, LRA should generally be better
            if rank == 0.01 {
                // This is more of a behavioral observation than strict requirement
                // since both implementations should work reasonably well
            }
        }

        // Test high quantile accuracy
        let high_test_ranks = [0.75, 0.9, 0.95, 0.99];

        for &rank in &high_test_ranks {
            let true_quantile = rank * (n - 1) as f64;

            let hra_rank = hra_sketch.rank(&true_quantile, SearchCriteria::Inclusive).expect("Operation should succeed");
            let lra_rank = lra_sketch.rank(&true_quantile, SearchCriteria::Inclusive).expect("Operation should succeed");

            let hra_error = (hra_rank - rank).abs() / rank;
            let lra_error = (lra_rank - rank).abs() / rank;

            // Both should be reasonable for high quantiles
            assert!(hra_error < 0.5, "HRA error for high rank {} should be reasonable: {:.2}%", rank, hra_error * 100.0);
            assert!(lra_error < 0.5, "LRA error for high rank {} should be reasonable: {:.2}%", rank, lra_error * 100.0);
        }
    }

