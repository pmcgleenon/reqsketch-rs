#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_hra_rank_accuracy() {
        let mut sketch = ReqSketch::new(); // HRA mode
        let n = 50_000;

        // Add uniformly distributed values
        for i in 0..n {
            sketch.update(i as f64);
        }

        // Verify basic sketch properties
        assert_eq!(sketch.total_n, n as u64);
        assert!(sketch.len() <= n as u64);

        // Focus on high ranks for HRA - this is where HRA should be most accurate
        let test_ranks = [0.5, 0.9, 0.95, 0.99, 0.999];

        for &rank in &test_ranks {
            let true_quantile = rank * (n - 1) as f64;
            let estimated_rank = sketch.rank(&true_quantile, SearchCriteria::Inclusive).unwrap();
            let abs_error = (estimated_rank - rank).abs();

            // Use absolute error thresholds appropriate for HRA
            // Tighter bounds for higher ranks where HRA should excel
            let max_abs_error = if rank >= 0.99 {
                0.005  // Very tight for top 1%
            } else if rank >= 0.9 {
                0.01   // Reasonable for top 10%
            } else {
                0.02   // Looser for middle ranks
            };

            assert!(abs_error <= max_abs_error);
        }

        // Test that bounds functions are self-consistent for high ranks
        for &rank in &[0.9, 0.99, 0.999] {
            let lb3 = sketch.get_rank_lower_bound(rank, 3);
            let ub3 = sketch.get_rank_upper_bound(rank, 3);

            // For uniform data, the true rank should lie within the bounds
            assert!(rank >= lb3 && rank <= ub3);
        }

        // Verify overall sketch health
        assert!(sketch.compactors.len() > 0);
        assert!(sketch.total_n > 0);
    }
}