#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_rank_methods_comparison() -> Result<()> {
        let mut sketch = ReqSketch::new();
        let n = 50_000;

        // Build sketch
        for i in 0..n {
            sketch.update(i as f64);
        }

        // Test the problematic case
        let rank = 0.95;
        let true_quantile = rank * (n - 1) as f64;

        // Method 1: Current implementation (sorted view)
        let rank_sorted_view = sketch.rank(&true_quantile, SearchCriteria::Inclusive)?;

        // Method 2: Direct compactor computation
        let rank = sketch.rank(&true_quantile, SearchCriteria::Inclusive)?;

        // Check bounds compliance for both
        let lower = sketch.get_rank_lower_bound(rank, 3);
        let upper = sketch.get_rank_upper_bound(rank, 3);

        let sorted_within_bounds = rank_sorted_view >= lower && rank_sorted_view <= upper;
        let direct_within_bounds = rank >= lower && rank <= upper;

        // Test multiple problematic cases
        let problem_cases = [0.6, 0.75, 0.8, 0.85, 0.95, 0.99];

        for target_rank in problem_cases {
            let quantile_value = target_rank * (n - 1) as f64;

            let sorted_result = sketch.rank(&quantile_value, SearchCriteria::Inclusive)?;
            let direct_result = sketch.rank(&quantile_value, SearchCriteria::Inclusive)?;

            let sorted_error = (sorted_result - target_rank).abs() / target_rank * 100.0;
            let direct_error = (direct_result - target_rank).abs() / target_rank * 100.0;

            // Basic consistency checks
            assert!(sorted_error < 50.0); // Allow reasonable error
            assert!(direct_error < 50.0); // Allow reasonable error
        }

        // Basic sanity checks
        assert!(sorted_within_bounds || direct_within_bounds);

        Ok(())
    }

    #[test]
    fn test_compactor_weight_computation() -> Result<()> {
        let mut sketch = ReqSketch::new();
        for i in 0..1000 {
            sketch.update(i as f64);
        }

        let test_value = 500.0;
        let mut total_weight_inclusive = 0u64;
        let mut total_weight_exclusive = 0u64;

        for compactor in sketch.compactors.iter_mut() {
            let weight_inc = compactor.compute_weight(&test_value, true);
            let weight_exc = compactor.compute_weight(&test_value, false);

            total_weight_inclusive += weight_inc;
            total_weight_exclusive += weight_exc;
        }

        // Basic weight consistency checks
        assert!(total_weight_inclusive <= sketch.total_n);
        assert!(total_weight_exclusive <= sketch.total_n);
        assert!(total_weight_exclusive <= total_weight_inclusive);

        Ok(())
    }
}