#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_basic_sketch_correctness() {
        let mut sketch = ReqSketch::new();
        let n = 1000;

        for i in 0..n {
            sketch.update(i as f64);

            // Basic invariants
            assert_eq!(sketch.total_n, (i + 1) as u64);
            assert!(sketch.len() > 0);
            assert!(sketch.compactors.len() > 0);
        }
    }

    #[test]
    fn test_monotonic_quantiles() -> Result<()> {
        let mut sketch = ReqSketch::new();
        for i in 0..10_000 {
            sketch.update(i as f64);
        }

        // Quantiles should be monotonically increasing
        let ranks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let mut prev_quantile = 0.0;

        for &rank in &ranks {
            let quantile = sketch.quantile(rank, SearchCriteria::Inclusive)?;
            assert!(quantile >= prev_quantile);
            prev_quantile = quantile;
        }

        Ok(())
    }

    #[test]
    fn test_rank_quantile_consistency() -> Result<()> {
        let mut sketch = ReqSketch::new();
        for i in 0..10_000 {
            sketch.update(i as f64);
        }

        // For any value, rank(quantile(r)) should be approximately r
        let test_ranks = [0.1, 0.25, 0.5, 0.75, 0.9];

        for &rank in &test_ranks {
            let quantile = sketch.quantile(rank, SearchCriteria::Inclusive)?;
            let estimated_rank = sketch.rank(&quantile, SearchCriteria::Inclusive)?;

            let error = (estimated_rank - rank).abs() / rank;
            assert!(error < 0.2);
        }

        Ok(())
    }

    #[test]
    fn test_bounds_sanity_checks() {
        let mut sketch = ReqSketch::new();
        for i in 0..10_000 {
            sketch.update(i as f64);
        }

        let test_ranks = [0.1, 0.5, 0.9];
        let sigmas = [1, 2, 3];

        for &rank in &test_ranks {
            for &sigma in &sigmas {
                let lower = sketch.get_rank_lower_bound(rank, sigma);
                let upper = sketch.get_rank_upper_bound(rank, sigma);

                // Basic sanity checks
                assert!(lower <= upper);
                assert!(lower >= 0.0 && lower <= 1.0);
                assert!(upper >= 0.0 && upper <= 1.0);
                assert!(lower <= rank && rank <= upper);
            }
        }
    }

    #[test]
    fn test_edge_cases() -> Result<()> {
        // Single value
        let mut sketch1 = ReqSketch::new();
        sketch1.update(42.0);

        let q = sketch1.quantile(0.5, SearchCriteria::Inclusive)?;
        assert_eq!(q, 42.0);

        // Two values
        let mut sketch2 = ReqSketch::new();
        sketch2.update(1.0);
        sketch2.update(100.0);

        let median = sketch2.quantile(0.5, SearchCriteria::Inclusive)?;
        assert!(median >= 1.0 && median <= 100.0);

        // Duplicates
        let mut sketch3 = ReqSketch::new();
        for _i in 0..100 {
            sketch3.update(42.0);
        }

        let median_dup = sketch3.quantile(0.5, SearchCriteria::Inclusive)?;
        assert_eq!(median_dup, 42.0);

        Ok(())
    }

    #[test]
    fn test_randomness_is_bounded() -> Result<()> {
        // REQ sketches are intentionally non-deterministic, but the randomness should be bounded
        // Multiple runs should still maintain correctness properties
        let n = 1000;
        let mut results = Vec::new();

        // Run multiple trials
        for _trial in 0..5 {
            let mut sketch = ReqSketch::new();
            for i in 0..n {
                sketch.update(i as f64);
            }

            let median = sketch.quantile(0.5, SearchCriteria::Inclusive)?;
            results.push(median);
        }

        // All medians should be in a reasonable range around the true median
        let true_median = (n - 1) as f64 * 0.5;
        for &median in results.iter() {
            let error = (median - true_median).abs() / true_median;
            assert!(error < 0.2);
        }

        Ok(())
    }

    #[test]
    fn test_search_criteria_consistency() -> Result<()> {
        let mut sketch = ReqSketch::new();
        for i in 0..1000 {
            sketch.update(i as f64);
        }

        let test_values = [100.0, 250.0, 500.0, 750.0];

        for &val in &test_values {
            let inclusive_rank = sketch.rank(&val, SearchCriteria::Inclusive)?;
            let exclusive_rank = sketch.rank(&val, SearchCriteria::Exclusive)?;

            // Exclusive should be ≤ Inclusive
            assert!(exclusive_rank <= inclusive_rank);

            // Both should be in [0, 1]
            assert!(inclusive_rank >= 0.0 && inclusive_rank <= 1.0);
            assert!(exclusive_rank >= 0.0 && exclusive_rank <= 1.0);
        }

        Ok(())
    }
}