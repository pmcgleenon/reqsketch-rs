#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_rank_space_error() -> Result<()> {
        let mut sketch = ReqSketch::new(); // HRA mode
        let n = 50_000;

        // Add uniformly distributed values
        for i in 0..n {
            sketch.update(i as f64);
        }

        // Verify sketch processed all items
        assert_eq!(sketch.total_n, n as u64, "Should have processed {} items", n);

        // Test rank-space error by round-trip: quantile -> rank
        let test_ranks = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];

        for &rank in &test_ranks {
            let q_est = sketch.quantile(rank, SearchCriteria::Inclusive)?;
            let est_rank = sketch.rank(&q_est, SearchCriteria::Inclusive)?;
            let abs_rank_err = (est_rank - rank).abs();

            // HRA: tighter bounds for higher ranks where HRA should excel
            let max_abs_rank_err = if rank >= 0.9 {
                0.01  // 1% absolute rank error for HRA's strong zone
            } else {
                0.02  // looser for middle/low ranks
            };

            assert!(abs_rank_err <= max_abs_rank_err,
                   "HRA rank {} abs error {:.4} > {:.4} (est_rank: {:.6}, true_rank: {:.6})",
                   rank, abs_rank_err, max_abs_rank_err, est_rank, rank);
        }

        // Verify sketch structure
        assert!(sketch.len() <= n as u64, "Should not retain more items than inserted");
        assert!(sketch.compactors.len() > 0, "Should have at least one compactor");
        Ok(())
    }
}