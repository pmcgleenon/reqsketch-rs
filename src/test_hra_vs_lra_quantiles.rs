#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_hra_vs_lra_for_low_quantiles() {
        println!("=== HRA vs LRA Low Quantile Test ===");
        println!("Testing if LRA mode fixes our 0.01 quantile problem");
        println!("");

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

        println!("=== Internal State Comparison ===");

        // Compare retention
        let hra_retained = hra_sketch.total_retained_items();
        let lra_retained = lra_sketch.total_retained_items();
        println!("HRA retained items: {}", hra_retained);
        println!("LRA retained items: {}", lra_retained);

        // Check first few items in sorted views
        let hra_sorted = hra_sketch.test_get_sorted_view().expect("Operation should succeed");
        let lra_sorted = lra_sketch.test_get_sorted_view().expect("Operation should succeed");

        println!("\nFirst 5 items:");
        println!("HRA: {:?}", hra_sorted.iter().take(5).collect::<Vec<_>>());
        println!("LRA: {:?}", lra_sorted.iter().take(5).collect::<Vec<_>>());

        println!("\n=== Low Quantile Accuracy Comparison ===");

        let test_ranks = [0.01, 0.1, 0.25];

        for &rank in &test_ranks {
            let true_quantile = rank * (n - 1) as f64;

            let hra_rank = hra_sketch.rank(&true_quantile, SearchCriteria::Inclusive).expect("Operation should succeed");
            let lra_rank = lra_sketch.rank(&true_quantile, SearchCriteria::Inclusive).expect("Operation should succeed");

            let hra_error = (hra_rank - rank).abs() / rank * 100.0;
            let lra_error = (lra_rank - rank).abs() / rank * 100.0;

            println!("Rank {:.2}:", rank);
            println!("  HRA: {:.6} (error: {:.1}%)", hra_rank, hra_error);
            println!("  LRA: {:.6} (error: {:.1}%)", lra_rank, lra_error);

            if rank == 0.01 {
                if lra_error < hra_error {
                    println!("  ✅ LRA is better for low quantiles!");
                } else {
                    println!("  ❌ LRA is not better - different issue");
                }
            }
        }

        println!("\n=== High Quantile Accuracy Comparison ===");

        let high_test_ranks = [0.75, 0.9, 0.95, 0.99];

        for &rank in &high_test_ranks {
            let true_quantile = rank * (n - 1) as f64;

            let hra_rank = hra_sketch.rank(&true_quantile, SearchCriteria::Inclusive).expect("Operation should succeed");
            let lra_rank = lra_sketch.rank(&true_quantile, SearchCriteria::Inclusive).expect("Operation should succeed");

            let hra_error = (hra_rank - rank).abs() / rank * 100.0;
            let lra_error = (lra_rank - rank).abs() / rank * 100.0;

            println!("Rank {:.2}:", rank);
            println!("  HRA: {:.6} (error: {:.1}%)", hra_rank, hra_error);
            println!("  LRA: {:.6} (error: {:.1}%)", lra_rank, lra_error);

            if hra_error < lra_error {
                println!("  ✅ HRA is better for high quantiles as expected");
            } else {
                println!("  ⚠️ HRA not better - may need investigation");
            }
        }
    }

    #[test]
    fn test_cpp_hra_mode_validation() {
        println!("=== Validate C++ uses HRA by default ===");

        // C++ defaults to HRA, so let's compare our HRA with C++ results
        let mut sketch = ReqSketch::builder()
            .rank_accuracy(RankAccuracy::HighRank)  // Same as C++ default
            .build().expect("Operation should succeed");

        let n = 50_000;
        for i in 0..n {
            sketch.update(i as f64);
        }

        println!("Our HRA implementation details:");
        println!("Total retained: {}", sketch.total_retained_items());

        // Test the 0.01 case that C++ handles well
        let target_rank = 0.01;
        let true_quantile = target_rank * (n - 1) as f64;
        let estimated_rank = sketch.rank(&true_quantile, SearchCriteria::Inclusive).expect("Operation should succeed");
        let error = (estimated_rank - target_rank).abs() / target_rank * 100.0;

        println!("0.01 quantile test:");
        println!("  Our HRA: {:.6} (error: {:.1}%)", estimated_rank, error);
        println!("  C++ HRA: 0.010240 (error: 2.4%)");

        if error > 50.0 {
            println!("  ❌ Still much worse than C++ - need deeper investigation");
            println!("  This suggests the issue is not just HRA/LRA mode");
        } else {
            println!("  ✅ Much better - HRA/LRA was the issue");
        }
    }
}