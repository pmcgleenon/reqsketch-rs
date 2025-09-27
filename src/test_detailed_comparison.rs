#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_detailed_rust_state() {
        let mut sketch = ReqSketch::new();
        let n = 50_000;

        // Add uniformly distributed values
        for i in 0..n {
            sketch.update(i as f64);
        }

        println!("=== Detailed Rust State ===");
        println!("Total weight: {}", sketch.len());
        println!("Is estimation mode: {}", sketch.is_estimation_mode());
        println!("Min/Max: {:?} / {:?}", sketch.min_item(), sketch.max_item());
        println!("Levels: {}", sketch.compactors.len());

        let retained: u32 = sketch.compactors.iter().map(|c| c.num_items()).sum();
        println!("Retained items: {}", retained);

        // Show level distribution
        println!("\nLevel distribution:");
        for (level, compactor) in sketch.compactors.iter().enumerate() {
            if compactor.num_items() > 0 {
                println!("  Level {}: {}/{} items (weight {})",
                    level, compactor.num_items(), compactor.nominal_capacity(), compactor.weight());
            }
        }

        // Test specific rank calculation that's failing
        let test_quantile = 4999.9;  // True quantile for rank 0.1
        let estimated_rank = sketch.rank(&test_quantile, SearchCriteria::Inclusive).unwrap();

        println!("\n=== Specific Test Case ===");
        println!("Query quantile: {}", test_quantile);
        println!("Estimated rank: {}", estimated_rank);
        println!("True rank: 0.1");
        println!("Error: {}", (estimated_rank - 0.1).abs());

        // Also test the quantile going the other way
        let estimated_quantile = sketch.quantile(0.1, SearchCriteria::Inclusive).unwrap();
        println!("\nReverse test:");
        println!("Query rank: 0.1");
        println!("Estimated quantile: {}", estimated_quantile);
        println!("True quantile: {}", test_quantile);
        println!("Error: {:.2}%", (estimated_quantile - test_quantile).abs() / test_quantile * 100.0);

        // Compare with C++ expected values
        println!("\n=== Comparison with C++ ===");
        println!("C++ estimated rank for quantile {}: 0.1024", test_quantile);
        println!("Rust estimated rank for quantile {}: {}", test_quantile, estimated_rank);
        println!("Difference: {:.4} ({:.1}% worse)",
                (estimated_rank - 0.1024).abs(),
                (estimated_rank - 0.1024).abs() / 0.0024 * 100.0);
    }
}