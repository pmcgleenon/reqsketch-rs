#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn debug_problematic_quantiles() {
        let mut sketch = ReqSketch::new();
        let n = 50_000;

        // Add uniformly distributed values
        for i in 0..n {
            sketch.update(i as f64);
        }

        println!("=== RUST DEBUG: Problematic Quantiles ===");

        // Focus on the problematic quantiles: 0.1, 0.25, 0.75
        let problem_ranks = [0.1, 0.25, 0.75];

        for &rank in &problem_ranks {
            let estimated = sketch.quantile(rank, SearchCriteria::Inclusive).unwrap();
            let true_val = rank * (n - 1) as f64;
            let error = (estimated - true_val).abs() / true_val * 100.0;

            println!("\nRank {}:", rank);
            println!("  Estimated: {}", estimated);
            println!("  True: {}", true_val);
            println!("  Error: {:.2}%", error);
            println!("  Internal sketch state:");
            println!("  Levels: {}", sketch.compactors.len());

            let retained: u32 = sketch.compactors.iter().map(|c| c.num_items()).sum();
            println!("  Retained items: {}", retained);

            // Show level distribution
            println!("  Level distribution:");
            for (level, compactor) in sketch.compactors.iter().enumerate() {
                if compactor.num_items() > 0 {
                    println!("    Level {}: {}/{} items (weight {})",
                        level, compactor.num_items(), compactor.nominal_capacity(), compactor.weight());
                }
            }
        }

        // Test if the issue is with SearchCriteria
        println!("\n=== SEARCH CRITERIA COMPARISON ===");
        let test_rank = 0.25;
        let inclusive = sketch.quantile(test_rank, SearchCriteria::Inclusive).unwrap();
        let exclusive = sketch.quantile(test_rank, SearchCriteria::Exclusive).unwrap();

        println!("Rank 0.25:");
        println!("  Inclusive: {}", inclusive);
        println!("  Exclusive: {}", exclusive);
        println!("  True: {}", test_rank * (n-1) as f64);
    }

    #[test]
    fn debug_1k_comparison() {
        let mut sketch = ReqSketch::new();
        let n = 1000;

        for i in 0..n {
            sketch.update(i as f64);
        }

        println!("Rust REQ sketch with 1000 items:");
        println!("Is estimation mode: {}", sketch.is_estimation_mode());
        println!("Total weight: {}", sketch.len());
        println!("Levels: {}", sketch.compactors.len());

        let retained: u32 = sketch.compactors.iter().map(|c| c.num_items()).sum();
        println!("Retained items: {}", retained);

        // Test the same quantiles as C++
        let q01 = sketch.quantile(0.1, SearchCriteria::Inclusive).unwrap();
        let q25 = sketch.quantile(0.25, SearchCriteria::Inclusive).unwrap();
        let q50 = sketch.quantile(0.5, SearchCriteria::Inclusive).unwrap();

        println!("\nQuantiles:");
        println!("0.1: {} (true: {})", q01, 0.1 * 999.0);
        println!("0.25: {} (true: {})", q25, 0.25 * 999.0);
        println!("0.5: {} (true: {})", q50, 0.5 * 999.0);

        // Show level distribution
        println!("\nLevel distribution:");
        for (level, compactor) in sketch.compactors.iter().enumerate() {
            if compactor.num_items() > 0 {
                println!("  Level {}: {}/{} items (weight {})",
                    level, compactor.num_items(), compactor.nominal_capacity(), compactor.weight());
            }
        }
    }

    #[test]
    fn debug_all_quantiles_50k() {
        let mut sketch = ReqSketch::new();
        let n = 50_000;

        // Add uniformly distributed values
        for i in 0..n {
            sketch.update(i as f64);
        }

        println!("Rust REQ sketch quantile accuracy test:");
        println!("Adding {} uniformly distributed values (0 to {})", n, n-1);
        println!("Is estimation mode: {}", sketch.is_estimation_mode());
        println!("Total weight: {}", sketch.len());

        // Test quantiles against known true values
        let test_ranks = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];

        println!("\nQuantile accuracy results:");
        println!("Rank\tEstimated\tTrue\t\tRel Error\tPass(<tolerance)");
        println!("----\t---------\t----\t\t---------\t-------------");

        for &rank in &test_ranks {
            let estimated_quantile = sketch.quantile(rank, SearchCriteria::Inclusive).unwrap();
            let true_quantile = rank * (n - 1) as f64;
            let relative_error = (estimated_quantile - true_quantile).abs() / true_quantile;

            let tolerance = if rank <= 0.1 || rank >= 0.9 {
                0.12  // 12% tolerance for extreme quantiles
            } else {
                0.02  // 2% tolerance for middle quantiles
            };

            let pass = relative_error < tolerance;

            println!("{}\t{}\t\t{}\t\t{:.2}%\t\t{} (<{:.0}%)",
                     rank, estimated_quantile, true_quantile,
                     relative_error * 100.0,
                     if pass { "PASS" } else { "FAIL" },
                     tolerance * 100.0);
        }
    }
}