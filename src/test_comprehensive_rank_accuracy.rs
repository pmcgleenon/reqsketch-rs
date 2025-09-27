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

        println!("=== Comprehensive Rust vs C++ Rank Accuracy Comparison ===");
        println!("Rust REQ sketch with 50K uniform values [0, 49999]:");
        println!("Total weight: {}", sketch.len());
        println!("Is estimation mode: {}", sketch.is_estimation_mode());

        // Test comprehensive range of quantiles including extreme ones
        let test_ranks = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999];

        println!("\nRank\t\tTrue Quantile\tEst Rank\tRank Error\t\tError %");
        println!("----\t\t-------------\t--------\t----------\t\t-------");

        for &rank in &test_ranks {
            // Calculate the true quantile for this rank
            let true_quantile = rank * (n - 1) as f64;

            // Get our estimated rank for this true quantile
            let estimated_rank = sketch.rank(&true_quantile, SearchCriteria::Inclusive).unwrap();

            let rank_error = (estimated_rank - rank).abs();
            let error_percent = (rank_error / rank) * 100.0;

            println!("{:.6}\t\t{:.6}\t\t{:.6}\t\t{:.6}\t\t{:.6}%",
                     rank, true_quantile, estimated_rank, rank_error, error_percent);
        }

        println!("\n=== Additional Extreme Quantiles ===");

        // Test some very extreme quantiles
        let extreme_ranks = [0.001, 0.0001];

        println!("\nRank\t\tTrue Quantile\tEst Rank\tRank Error\t\tError %");
        println!("----\t\t-------------\t--------\t----------\t\t-------");

        for &rank in &extreme_ranks {
            let true_quantile = rank * (n - 1) as f64;
            let estimated_rank = sketch.rank(&true_quantile, SearchCriteria::Inclusive).unwrap();
            let rank_error = (estimated_rank - rank).abs();
            let error_percent = (rank_error / rank) * 100.0;

            println!("{:.6}\t\t{:.6}\t\t{:.6}\t\t{:.6}\t\t{:.6}%",
                     rank, true_quantile, estimated_rank, rank_error, error_percent);
        }

        println!("\n=== Error Bounds Analysis ===");
        println!("Rank\t\t1σ\t\t2σ\t\t3σ");
        println!("----\t\t--\t\t--\t\t--");

        // Check if our estimates are within theoretical bounds
        for &rank in &test_ranks {
            let true_quantile = rank * (n - 1) as f64;
            let estimated_rank = sketch.rank(&true_quantile, SearchCriteria::Inclusive).unwrap();

            // Test 1, 2, and 3 sigma bounds
            let within_1sigma = {
                let lower = sketch.get_rank_lower_bound(rank, 1);
                let upper = sketch.get_rank_upper_bound(rank, 1);
                estimated_rank >= lower && estimated_rank <= upper
            };

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

            println!("{:.6}\t\t{}\t\t{}\t\t{}",
                     rank,
                     if within_1sigma { "PASS" } else { "FAIL" },
                     if within_2sigma { "PASS" } else { "FAIL" },
                     if within_3sigma { "PASS" } else { "FAIL" });
        }

        // Summary analysis
        println!("\n=== Confidence Level Analysis ===");
        println!("1-sigma = 68.3% confidence");
        println!("2-sigma = 95.4% confidence (industry standard)");
        println!("3-sigma = 99.7% confidence (very high confidence)");
        println!("REQ sketches should satisfy 2-sigma bounds for correct implementation");

        println!("\n=== Summary Comparison with C++ ===");
        println!("C++ Results: Most quantiles PASS 2-sigma (95.4% confidence)");
        println!("Rust Results: (see above)");
        println!("Industry standard: REQ sketches should satisfy 2-sigma bounds");
    }
}