use reqsketch::{ReqSketch, RankAccuracy, SearchCriteria};

/// Detailed 3-Sigma compliance analysis matching C++ implementation
/// This test provides comprehensive comparison data to identify gaps

#[test]
fn test_rust_3_sigma_detailed_analysis() {
    println!("=== Rust Detailed 3-Sigma Compliance Analysis ===");

    let mut sketch = ReqSketch::builder()
        .k(12).unwrap()
        .rank_accuracy(RankAccuracy::HighRank)
        .build()
        .unwrap();

    let n = 50_000;
    for i in 0..n {
        sketch.update(i as f64);
    }

    println!("Data: 50,000 uniform values [0, 49999]");
    println!("Sketch configuration: k=12, HRA mode");
    println!("Total weight: {}", sketch.len());
    println!("Is estimation mode: {}", sketch.is_estimation_mode());

    // Test comprehensive range of quantiles - EXACTLY matching C++
    let test_ranks = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999];

    println!("\n=== 3-Sigma Bounds Detailed Analysis ===");
    println!("Rank\t\tTrue Quantile\tEst Rank\tLower Bound\tUpper Bound\tWithin 3σ\tError %");
    println!("----\t\t-------------\t--------\t-----------\t-----------\t---------\t-------");

    let mut passed_3sigma = 0;
    let total_tests = test_ranks.len();

    for &rank in &test_ranks {
        let true_quantile = rank * (n - 1) as f64;
        let estimated_rank = sketch.rank(&true_quantile, SearchCriteria::Inclusive).unwrap();

        let lower_bound = sketch.get_rank_lower_bound(rank, 3);
        let upper_bound = sketch.get_rank_upper_bound(rank, 3);

        let within_3sigma = estimated_rank >= lower_bound && estimated_rank <= upper_bound;
        let error_pct = (estimated_rank - rank).abs() / rank * 100.0;

        if within_3sigma {
            passed_3sigma += 1;
        }

        println!("{:.6}\t\t{:.6}\t\t{:.6}\t{:.6}\t{:.6}\t{}\t\t{:.6}%",
                rank, true_quantile, estimated_rank, lower_bound, upper_bound,
                if within_3sigma { "PASS" } else { "FAIL" }, error_pct);
    }

    let pass_rate_3sigma = passed_3sigma as f64 / total_tests as f64 * 100.0;
    println!("\n=== Rust 3-Sigma Summary ===");
    println!("3-sigma pass rate: {:.6}% ({}/{} quantiles)", pass_rate_3sigma, passed_3sigma, total_tests);

    // Test 2-sigma for comparison
    println!("\n=== 2-Sigma Bounds for Comparison ===");
    println!("Rank\t\tLower Bound\tUpper Bound\tWithin 2σ");
    println!("----\t\t-----------\t-----------\t---------");

    let mut passed_2sigma = 0;
    for &rank in &test_ranks {
        let true_quantile = rank * (n - 1) as f64;
        let estimated_rank = sketch.rank(&true_quantile, SearchCriteria::Inclusive).unwrap();

        let lower_bound = sketch.get_rank_lower_bound(rank, 2);
        let upper_bound = sketch.get_rank_upper_bound(rank, 2);

        let within_2sigma = estimated_rank >= lower_bound && estimated_rank <= upper_bound;

        if within_2sigma {
            passed_2sigma += 1;
        }

        println!("{:.6}\t\t{:.6}\t{:.6}\t{}",
                rank, lower_bound, upper_bound,
                if within_2sigma { "PASS" } else { "FAIL" });
    }

    let pass_rate_2sigma = passed_2sigma as f64 / total_tests as f64 * 100.0;
    println!("\n2-sigma pass rate: {:.6}% ({}/{} quantiles)", pass_rate_2sigma, passed_2sigma, total_tests);

    // Detailed bounds analysis for key failing quantiles
    println!("\n=== Detailed Bounds Calculation for Key Quantiles ===");
    let focus_ranks = [0.01, 0.1, 0.99, 0.999];

    for &rank in &focus_ranks {
        println!("\nRank {} detailed analysis:", rank);
        let true_quantile = rank * (n - 1) as f64;
        let estimated_rank = sketch.rank(&true_quantile, SearchCriteria::Inclusive).unwrap();

        println!("  True quantile: {:.6}", true_quantile);
        println!("  Estimated rank: {:.6}", estimated_rank);
        println!("  Rank error: {:.6}", (estimated_rank - rank).abs());

        for sigma in 1..=3 {
            let lower = sketch.get_rank_lower_bound(rank, sigma);
            let upper = sketch.get_rank_upper_bound(rank, sigma);
            let within = estimated_rank >= lower && estimated_rank <= upper;

            println!("  {}-sigma bounds: [{:.6}, {:.6}] width={:.6} within={}",
                    sigma, lower, upper, upper - lower, if within { "YES" } else { "NO" });
        }
    }

    println!("\n=== Rust Implementation Characteristics ===");
    println!("Expected theoretical performance:");
    println!("- 1-sigma: ~68.3% compliance");
    println!("- 2-sigma: ~95.4% compliance");
    println!("- 3-sigma: ~99.7% compliance");
    println!("\nActual Rust performance:");
    println!("- 2-sigma: {:.6}% compliance", pass_rate_2sigma);
    println!("- 3-sigma: {:.6}% compliance", pass_rate_3sigma);

    // Comparison with C++ (from our previous run)
    println!("\n=== Comparison with C++ ===");
    println!("C++ 2-sigma: 100.0% | Rust 2-sigma: {:.1}% | Gap: {:.1}%",
            pass_rate_2sigma, 100.0 - pass_rate_2sigma);
    println!("C++ 3-sigma: 100.0% | Rust 3-sigma: {:.1}% | Gap: {:.1}%",
            pass_rate_3sigma, 100.0 - pass_rate_3sigma);

    // Flag critical failures for debugging
    if pass_rate_3sigma < 100.0 {
        println!("\n⚠️  CRITICAL: 3-sigma compliance below C++ standard");
        println!("   Investigation needed in bounds calculation algorithms");
    }
}