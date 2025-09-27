//! Basic usage example for REQ sketch

use reqsketch::{ReqSketch, SearchCriteria, RankAccuracy};
// use std::io::{self, Write};  // Not needed for this example

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("REQ Sketch Demo - Relative Error Quantiles");
    println!("==========================================\n");

    // Create a new sketch with default parameters
    let mut sketch = ReqSketch::new();
    println!("Created REQ sketch with k={}", sketch.k());

    // Generate some sample data
    println!("Adding 10,000 random values...");
    for i in 0..10_000 {
        let value = (i as f64) + (i as f64 * 0.1).sin() * 100.0; // Add some noise
        sketch.update(value);
    }

    println!("Sketch statistics:");
    println!("  Total items processed: {}", sketch.len());
    println!("  Items retained: {}", sketch.num_retained());
    println!("  Estimation mode: {}", sketch.is_estimation_mode());
    println!("  Min value: {:?}", sketch.min_item());
    println!("  Max value: {:?}", sketch.max_item());
    println!();

    // Query some quantiles
    println!("Quantile queries:");
    let quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0];

    for &q in &quantiles {
        let value = sketch.quantile(q, SearchCriteria::Inclusive)?;
        println!("  {:.0}th percentile: {:.2}", q * 100.0, value);
    }
    println!();

    // Query some ranks
    println!("Rank queries:");
    let test_values = [1000.0, 5000.0, 9000.0];

    for &value in &test_values {
        let rank = sketch.rank(&value, SearchCriteria::Inclusive)?;
        println!("  Rank of {:.0}: {:.4} ({:.1}%)", value, rank, rank * 100.0);
    }
    println!();

    // Demonstrate PMF (Probability Mass Function)
    println!("Probability Mass Function:");
    let split_points = [2000.0, 5000.0, 8000.0];
    let pmf = sketch.pmf(&split_points, SearchCriteria::Inclusive)?;

    println!("  Intervals and their probabilities:");
    println!("    (-∞, {:.0}]: {:.4}", split_points[0], pmf[0]);
    for i in 1..split_points.len() {
        println!("    ({:.0}, {:.0}]: {:.4}", split_points[i-1], split_points[i], pmf[i]);
    }
    println!("    ({:.0}, ∞): {:.4}", split_points[split_points.len()-1], pmf[pmf.len()-1]);
    println!();

    // Demonstrate different accuracy modes
    println!("Comparing accuracy modes:");

    let high_rank_sketch: ReqSketch<f64> = ReqSketch::builder()
        .k(16)?
        .rank_accuracy(RankAccuracy::HighRank)
        .build()?;

    let low_rank_sketch: ReqSketch<f64> = ReqSketch::builder()
        .k(16)?
        .rank_accuracy(RankAccuracy::LowRank)
        .build()?;

    println!("  High rank accuracy mode optimizes for accuracy near rank 1.0");
    println!("  Low rank accuracy mode optimizes for accuracy near rank 0.0");
    println!();

    // Demonstrate sketch merging
    println!("Sketch merging demonstration:");
    let mut sketch1 = ReqSketch::new();
    let mut sketch2 = ReqSketch::new();

    // Add different ranges to each sketch
    for i in 0..1000 {
        sketch1.update(i as f64);
        sketch2.update((i + 1000) as f64);
    }

    println!("  Sketch 1: {} items, median = {:.2}",
             sketch1.len(),
             sketch1.quantile(0.5, SearchCriteria::Inclusive)?);
    println!("  Sketch 2: {} items, median = {:.2}",
             sketch2.len(),
             sketch2.quantile(0.5, SearchCriteria::Inclusive)?);

    sketch1.merge(&sketch2)?;
    println!("  Merged sketch: {} items, median = {:.2}",
             sketch1.len(),
             sketch1.quantile(0.5, SearchCriteria::Inclusive)?);

    println!("\nDemo completed successfully!");
    Ok(())
}