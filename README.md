# reqsketch-rs 

An implementation of the **Relative Error Quantiles (REQ) sketch** algorithm in Rust.

## Overview

REQ sketch is a probabilistic data structure for approximate quantile estimation with relative error guarantees, particularly useful for streaming scenarios where you need to estimate quantiles over large data streams with bounded memory usage.

This implementation is based on the paper ["Relative Error Streaming Quantiles"](https://arxiv.org/abs/2004.01668) by Graham Cormode, Zohar Karnin, Edo Liberty, Justin Thaler, and Pavel Veselý.   A lot of inspiration was taken from the C++ implementation in Apache DataSketches https://datasketches.apache.org/docs/REQ/ReqSketch.html

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
reqsketch = "0.1"
```

### Basic Usage

```rust
use reqsketch::{ReqSketch, SearchCriteria};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new sketch
    let mut sketch = ReqSketch::new();

    // Add values to the sketch
    for i in 0..10000 {
        sketch.update(i as f64);
    }

    // Query quantiles
    let median = sketch.quantile(0.5, SearchCriteria::Inclusive)?;
    let p99 = sketch.quantile(0.99, SearchCriteria::Inclusive)?;

    println!("Median: {:.2}", median);
    println!("99th percentile: {:.2}", p99);

    // Query ranks
    let rank = sketch.rank(&5000.0, SearchCriteria::Inclusive)?;
    println!("Rank of 5000: {:.4}", rank);

    Ok(())
}
```

### Sketch Merging

```rust
let mut sketch1 = ReqSketch::new();
let mut sketch2 = ReqSketch::new();

// Add data to both sketches
for i in 0..1000 {
    sketch1.update(i as f64);
    sketch2.update((i + 1000) as f64);
}

// Merge sketch2 into sketch1
sketch1.merge(&sketch2)?;
```

## Examples

Run the examples to see the sketch in action:

```bash
cargo run --example basic_usage
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Licensed under the MIT or Apache License.

## References

- [Relative Error Streaming Quantiles Paper](https://arxiv.org/abs/2004.01668)
- [Apache DataSketches Project](https://datasketches.apache.org/)
- [DataSketches REQ Sketch Documentation](https://datasketches.apache.org/docs/Quantiles/ReqSketch.html)
