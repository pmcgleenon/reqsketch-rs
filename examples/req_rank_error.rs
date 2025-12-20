use plotters::prelude::*;
use rand::seq::SliceRandom;
use rand::rng;

use reqsketch::{ReqSketch, RankAccuracy, SearchCriteria};

const STREAM_LEN: usize = 1 << 11; // 2^11 = 2048 
const PLOT_POINTS: usize = 100;    // points along rank axis 
const TRIALS: usize = 1 << 14;     // 2^14 = 16384 trials 

// Quantile levels for the "pitchfork" curves (approx. +/- 1, 2, 3 sigma)
const P_MEDIAN: f64 = 0.5;
const P_PLUS1: f64 = 0.8413;
const P_MINUS1: f64 = 0.1587;
const P_PLUS2: f64 = 0.9772;
const P_MINUS2: f64 = 0.0228;
const P_PLUS3: f64 = 0.99865;
const P_MINUS3: f64 = 0.00135;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting REQ rank error analysis...");
    println!("Parameters: K=12, SL=2^{} ({}), PP={}, LgT={} ({}), Crit=LE",
             (STREAM_LEN as f64).log2() as u32, STREAM_LEN, PLOT_POINTS,
             (TRIALS as f64).log2() as u32, TRIALS);

    // HRA plot - legend at top right (where error converges to 0)
    run_experiment(
        RankAccuracy::HighRank,
        "assets/req_rank_error_hra.png",
        "REQ Rank Error – HighRank",
        SeriesLabelPosition::UpperRight,
    )?;

    // LRA plot - legend at top left (where error converges to 0)
    run_experiment(
        RankAccuracy::LowRank,
        "assets/req_rank_error_lra.png",
        "REQ Rank Error – LowRank",
        SeriesLabelPosition::UpperLeft,
    )?;

    println!("Generated files:");
    println!("  - assets/req_rank_error_hra.png");
    println!("  - assets/req_rank_error_lra.png");
    Ok(())
}

/// Run the experiment for a given accuracy mode and write a PNG.
fn run_experiment(
    accuracy: RankAccuracy,
    output_file: &str,
    title: &str,
    legend_pos: SeriesLabelPosition,
) -> Result<(), Box<dyn std::error::Error>> {
    // Precompute the "true" ranks and query values for the plot points.
    //
    // We treat the underlying distribution conceptually as uniform on [1, STREAM_LEN]
    // and define true_rank_j = j / (PLOT_POINTS - 1).
    // Then we choose a query value in that range, e.g. value_j = true_rank_j * STREAM_LEN.
    let mut true_ranks = Vec::with_capacity(PLOT_POINTS);
    let mut query_values = Vec::with_capacity(PLOT_POINTS);
    for j in 0..PLOT_POINTS {
        let r = j as f64 / (PLOT_POINTS as f64 - 1.0);
        let v = (r * STREAM_LEN as f64) as f32;
        true_ranks.push(r);
        query_values.push(v);
    }

    // errors[j] will hold TRIALS samples of est_rank - true_rank at plot point j
    let mut errors: Vec<Vec<f64>> = vec![Vec::with_capacity(TRIALS); PLOT_POINTS];

    // Base stream: 1..=STREAM_LEN as f32
    let mut stream: Vec<f32> = (1..=STREAM_LEN as u32).map(|x| x as f32).collect();

    for t in 0..TRIALS {
        if t % 1000 == 0 {
            println!("  Trial {}/{} for {:?} ...", t + 1, TRIALS, accuracy);
        }
        stream.shuffle(&mut rng());

        let mut sketch = ReqSketch::builder()
            .k(12)? // match the docs' K=12 example; adjust if you like
            .rank_accuracy(accuracy)
            .build()?;

        for &x in &stream {
            sketch.update(x);
        }

        // For each plot point, record the rank error est_rank - true_rank.
        for j in 0..PLOT_POINTS {
            let v = query_values[j];
            let true_rank = true_ranks[j];

            let est_rank = sketch
                .rank(&v, SearchCriteria::Inclusive)?;

            let err = est_rank - true_rank;
            errors[j].push(err);
        }
    }

    // Now compute the "pitchfork" curves: median and +/- 1, 2, 3 SD-like quantiles.
    let mut median_curve = Vec::with_capacity(PLOT_POINTS);
    let mut plus1_curve = Vec::with_capacity(PLOT_POINTS);
    let mut minus1_curve = Vec::with_capacity(PLOT_POINTS);
    let mut plus2_curve = Vec::with_capacity(PLOT_POINTS);
    let mut minus2_curve = Vec::with_capacity(PLOT_POINTS);
    let mut plus3_curve = Vec::with_capacity(PLOT_POINTS);
    let mut minus3_curve = Vec::with_capacity(PLOT_POINTS);

    for j in 0..PLOT_POINTS {
        let mut e = errors[j].clone();
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = get_quantile(&e, P_MEDIAN);
        let p1 = get_quantile(&e, P_PLUS1);
        let m1 = get_quantile(&e, P_MINUS1);
        let p2 = get_quantile(&e, P_PLUS2);
        let m2 = get_quantile(&e, P_MINUS2);
        let p3 = get_quantile(&e, P_PLUS3);
        let m3 = get_quantile(&e, P_MINUS3);

        median_curve.push(median);
        plus1_curve.push(p1);
        minus1_curve.push(m1);
        plus2_curve.push(p2);
        minus2_curve.push(m2);
        plus3_curve.push(p3);
        minus3_curve.push(m3);
    }

    plot_rank_error(
        output_file,
        &true_ranks,
        &median_curve,
        &plus1_curve,
        &minus1_curve,
        &plus2_curve,
        &minus2_curve,
        &plus3_curve,
        &minus3_curve,
        title,
        legend_pos,
    )?;

    Ok(())
}

/// Simple quantile extractor from a sorted slice [0..n).
fn get_quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len();
    let pos = (q * (n as f64 - 1.0)).round() as usize;
    sorted[pos.clamp(0, n - 1)]
}

/// Draws the rank error "pitchfork" plot.
fn plot_rank_error(
    filename: &str,
    ranks: &[f64],
    median: &[f64],
    plus1: &[f64],
    minus1: &[f64],
    plus2: &[f64],
    minus2: &[f64],
    plus3: &[f64],
    minus3: &[f64],
    title: &str,
    legend_pos: SeriesLabelPosition,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    // Compute a symmetric y-axis range based on +/- 3σ curves.
    let max_abs = plus3
        .iter()
        .chain(minus3.iter())
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
        .max(1e-6); // avoid degenerate range

    let x_range = 0.0..1.0;
    let y_range = -max_abs..max_abs;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_range.clone(), y_range.clone())?;

    chart
        .configure_mesh()
        .x_desc("True rank")
        .y_desc("Error")
        .draw()?;

    // Helper to build line series from (ranks, ys)
    let make_series = |ys: &[f64]| -> Vec<(f64, f64)> {
        ranks
            .iter()
            .zip(ys.iter())
            .map(|(r, y)| (*r, *y))
            .collect()
    };

    // Median (center) line
    chart
        .draw_series(LineSeries::new(make_series(median), &BLACK))?
        .label("median")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));

    // +/- 1σ
    chart
        .draw_series(LineSeries::new(make_series(plus1), &BLUE))?
        .label("+1σ")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    chart
        .draw_series(LineSeries::new(make_series(minus1), &BLUE))?
        .label("-1σ")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    // +/- 2σ
    chart
        .draw_series(LineSeries::new(make_series(plus2), &RED))?
        .label("+2σ")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
    chart
        .draw_series(LineSeries::new(make_series(minus2), &RED))?
        .label("-2σ")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    // +/- 3σ
    chart
        .draw_series(LineSeries::new(make_series(plus3), &MAGENTA))?
        .label("+3σ")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], MAGENTA));
    chart
        .draw_series(LineSeries::new(make_series(minus3), &MAGENTA))?
        .label("-3σ")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], MAGENTA));

    chart
        .configure_series_labels()
        .position(legend_pos)
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()?;

    root.present()?;
    Ok(())
}
