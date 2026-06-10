//! Tests for the user-managed SortedView API: queries on `&self`,
//! `sorted_view()` returning an owned snapshot.

use reqsketch::{ReqError, ReqSketch, SearchCriteria, SortedView};

fn populated_sketch(n: u64) -> ReqSketch<f64> {
    let mut sketch = ReqSketch::new();
    for i in 0..n {
        sketch.update(i as f64);
    }
    sketch
}

/// All distribution queries must work through a shared reference.
fn query_through_shared_ref(sketch: &ReqSketch<f64>) {
    sketch
        .quantile(0.5, SearchCriteria::Inclusive)
        .expect("quantile");
    sketch
        .quantiles(&[0.25, 0.5, 0.75], SearchCriteria::Inclusive)
        .expect("quantiles");
    sketch.rank(&50.0, SearchCriteria::Inclusive).expect("rank");
    sketch.rank_inclusive(&50.0).expect("rank_inclusive");
    sketch.quantile_inclusive(0.5).expect("quantile_inclusive");
    sketch
        .pmf(&[10.0, 50.0], SearchCriteria::Inclusive)
        .expect("pmf");
    sketch
        .cdf(&[10.0, 50.0], SearchCriteria::Inclusive)
        .expect("cdf");
    assert!(!sketch.sorted_view().is_empty());
}

#[test]
fn queries_work_through_shared_reference() {
    let sketch = populated_sketch(100);
    query_through_shared_ref(&sketch);
}

#[test]
fn sorted_view_is_an_owned_snapshot() {
    let mut sketch = populated_sketch(100);

    let view: SortedView<f64> = sketch.sorted_view();
    assert_eq!(view.total_weight(), 100);

    // Updating the sketch while the view is alive must compile (owned view)
    // and must not affect the snapshot.
    for i in 100..200 {
        sketch.update(i as f64);
    }
    assert_eq!(view.total_weight(), 100);

    // A fresh view reflects the new state.
    let fresh = sketch.sorted_view();
    assert_eq!(fresh.total_weight(), 200);
}

#[test]
fn sorted_view_on_empty_sketch_is_an_empty_view() {
    let sketch: ReqSketch<f64> = ReqSketch::new();
    let view = sketch.sorted_view();
    assert!(view.is_empty());
    assert_eq!(view.len(), 0);
    assert_eq!(view.total_weight(), 0);
    // Queries on the empty view still report EmptySketch.
    assert_eq!(
        view.quantile(0.5, SearchCriteria::Inclusive).unwrap_err(),
        ReqError::EmptySketch
    );
}

#[test]
fn empty_sketch_pmf_cdf_report_empty_sketch() {
    let sketch: ReqSketch<f64> = ReqSketch::new();
    assert_eq!(
        sketch.pmf(&[1.0], SearchCriteria::Inclusive).unwrap_err(),
        ReqError::EmptySketch
    );
    assert_eq!(
        sketch.cdf(&[1.0], SearchCriteria::Inclusive).unwrap_err(),
        ReqError::EmptySketch
    );
}

#[test]
fn view_rank_is_primary_query_name() {
    let sketch = populated_sketch(10);
    let view = sketch.sorted_view();
    let r = view.rank(&5.0, SearchCriteria::Inclusive).expect("rank");
    assert!((r - 0.6).abs() < 1e-10, "rank of 5.0 in 0..10, got {r}");
}

#[test]
fn nan_query_items_are_rejected() {
    let sketch = populated_sketch(100);
    assert_eq!(
        sketch
            .rank(&f64::NAN, SearchCriteria::Inclusive)
            .unwrap_err(),
        ReqError::NanItem
    );
    let view = sketch.sorted_view();
    assert_eq!(
        view.rank(&f64::NAN, SearchCriteria::Inclusive).unwrap_err(),
        ReqError::NanItem
    );
}

#[test]
fn error_precedence_empty_before_invalid_rank() {
    let empty: ReqSketch<f64> = ReqSketch::new();
    assert_eq!(
        empty.quantile(2.0, SearchCriteria::Inclusive).unwrap_err(),
        ReqError::EmptySketch
    );
    let sketch = populated_sketch(10);
    assert_eq!(
        sketch.quantile(2.0, SearchCriteria::Inclusive).unwrap_err(),
        ReqError::InvalidRank(2.0)
    );
    assert_eq!(
        sketch
            .quantiles(&[0.5, -0.1], SearchCriteria::Inclusive)
            .unwrap_err(),
        ReqError::InvalidRank(-0.1)
    );
}

#[test]
fn view_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SortedView<f64>>();
    assert_send_sync::<ReqSketch<f64>>();
}

#[test]
fn concurrent_readers_share_the_sketch() {
    let sketch = std::sync::Arc::new(populated_sketch(1_000));
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let sketch = std::sync::Arc::clone(&sketch);
            std::thread::spawn(move || {
                let rank = 0.2 * (i + 1) as f64;
                sketch
                    .quantile(rank, SearchCriteria::Inclusive)
                    .expect("quantile from shared sketch")
            })
        })
        .collect();
    for handle in handles {
        let q = handle.join().expect("thread");
        assert!((0.0..1_000.0).contains(&q));
    }
}
