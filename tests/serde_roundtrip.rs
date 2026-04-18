//! Feature-gated round-trip tests for serde serialization.
//!
//! These tests only compile when the `serde` feature is enabled; the whole
//! file is a no-op otherwise so the default CI job can still build it.

#![cfg(feature = "serde")]

use reqsketch::{RankAccuracy, ReqSketch, SearchCriteria};

fn rebuilt_matches<T>(original: &ReqSketch<T>, rebuilt: &ReqSketch<T>)
where
    T: Clone + reqsketch::TotalOrd + PartialEq + std::fmt::Debug,
{
    assert_eq!(original.k(), rebuilt.k());
    assert_eq!(original.rank_accuracy(), rebuilt.rank_accuracy());
    assert_eq!(original.len(), rebuilt.len());
    assert_eq!(original.num_retained(), rebuilt.num_retained());
    assert_eq!(original.min_item(), rebuilt.min_item());
    assert_eq!(original.max_item(), rebuilt.max_item());
    assert_eq!(original.is_estimation_mode(), rebuilt.is_estimation_mode());
}

#[test]
fn json_roundtrip_exact_mode() {
    let mut sketch: ReqSketch<f64> = ReqSketch::new();
    for i in 0..50 {
        sketch.update(i as f64);
    }

    let json = serde_json::to_string(&sketch).expect("serialize");
    let mut rebuilt: ReqSketch<f64> = serde_json::from_str(&json).expect("deserialize");

    rebuilt_matches(&sketch, &rebuilt);

    for &q in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let before = sketch.quantile(q, SearchCriteria::Inclusive).unwrap();
        let after = rebuilt.quantile(q, SearchCriteria::Inclusive).unwrap();
        assert_eq!(before, after, "quantile({}) diverged after round-trip", q);
    }
}

#[test]
fn json_roundtrip_estimation_mode() {
    let mut sketch: ReqSketch<f64> = ReqSketch::builder()
        .k(12)
        .unwrap()
        .rank_accuracy(RankAccuracy::LowRank)
        .build()
        .unwrap();
    for i in 0..10_000 {
        sketch.update(i as f64);
    }
    assert!(sketch.is_estimation_mode());

    let json = serde_json::to_string(&sketch).expect("serialize");
    let mut rebuilt: ReqSketch<f64> = serde_json::from_str(&json).expect("deserialize");

    rebuilt_matches(&sketch, &rebuilt);

    for &q in &[0.01, 0.1, 0.5, 0.9, 0.99] {
        let before = sketch.quantile(q, SearchCriteria::Inclusive).unwrap();
        let after = rebuilt.quantile(q, SearchCriteria::Inclusive).unwrap();
        assert_eq!(before, after, "quantile({}) diverged after round-trip", q);
    }

    for &v in &[0.0_f64, 2500.0, 5000.0, 9999.0] {
        let before = sketch.rank(&v, SearchCriteria::Inclusive).unwrap();
        let after = rebuilt.rank(&v, SearchCriteria::Inclusive).unwrap();
        assert_eq!(before, after, "rank({}) diverged after round-trip", v);
    }
}

#[test]
fn roundtrip_still_mergeable() {
    let mut a: ReqSketch<f64> = ReqSketch::new();
    let mut b: ReqSketch<f64> = ReqSketch::new();
    for i in 0..500 {
        a.update(i as f64);
        b.update((i + 500) as f64);
    }

    let a_rebuilt: ReqSketch<f64> =
        serde_json::from_str(&serde_json::to_string(&a).unwrap()).expect("round-trip a");
    let b_rebuilt: ReqSketch<f64> =
        serde_json::from_str(&serde_json::to_string(&b).unwrap()).expect("round-trip b");

    let mut merged_direct = a.clone();
    merged_direct.merge(&b).unwrap();

    let mut merged_roundtripped = a_rebuilt.clone();
    merged_roundtripped.merge(&b_rebuilt).unwrap();

    assert_eq!(merged_direct.len(), merged_roundtripped.len());
    assert_eq!(merged_direct.min_item(), merged_roundtripped.min_item());
    assert_eq!(merged_direct.max_item(), merged_roundtripped.max_item());
}
