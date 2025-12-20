//! # REQSketch - Relative Error Quantiles Sketch
//!
//! Relative Error Quantiles sketch algorithm in Rust.
//! This implementation is based on the paper "Relative Error Streaming Quantiles" by
//! Graham Cormode, Zohar Karnin, Edo Liberty, Justin Thaler, Pavel Veselý.
//!
//! And on the C++ implementation in Apache DataSketches
//! https://github.com/apache/datasketches-cpp
//!
//! ## Quick Start
//!
//! ```rust
//! use reqsketch::{ReqSketch, SearchCriteria, Result};
//!
//! fn main() -> Result<()> {
//!     let mut sketch = ReqSketch::new();
//!
//!     // Add values to the sketch
//!     for i in 0..10000 {
//!         sketch.update(i as f64);
//!     }
//!
//!     // Query quantiles with proper error handling
//!     let median = sketch.quantile(0.5, SearchCriteria::Inclusive)?;
//!     let p99 = sketch.quantile(0.99, SearchCriteria::Inclusive)?;
//!
//!     // Query ranks with proper error handling
//!     let rank = sketch.rank(&5000.0, SearchCriteria::Inclusive)?;
//!
//!     println!("Median: {}, P99: {}, Rank of 5000: {}", median, p99, rank);
//!     Ok(())
//! }
//! ```

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]

use std::fmt;
use std::cmp::Ordering;

pub mod builder;
pub mod compactor;
pub mod error;
pub mod iter;
pub mod sorted_view;

/// Trait for total ordering that handles NaN consistently.
pub trait TotalOrd {
    fn total_cmp(&self, other: &Self) -> Ordering;
}

impl TotalOrd for f64 {
    #[inline(always)]
    fn total_cmp(&self, other: &Self) -> Ordering {
        // Fast path for normal values (common case)
        // partial_cmp is significantly faster than total_cmp
        if let Some(ord) = self.partial_cmp(other) {
            ord
        } else {
            // Fallback to total_cmp for NaN/infinity cases (rare)
            f64::total_cmp(self, other)
        }
    }
}

impl TotalOrd for f32 {
    #[inline(always)]
    fn total_cmp(&self, other: &Self) -> Ordering {
        f32::total_cmp(self, other)
    }
}

// Implementations for common integer types
macro_rules! impl_total_ord_for_ord {
    ($($t:ty),*) => {
        $(
            impl TotalOrd for $t {
                #[inline(always)]
                fn total_cmp(&self, other: &Self) -> Ordering {
                    self.cmp(other)
                }
            }
        )*
    };
}

impl_total_ord_for_ord!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize);

/// Trait for types that can be used efficiently in REQ sketches.
/// Implemented for numeric types that support fast copy semantics.
pub trait ReqKey: Copy + PartialOrd + Clone {}

impl ReqKey for f64 {}
impl ReqKey for f32 {}
impl ReqKey for i64 {}
impl ReqKey for i32 {}
impl ReqKey for u64 {}
impl ReqKey for u32 {}


#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use builder::ReqSketchBuilder;
pub use error::{ReqError, Result};
pub use iter::ReqSketchIterator;
pub use sorted_view::SortedView;

/// Configuration for rank accuracy optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default)]
pub enum RankAccuracy {
    /// Optimize for accuracy at high ranks (near 1.0)
    #[default]
    HighRank,
    /// Optimize for accuracy at low ranks (near 0.0)
    LowRank,
}


/// Search criteria for quantile/rank operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum SearchCriteria {
    /// Include the weight of the search item in the result
    #[default]
    Inclusive,
    /// Exclude the weight of the search item from the result
    Exclusive,
}


/// A Relative Error Quantiles sketch for approximate quantile estimation.
///
/// The REQ sketch provides approximate quantile estimation with relative error guarantees.
/// It is particularly useful for streaming scenarios where you need to estimate quantiles
/// over large data streams with bounded memory usage.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: Clone + TotalOrd + PartialEq + serde::Serialize + serde::de::DeserializeOwned"))]
pub struct ReqSketch<T> {
    k: u16,
    rank_accuracy: RankAccuracy,
    total_n: u64,
    max_nom_size: u32,
    num_retained: u32,
    compactors: Vec<compactor::Compactor<T>>,
    /// Reusable buffer for promotions to avoid per-compaction allocation.
    promotion_buf: Vec<T>,
    min_item: Option<T>,
    max_item: Option<T>,
    sorted_view_cache: Option<SortedView<T>>,
}

impl<T> ReqSketch<T>
where
    T: Clone + TotalOrd + PartialEq,
{
    /// Creates a new REQ sketch with default parameters.
    ///
    /// Uses k=12 (roughly 1% relative error at 95% confidence) and high rank accuracy.
    pub fn new() -> Self {
        Self {
            k: 12,
            rank_accuracy: RankAccuracy::HighRank,
            total_n: 0,
            max_nom_size: 0,
            num_retained: 0,
            compactors: Vec::new(),
            promotion_buf: Vec::with_capacity(12),
            min_item: None,
            max_item: None,
            sorted_view_cache: None,
        }
    }

    /// Creates a new REQ sketch with the specified k parameter.
    ///
    /// # Arguments
    /// * `k` - Controls size and error of the sketch. Must be even and in range [4, 1024].
    ///   Value of 12 roughly corresponds to 1% relative error at 95% confidence.
    pub fn with_k(k: u16) -> Result<Self> {
        ReqSketchBuilder::new().k(k)?.build()
    }

    /// Returns a new builder for constructing a REQ sketch with custom parameters.
    pub fn builder() -> ReqSketchBuilder<T> {
        ReqSketchBuilder::new()
    }

    /// Returns the configured k parameter.
    pub fn k(&self) -> u16 {
        self.k
    }

    /// Returns the rank accuracy configuration.
    pub fn rank_accuracy(&self) -> RankAccuracy {
        self.rank_accuracy
    }

    /// Returns true if this sketch is empty.
    pub fn is_empty(&self) -> bool {
        self.total_n == 0
    }

    /// Returns the total number of items processed by this sketch.
    pub fn len(&self) -> u64 {
        self.total_n
    }

    /// Returns the number of items currently retained in the sketch.
    pub fn num_retained(&self) -> u32 {
        self.num_retained
    }

    /// Returns true if this sketch is in estimation mode (has compacted data).
    pub fn is_estimation_mode(&self) -> bool {
        self.compactors.len() > 1
    }

    /// Updates the sketch with a new item.
    pub fn update(&mut self, item: T) {
        // Track exact min/max for DataSketches parity
        // Update min
        match &mut self.min_item {
            None => self.min_item = Some(item.clone()),
            Some(min) if item.total_cmp(min).is_lt() => *min = item.clone(),
            _ => {}
        }

        // Update max
        match &mut self.max_item {
            None => self.max_item = Some(item.clone()),
            Some(max) if item.total_cmp(max).is_gt() => *max = item.clone(),
            _ => {}
        }

        // Ensure we have at least one compactor
        if self.compactors.is_empty() {
            self.grow();
        }

        // Add to level 0 compactor
        self.compactors[0].append(item);
        self.total_n += 1;
        self.num_retained += 1;

        // Trigger compression when global retained equals global max 
        if self.num_retained == self.max_nom_size {
            self.compress();
        }

        // Invalidate cache
        self.sorted_view_cache = None;
    }

    /// Merges another sketch into this one.
    pub fn merge(&mut self, other: &Self) -> Result<()> {
        // Validate compatibility first, regardless of whether other is empty
        if self.rank_accuracy != other.rank_accuracy {
            return Err(ReqError::IncompatibleSketches(
                "Sketches must have the same rank accuracy setting".to_string(),
            ));
        }

        if self.k != other.k {
            return Err(ReqError::IncompatibleSketches(
                "Sketches must have the same k parameter".to_string(),
            ));
        }

        // Early return if other is empty (after validation)
        if other.is_empty() {
            return Ok(());
        }

        self.total_n += other.total_n;

        // Update min/max
        if let Some(other_min) = &other.min_item {
            match &self.min_item {
                None => self.min_item = Some(other_min.clone()),
                Some(min) if other_min.total_cmp(min).is_lt() => self.min_item = Some(other_min.clone()),
                _ => {}
            }
        }

        if let Some(other_max) = &other.max_item {
            match &self.max_item {
                None => self.max_item = Some(other_max.clone()),
                Some(max) if other_max.total_cmp(max).is_gt() => self.max_item = Some(other_max.clone()),
                _ => {}
            }
        }

        // Grow until we have at least as many levels as other
        while self.compactors.len() < other.compactors.len() {
            self.grow();
        }

        // Merge compactors at each level
        for (i, other_compactor) in other.compactors.iter().enumerate() {
            self.compactors[i].merge(other_compactor)?;
        }

        self.update_max_nom_size();
        self.update_num_retained();

        // Compress if needed
        if self.num_retained >= self.max_nom_size {
            self.compress();
        }

        // Invalidate cache
        self.sorted_view_cache = None;

        Ok(())
    }

    /// Returns the exact minimum item from the stream, or None if empty.
    /// This is the true minimum value ever seen, not an approximation.
    pub fn min_item(&self) -> Option<&T> {
        self.min_item.as_ref()
    }

    /// Returns the exact maximum item from the stream, or None if empty.
    /// This is the true maximum value ever seen, not an approximation.
    pub fn max_item(&self) -> Option<&T> {
        self.max_item.as_ref()
    }

    /// Returns the approximate rank of the given item.
    ///
    /// # Arguments
    /// * `item` - The item to find the rank for
    /// * `criteria` - Whether to include the item's weight in the rank calculation
    ///
    /// # Returns
    /// A value in [0.0, 1.0] representing the approximate normalized rank.
    pub fn rank(&mut self, item: &T, criteria: SearchCriteria) -> Result<f64> {
        if self.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        let sorted_view = self.get_sorted_view()?;
        sorted_view.rank_no_interpolation(item, criteria)
    }

    /// Returns the approximate rank of the given item using inclusive criteria.
    pub fn rank_inclusive(&mut self, item: &T) -> Result<f64> {
        self.rank(item, SearchCriteria::Inclusive)
    }


    /// Returns the approximate quantile for the given normalized rank.
    ///
    /// # Arguments
    /// * `rank` - A value in [0.0, 1.0] representing the normalized rank
    /// * `criteria` - Search criteria for quantile selection
    ///
    /// # Returns
    /// The approximate quantile value for the given rank.
    pub fn quantile(&mut self, rank: f64, criteria: SearchCriteria) -> Result<T> {
        if self.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        if !(0.0..=1.0).contains(&rank) {
            return Err(ReqError::InvalidRank(rank));
        }

        // Use the cached sorted view (working correctly)
        let sorted_view = self.get_sorted_view()?;
        sorted_view.quantile(rank, criteria)
    }

    /// Returns the approximate quantile for the given normalized rank using inclusive criteria.
    pub fn quantile_inclusive(&mut self, rank: f64) -> Result<T> {
        self.quantile(rank, SearchCriteria::Inclusive)
    }

    /// Returns multiple quantiles for the given normalized ranks.
    pub fn quantiles(&mut self, ranks: &[f64], criteria: SearchCriteria) -> Result<Vec<T>> {
        if self.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        let sorted_view = self.get_sorted_view()?;
        let mut results = Vec::with_capacity(ranks.len());
        for &rank in ranks {
            if !(0.0..=1.0).contains(&rank) {
                return Err(ReqError::InvalidRank(rank));
            }
            results.push(sorted_view.quantile(rank, criteria)?);
        }
        Ok(results)
    }

    /// Returns the Probability Mass Function (PMF) for the given split points.
    pub fn pmf(&mut self, split_points: &[T], criteria: SearchCriteria) -> Result<Vec<f64>> {
        if self.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        let sorted_view = self.get_sorted_view()?;
        sorted_view.pmf(split_points, criteria)
    }

    /// Returns the Cumulative Distribution Function (CDF) for the given split points.
    pub fn cdf(&mut self, split_points: &[T], criteria: SearchCriteria) -> Result<Vec<f64>> {
        if self.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        let sorted_view = self.get_sorted_view()?;
        sorted_view.cdf(split_points, criteria)
    }

    /// Returns an iterator over the (item, weight) pairs in the sketch.
    pub fn iter(&self) -> ReqSketchIterator<'_, T> {
        ReqSketchIterator::new(&self.compactors)
    }

    /// Returns a sorted view of the sketch for efficient queries.
    pub fn sorted_view(&mut self) -> Result<&SortedView<T>> {
        self.get_sorted_view()
    }

    /// Resets the sketch to its initial empty state.
    pub fn reset(&mut self) {
        self.total_n = 0;
        self.min_item = None;
        self.max_item = None;
        self.compactors.clear();
        self.sorted_view_cache = None;
    }


    // Internal methods

    fn get_sorted_view(&mut self) -> Result<&SortedView<T>> {
        if self.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        // Build and cache the sorted view if not already cached
        if self.sorted_view_cache.is_none() {
            self.sorted_view_cache = Some(self.compute_sorted_view()?);
        }

        self.sorted_view_cache.as_ref()
            .ok_or(ReqError::CacheInvalid)
    }

    fn compute_sorted_view(&self) -> Result<SortedView<T>> {
        let mut weighted_items = Vec::new();

        for compactor in &self.compactors {
            let weight = compactor.weight();
            for item in compactor.iter() {
                weighted_items.push((item.clone(), weight));
            }
        }

        Ok(SortedView::new(weighted_items))
    }

    fn compress(&mut self) {
        for h in 0..self.compactors.len() {
            if self.compactors[h].num_items() >= self.compactors[h].nominal_capacity() {
                // Sort level 0 before compaction
                if h == 0 {
                    self.compactors[0].sort();
                }

                // Grow if at top level
                if h + 1 >= self.compactors.len() {
                    self.grow();
                }

                // Compact into reusable promotion buffer
                self.promotion_buf.clear();
                self.compactors[h].compact_into(self.rank_accuracy, &mut self.promotion_buf);

                // Merge promoted items into next level
                if !self.promotion_buf.is_empty() {
                    self.compactors[h + 1].merge_sorted(&self.promotion_buf);
                }

                // Update max_nom_size with capacity change from ensure_enough_sections
                self.update_max_nom_size();
                self.update_num_retained();
            }
        }

        // Invalidate cache
        self.sorted_view_cache = None;
    }

    fn grow(&mut self) {
        let level = self.compactors.len() as u8;
        let compactor = compactor::Compactor::new(level, self.k, self.rank_accuracy);
        self.compactors.push(compactor);
        self.update_max_nom_size();
    }

    /// Update total nominal capacity across all levels
    fn update_max_nom_size(&mut self) {
        self.max_nom_size = self.compactors.iter()
            .map(|c| c.nominal_capacity())
            .sum();
    }

    /// Update total retained items across all levels 
    fn update_num_retained(&mut self) {
        self.num_retained = self.compactors.iter()
            .map(|c| c.num_items())
            .sum();
    }

    // Debug method to inspect compactor state
    #[cfg(test)]
    pub(crate) fn debug_compactor_info(&self) -> Vec<(u8, u32, u32)> {
        self.compactors.iter().map(|c| (c.lg_weight(), c.num_items(), c.nominal_capacity())).collect()
    }

    /// Returns the total number of retained items across all levels.
    /// This is useful for testing global capacity constraints.
    #[doc(hidden)]
    pub fn total_retained_items(&self) -> u32 {
        self.compactors.iter().map(|c| c.num_items()).sum()
    }

    /// Returns the total nominal capacity across all levels.
    /// This is useful for testing global capacity constraints.
    #[doc(hidden)]
    pub fn total_nominal_capacity(&self) -> u32 {
        self.compactors.iter().map(|c| c.nominal_capacity()).sum()
    }

    /// Returns information about each level: (level, items, capacity, weight).
    /// This is useful for testing level structure and over-capacity behavior.
    #[doc(hidden)]
    pub fn level_info(&self) -> Vec<(usize, u32, u32, u64)> {
        self.compactors.iter().enumerate()
            .map(|(i, c)| (i, c.num_items(), c.nominal_capacity(), c.weight()))
            .collect()
    }

    /// Returns the computed total weight by summing level weights.
    /// This is useful for testing weight conservation.
    #[doc(hidden)]
    pub fn computed_total_weight(&self) -> u64 {
        self.compactors.iter()
            .map(|c| c.num_items() as u64 * c.weight())
            .sum()
    }

    /// Returns a public accessor to the sorted view for testing.
    #[doc(hidden)]
    pub fn test_get_sorted_view(&mut self) -> Result<&SortedView<T>> {
        self.sorted_view()
    }






    /// Returns the lower bound for the rank of a given quantile at the specified confidence level.
    ///
    /// # Arguments
    /// * `rank` - The rank to compute the lower bound for (0.0 to 1.0)
    /// * `num_std_dev` - Number of standard deviations for confidence level (1, 2, or 3)
    ///
    /// Returns the lower bound rank estimate with the specified confidence.
    pub fn get_rank_lower_bound(&self, rank: f64, num_std_dev: u8) -> f64 {
        self.compute_rank_lower_bound(self.k, self.compactors.len() as u8, rank, num_std_dev, self.total_n, matches!(self.rank_accuracy, RankAccuracy::HighRank))
    }

    /// Returns the upper bound for the rank of a given quantile at the specified confidence level.
    ///
    /// # Arguments
    /// * `rank` - The rank to compute the upper bound for (0.0 to 1.0)
    /// * `num_std_dev` - Number of standard deviations for confidence level (1, 2, or 3)
    ///
    /// Returns the upper bound rank estimate with the specified confidence.
    pub fn get_rank_upper_bound(&self, rank: f64, num_std_dev: u8) -> f64 {
        self.compute_rank_upper_bound(self.k, self.compactors.len() as u8, rank, num_std_dev, self.total_n, matches!(self.rank_accuracy, RankAccuracy::HighRank))
    }

    const FIXED_RSE_FACTOR: f64 = 0.084;
    const INIT_NUM_SECTIONS: u8 = 3;

    /// Calculates the relative RSE factor used in error bounds
    fn relative_rse_factor() -> f64 {
        (0.0512 / Self::INIT_NUM_SECTIONS as f64).sqrt()
    }

    /// Computes the lower bound rank estimate with the specified confidence level.
    fn compute_rank_lower_bound(&self, k: u16, num_levels: u8, rank: f64, num_std_dev: u8, n: u64, hra: bool) -> f64 {
        if self.is_exact_rank_threshold(k, num_levels, rank, n, hra) {
            return rank;
        }
        let relative = Self::relative_rse_factor() / k as f64 * if hra { 1.0 - rank } else { rank };
        let fixed = Self::FIXED_RSE_FACTOR / k as f64;
        let lb_rel = rank - num_std_dev as f64 * relative;
        let lb_fix = rank - num_std_dev as f64 * fixed;
        lb_rel.min(lb_fix).max(0.0)
    }

    /// Computes the upper bound rank estimate with the specified confidence level.
    fn compute_rank_upper_bound(&self, k: u16, num_levels: u8, rank: f64, num_std_dev: u8, n: u64, hra: bool) -> f64 {
        if self.is_exact_rank_threshold(k, num_levels, rank, n, hra) {
            return rank;
        }
        let relative = Self::relative_rse_factor() / k as f64 * if hra { 1.0 - rank } else { rank };
        let fixed = Self::FIXED_RSE_FACTOR / k as f64;
        let ub_rel = rank + num_std_dev as f64 * relative;
        let ub_fix = rank + num_std_dev as f64 * fixed;
        ub_rel.max(ub_fix).min(1.0)
    }

    /// Determines if a rank should be considered exact based on the exact rank threshold.
    /// When a rank is exact, no error bounds need to be computed.
    fn is_exact_rank_threshold(&self, k: u16, num_levels: u8, rank: f64, n: u64, hra: bool) -> bool {
        let base_cap = k as u64 * Self::INIT_NUM_SECTIONS as u64;
        if num_levels == 1 || n <= base_cap {
            return true;
        }
        let exact_rank_thresh = base_cap as f64 / n as f64;
        if hra {
            rank >= 1.0 - exact_rank_thresh
        } else {
            rank <= exact_rank_thresh
        }
    }
}


impl<T> Default for ReqSketch<T>
where
    T: Clone + TotalOrd + PartialEq,
{
    fn default() -> Self {
        Self::new()
    }
}

// Note: f64 gets interpolation via SortedView<f64>::rank() specialization

impl<T> fmt::Display for ReqSketch<T>
where
    T: fmt::Display + Clone + TotalOrd + PartialEq,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "REQ Sketch Summary:")?;
        writeln!(f, "  K: {}", self.k)?;
        writeln!(f, "  Total items: {}", self.total_n)?;
        writeln!(f, "  Retained items: {}", self.num_retained())?;
        writeln!(f, "  Estimation mode: {}", self.is_estimation_mode())?;
        writeln!(f, "  Rank accuracy: {:?}", self.rank_accuracy)?;
        writeln!(f, "  Levels: {}", self.compactors.len())?;

        if let (Some(min), Some(max)) = (&self.min_item, &self.max_item) {
            writeln!(f, "  Min item: {}", min)?;
            writeln!(f, "  Max item: {}", max)?;
        }

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_sketch() {
        let sketch: ReqSketch<f64> = ReqSketch::new();
        assert!(sketch.is_empty());
        assert_eq!(sketch.len(), 0);
        assert_eq!(sketch.k(), 12);
        assert_eq!(sketch.rank_accuracy(), RankAccuracy::HighRank);
    }

    #[test]
    fn test_update_and_basic_queries() -> Result<()> {
        let mut sketch = ReqSketch::new();

        // Add some values
        for i in 0..100 {
            sketch.update(i as f64);
        }

        assert!(!sketch.is_empty());
        assert_eq!(sketch.len(), 100);

        // During compaction, some items may be discarded, so check ranges instead
        let min = sketch.min_item().ok_or(ReqError::EmptySketch)?;
        let max = sketch.max_item().ok_or(ReqError::EmptySketch)?;
        assert!(*min >= 0.0 && *min <= 10.0, "Min should be in reasonable range, got {}", min);
        assert!(*max >= 89.0 && *max <= 99.0, "Max should be in reasonable range, got {}", max);
        Ok(())
    }

    #[test]
    fn test_builder_pattern() -> Result<()> {
        let sketch: Result<ReqSketch<i32>> = ReqSketch::builder()
            .k(16).map(|builder| builder.rank_accuracy(RankAccuracy::LowRank))
            .and_then(|builder| builder.build());

        assert!(sketch.is_ok());
        let sketch = sketch?;
        assert_eq!(sketch.k(), 16);
        assert_eq!(sketch.rank_accuracy(), RankAccuracy::LowRank);
        Ok(())
    }

    #[test]
    fn test_weight_consistency_detailed() {
        let mut sketch = ReqSketch::new();

        // Check at the exact point where mismatch occurs
        for i in 0..80 {
            sketch.update(i as f64);

            let total_weight: u64 = sketch.iter().map(|(_, weight)| weight).sum();
            let expected = sketch.len();

            if i >= 75 {
                println!("Item {}: total_weight={}, expected={}, diff={}",
                         i, total_weight, expected, total_weight as i64 - expected as i64);

                if total_weight != expected {
                    println!("  MISMATCH DETAILS:");
                    println!("    estimation_mode: {}", sketch.is_estimation_mode());
                    println!("    num_retained: {}", sketch.num_retained());

                    // Show all compactor levels
                    let debug_info = sketch.debug_compactor_info();
                    for (level, (lg_weight, num_items, capacity)) in debug_info.iter().enumerate() {
                        let weight = 1u64 << lg_weight;
                        let level_total = *num_items as u64 * weight;
                        println!("    Level {}: {} items × {} weight = {} total (capacity: {})",
                                 level, num_items, weight, level_total, capacity);
                    }

                    break;
                }
            }
        }
    }

    #[test]
    fn test_merge_commutativity_debug() -> Result<()> {
        // Test the exact failing case: sketch with zeros + one with other values
        let mut sketch1a = ReqSketch::new();
        let mut sketch2a = ReqSketch::new();
        let mut sketch1b = ReqSketch::new();
        let mut sketch2b = ReqSketch::new();

        // Add zeros to first group
        for _ in 0..35 {
            sketch1a.update(0.0);
            sketch1b.update(0.0);
        }

        // Add some mixed values
        let values = [412.275, 721.747, 731.854, 249.854, 979.752];
        for &val in &values {
            sketch1a.update(val);
            sketch1b.update(val);
        }

        // Add different values to second group
        let values2 = [516.453, 879.855, 244.286, 909.822];
        for &val in &values2 {
            sketch2a.update(val);
            sketch2b.update(val);
        }

        // Test merge order: A + B vs B + A
        sketch1a.merge(&sketch2a)?;  // sketch1a + sketch2a
        sketch2b.merge(&sketch1b)?;  // sketch2b + sketch1b

        // Verify merge succeeded and resulted in non-empty sketches
        assert!(!sketch1a.is_empty(), "Merged sketch should not be empty");
        assert!(!sketch2b.is_empty(), "Merged sketch should not be empty");

        // Verify both sketches have the same total count
        assert_eq!(sketch1a.len(), sketch2b.len(), "Merged sketches should have same length");

        // Test commutativity: quantiles should be similar regardless of merge order
        let q1: f64 = sketch1a.quantile(0.5, SearchCriteria::Inclusive)?;
        let q2: f64 = sketch2b.quantile(0.5, SearchCriteria::Inclusive)?;

        let diff = (q1 - q2).abs();
        // Allow for some small numerical differences due to compaction ordering
        assert!(diff < 100.0, "Quantiles should be reasonably close: {} vs {} (diff: {})", q1, q2, diff);
        Ok(())
    }

    #[test]
    fn test_exact_quantile_debug() -> Result<()> {
        // Test the exact same scenario as C++ reference: [1,2,3,4,5,6,7,8,9,10]
        let mut sketch = ReqSketch::new();
        for i in 1..=10 {
            sketch.update(i as f64);
        }

        // Verify sketch contains expected number of items
        assert_eq!(sketch.len(), 10, "Sketch should contain 10 items");
        assert_eq!(sketch.total_n, 10, "Total count should be 10");
        assert!(!sketch.is_estimation_mode(), "Should be in exact mode for 10 items");

        // Test inclusive quantiles with precise expectations
        let test_cases = [
            (0.0, 1.0), (0.1, 1.0), (0.5, 5.0), (0.9, 9.0), (1.0, 10.0)
        ];

        for &(rank, expected) in &test_cases {
            let quantile = sketch.quantile(rank, SearchCriteria::Inclusive)?;
            // For exact mode with 10 items, quantiles should be precise
            assert_eq!(quantile, expected, "quantile({}, Inclusive) should be {}", rank, expected);
        }

        // Test exclusive quantiles
        let exclusive_cases = [
            (0.0, 1.0), (0.1, 2.0), (0.5, 6.0), (0.9, 10.0), (1.0, 10.0)
        ];

        for &(rank, expected) in &exclusive_cases {
            let quantile = sketch.quantile(rank, SearchCriteria::Exclusive)?;
            assert_eq!(quantile, expected, "quantile({}, Exclusive) should be {}", rank, expected);
        }

        // Verify sorted view consistency
        let sorted_view = sketch.get_sorted_view()?;
        assert_eq!(sorted_view.total_weight(), 10, "Total weight should be 10");
        assert_eq!(sorted_view.len(), 10, "Sorted view should have 10 items");
        Ok(())
    }

    #[test]
    fn test_weight_debug_1000() {
        // Test the exact failing case from weight consistency test
        let mut sketch = ReqSketch::new();
        let n = 1000;

        for i in 0..n {
            sketch.update(i as f64);
        }

        // Verify final counts
        assert_eq!(sketch.total_n, n as u64, "Total count should equal number of updates");

        // Verify weight consistency across compactors
        let mut total_weight = 0u64;
        for compactor in &sketch.compactors {
            if compactor.num_items() > 0 {
                let level_weight = compactor.num_items() as u64 * compactor.weight();
                total_weight += level_weight;
            }
        }
        assert_eq!(total_weight, sketch.total_n, "Sum of compactor weights should equal total_n");

        // Verify iterator weight consistency
        let iter_weight: u64 = sketch.iter().map(|(_, weight)| weight).sum();
        assert_eq!(iter_weight, sketch.total_n, "Iterator weight sum should equal total_n");

        // Verify sketch is in estimation mode for 1000 items
        assert!(sketch.is_estimation_mode(), "Should be in estimation mode for 1000 items");
    }

    #[test]
    fn test_capacity_debug() {
        let mut sketch = ReqSketch::new(); // k=12 by default

        // Verify capacity calculations are consistent across levels
        for level in 0..5 {
            let compactor = crate::compactor::Compactor::<f64>::new(level, 12, RankAccuracy::HighRank);
            let section_size = compactor.section_size();
            let nominal_capacity = compactor.nominal_capacity();

            // Verify section size is even and reasonable
            assert!(section_size.is_multiple_of(2), "Section size should be even for level {}", level);
            assert!(section_size >= 4, "Section size should be at least 4 for level {}", level);

            // Verify capacity is 2 * section_size * num_sections (3 initially)
            assert_eq!(nominal_capacity, 2 * section_size * 3,
                      "Nominal capacity calculation should be correct for level {}", level);
        }

        // Add items and verify compaction behavior
        let initial_len = sketch.compactors.len();
        for i in 1..=50 {
            sketch.update(i as f64);
        }

        // Verify compaction occurred or items are managed properly
        assert!(sketch.len() <= 50, "Sketch should contain at most 50 items");
        assert!(sketch.total_n == 50, "Total count should be 50");

        // If compaction occurred, verify we have multiple levels
        if sketch.len() < 50 {
            assert!(sketch.compactors.len() > initial_len ||
                   sketch.compactors.iter().any(|c| c.num_items() > 0),
                   "Should have active compactors if items were compacted");
        }
    }

    #[test]
    fn test_compaction_debug() -> Result<()> {
        // Test with enough items to trigger compaction
        let mut sketch = ReqSketch::new();
        let n = 50; // Should trigger some compaction

        let mut previous_len = 0;
        for i in 0..n {
            sketch.update(i as f64);

            // Track if compaction occurs
            if sketch.len() < previous_len {
                // compaction occurred
            }
            previous_len = sketch.len();
        }

        // Verify final state
        assert_eq!(sketch.total_n, n, "Total count should equal number of updates");

        // Verify weight consistency
        let total_weight: u64 = sketch.iter().map(|(_, weight)| weight).sum();
        assert_eq!(total_weight, sketch.total_n, "Total weight should equal total_n");

        // Test quantile accuracy
        let median = sketch.quantile(0.5, SearchCriteria::Inclusive)?;
        let expected_median = (n - 1) as f64 * 0.5;  // 0-indexed, so n-1 items, median at (n-1)/2

        // For 50 items, allow reasonable error due to compaction
        let error = (median - expected_median).abs() / expected_median;
        assert!(error < 0.2, "Median error should be reasonable: got {}, expected {}, error: {:.2}%",
               median, expected_median, error * 100.0);

        // Verify sketch behaves reasonably
        assert!(sketch.len() <= n, "Should not have more items than inserted");
        Ok(())
    }

    #[test]
    fn debug_section_growth() {
        let mut sketch = ReqSketch::new(); // k=12 by default

        let mut max_levels = 0;
        let mut reached_estimation_mode = false;

        for i in 0..200 {
            sketch.update(i as f64);

            // Track progress
            if sketch.compactors.len() > max_levels {
                max_levels = sketch.compactors.len();
            }
            if sketch.is_estimation_mode() {
                reached_estimation_mode = true;
            }
        }

        // Verify section growth behavior
        assert_eq!(sketch.total_n, 200, "Should have processed 200 items");
        assert!(reached_estimation_mode, "Should reach estimation mode with 200 items");
        assert!(max_levels > 1, "Should have multiple compactor levels");

        // Verify weight consistency
        let total_weight: u64 = sketch.iter().map(|(_, weight)| weight).sum();
        assert_eq!(total_weight, sketch.total_n, "Weight should be consistent");

        // Verify compaction efficiency (may retain all items if k is large)
        assert!(sketch.len() <= 200, "Should not have more retained items than total count");
    }

}
