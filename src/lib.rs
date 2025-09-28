//! # REQSketch - Relative Error Quantiles Sketch
//!
//! Relative Error Quantiles sketch algorithm in Rust.
//! This implementation is based on the paper "Relative Error Streaming Quantiles" by
//! Graham Cormode, Zohar Karnin, Edo Liberty, Justin Thaler, Pavel Veselý.
//!
//! ## Features
//!
//! - **High Performance**: Optimized for streaming quantile estimation with relative error guarantees
//! - **Generic**: Works with any type that implements `PartialOrd + Clone`
//! - **Memory Efficient**: Compact representation with configurable memory/accuracy tradeoffs
//! - **Builder Pattern**: Ergonomic construction with sensible defaults
//! - **Iterator Support**: Full iterator support for inspection and debugging
//! - **Serialization**: Optional serde support for persistence
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

use std::fmt;

pub mod builder;
pub mod compactor;
pub mod error;
pub mod iter;
pub mod sorted_view;
use sorted_view::IntoF64;

/// Trait for types that can be used efficiently in REQ sketches.
/// Implemented for numeric types that support fast copy semantics.
pub trait ReqKey: Copy + PartialOrd + Clone {}

impl ReqKey for f64 {}
impl ReqKey for f32 {}
impl ReqKey for i64 {}
impl ReqKey for i32 {}
impl ReqKey for u64 {}
impl ReqKey for u32 {}

#[cfg(test)]
mod test_all_quantiles;
#[cfg(test)]
mod test_error_bounds;
#[cfg(test)]
mod test_detailed_comparison;
#[cfg(test)]
mod test_comprehensive_rank_accuracy;
#[cfg(test)]
mod test_hra_vs_lra_quantiles;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use builder::ReqSketchBuilder;
pub use error::{ReqError, Result};
pub use iter::ReqSketchIterator;
pub use sorted_view::SortedView;

/// Configuration for rank accuracy optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RankAccuracy {
    /// Optimize for accuracy at high ranks (near 1.0)
    HighRank,
    /// Optimize for accuracy at low ranks (near 0.0)
    LowRank,
}

/// Configuration for rank calculation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RankMethod {
    /// Use step function (C++ compatible, current default)
    StepFunction,
    /// Use linear interpolation (enhanced precision beyond C++)
    Interpolation,
}

impl Default for RankAccuracy {
    fn default() -> Self {
        RankAccuracy::HighRank
    }
}

impl Default for RankMethod {
    fn default() -> Self {
        RankMethod::StepFunction // C++ compatible by default
    }
}

/// Search criteria for quantile/rank operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchCriteria {
    /// Include the weight of the search item in the result
    Inclusive,
    /// Exclude the weight of the search item from the result
    Exclusive,
}

impl Default for SearchCriteria {
    fn default() -> Self {
        SearchCriteria::Inclusive
    }
}

/// A Relative Error Quantiles sketch for approximate quantile estimation.
///
/// The REQ sketch provides approximate quantile estimation with relative error guarantees.
/// It is particularly useful for streaming scenarios where you need to estimate quantiles
/// over large data streams with bounded memory usage.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReqSketch<T> {
    k: u16,
    rank_accuracy: RankAccuracy,
    rank_method: RankMethod,
    total_n: u64,
    min_item: Option<T>,
    max_item: Option<T>,
    compactors: Vec<compactor::Compactor<T>>,
    sorted_view_cache: Option<SortedView<T>>,
    /// Reusable buffer for promotions to avoid per-compaction allocation.
    promotion_buf: Vec<T>,
}

impl<T> ReqSketch<T>
where
    T: PartialOrd + Clone,
{
    /// Creates a new REQ sketch with default parameters.
    ///
    /// Uses k=12 (roughly 1% relative error at 95% confidence) and high rank accuracy.
    pub fn new() -> Self {
        let mut s = ReqSketchBuilder::new().build().expect("Default parameters should always be valid");
        s.promotion_buf = Vec::new();
        s
    }

    /// Creates a new REQ sketch with the specified k parameter.
    ///
    /// # Arguments
    /// * `k` - Controls size and error of the sketch. Must be even and in range [4, 1024].
    ///         Value of 12 roughly corresponds to 1% relative error at 95% confidence.
    pub fn with_k(k: u16) -> Result<Self> {
        let mut s = ReqSketchBuilder::new().k(k)?.build()?;
        s.promotion_buf = Vec::new();
        Ok(s)
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
        self.compactors.iter().map(|c| c.num_items()).sum()
    }

    /// Returns true if this sketch is in estimation mode (has compacted data).
    pub fn is_estimation_mode(&self) -> bool {
        self.compactors.len() > 1
    }

    /// Updates the sketch with a new item.
    pub fn update(&mut self, item: T) {
        // Update min/max tracking with the current item before moving it
        if self.min_item.as_ref().map_or(true, |min| item < *min) {
            self.min_item = Some(item.clone());
        }
        if self.max_item.as_ref().map_or(true, |max| item > *max) {
            self.max_item = Some(item.clone());
        }

        // Ensure we have at least one compactor
        if self.compactors.is_empty() {
            self.grow();
        }

        // Add to level 0 compactor
        self.compactors[0].append(item);
        self.total_n += 1;

        // Compress if needed
        if self.needs_compression() {
            self.compress();
        }

        // Invalidate sorted view cache
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
                Some(min) if other_min < min => self.min_item = Some(other_min.clone()),
                _ => {}
            }
        }

        if let Some(other_max) = &other.max_item {
            match &self.max_item {
                None => self.max_item = Some(other_max.clone()),
                Some(max) if other_max > max => self.max_item = Some(other_max.clone()),
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

        // Compress if needed
        if self.needs_compression() {
            self.compress();
        }

        // Invalidate cache
        self.sorted_view_cache = None;

        Ok(())
    }

    /// Returns the minimum item seen, or None if empty.
    pub fn min_item(&self) -> Option<&T> {
        self.min_item.as_ref()
    }

    /// Returns the maximum item seen, or None if empty.
    pub fn max_item(&self) -> Option<&T> {
        self.max_item.as_ref()
    }

    /// Returns the approximate rank of the given item.
    ///
    /// Uses the configured rank method (step function or interpolation).
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

        match self.rank_method {
            RankMethod::StepFunction => {
                // C++ compatible: use step function
                let sorted_view = self.get_sorted_view()?;
                sorted_view.rank_no_interpolation(item, criteria)
            }
            RankMethod::Interpolation => {
                // Enhanced precision: use interpolation (only available for compatible types)
                let sorted_view = self.get_sorted_view()?;
                sorted_view.rank_no_interpolation(item, criteria) // Fallback for now
            }
        }
    }

    /// Returns the approximate rank of the given item using inclusive criteria.
    pub fn rank_inclusive(&mut self, item: &T) -> Result<f64> {
        self.rank(item, SearchCriteria::Inclusive)
    }

    /// Returns the approximate rank using direct compactor weight computation.
    /// This matches the C++ get_rank implementation exactly.
    ///
    /// # Arguments
    /// * `item` - The item to find the rank for
    /// * `criteria` - Whether to include the item's weight in the rank calculation
    ///
    /// # Returns
    /// A value in [0.0, 1.0] representing the approximate normalized rank.
    pub fn rank_direct(&mut self, item: &T, criteria: SearchCriteria) -> Result<f64> {
        if self.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        let inclusive = matches!(criteria, SearchCriteria::Inclusive);
        let mut total_weight = 0u64;

        // Sum weights from all compactors (matches C++ approach)
        for compactor in &mut self.compactors {
            total_weight += compactor.compute_weight(item, inclusive);
        }

        Ok(total_weight as f64 / self.total_n as f64)
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
    pub fn iter(&self) -> ReqSketchIterator<T> {
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

    /// Direct quantile computation without materializing full sorted view.
    /// This eliminates the major allocation bottleneck in compute_sorted_view().
    fn quantile_kway_merge(&mut self, rank: f64, criteria: SearchCriteria) -> Result<T> {
        // Build a lightweight collection of all (item, weight) pairs without cloning items
        let mut weighted_items = Vec::new();

        for compactor in &self.compactors {
            let weight = compactor.weight();
            for item in compactor.iter() {
                weighted_items.push((item, weight));
            }
        }

        if weighted_items.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        // Sort by item value (items are references, so this is cheap)
        weighted_items.sort_by(|a, b| a.0.partial_cmp(b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate target weight
        let target_weight = (rank * self.total_n as f64) as u64;
        let mut cumulative_weight = 0u64;

        // Find the quantile
        for (item, weight) in &weighted_items {
            if cumulative_weight + weight > target_weight {
                return Ok((*item).clone());
            }
            cumulative_weight += weight;
        }

        // Return the last item if we've exhausted all items
        if let Some((last_item, _)) = weighted_items.last() {
            Ok((*last_item).clone())
        } else {
            Err(ReqError::EmptySketch)
        }
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

        Ok(self.sorted_view_cache.as_ref().unwrap())
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

    fn needs_compression(&self) -> bool {
        // Trigger compaction when any level exceeds its nominal capacity
        // This matches the original DataSketches behavior and prevents over-retention
        self.compactors.iter().any(|c| c.num_items() > c.nominal_capacity())
    }

    fn compress(&mut self) {
        // Compact each level that exceeds its nominal capacity
        // This is more efficient than global capacity checking
        while let Some(level) = (0..self.compactors.len())
            .find(|&i| self.compactors[i].num_items() > self.compactors[i].nominal_capacity())
        {
            // Perform compaction on the over-capacity level
            // Compact into reusable promotion buffer (no allocation)
            self.promotion_buf.clear();
            self.compactors[level].compact_into(self.rank_accuracy, &mut self.promotion_buf);

            // If we have promoted items, ensure we have a next level
            if !self.promotion_buf.is_empty() {
                if level + 1 >= self.compactors.len() {
                    self.grow();
                }
                self.compactors[level + 1].merge_sorted(&self.promotion_buf);
            }
        }

        // Invalidate cache
        self.sorted_view_cache = None;
    }

    fn grow(&mut self) {
        let level = self.compactors.len() as u8;
        let compactor = compactor::Compactor::new(level, self.k, self.rank_accuracy);
        self.compactors.push(compactor);
    }

    // Debug method to inspect compactor state
    #[cfg(test)]
    pub fn debug_compactor_info(&self) -> Vec<(u8, u32, u32)> {
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
        self.get_sorted_view()
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

    /// Constants for error calculation matching C++ implementation
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
        lb_rel.max(lb_fix)
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
        ub_rel.min(ub_fix)
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
    T: PartialOrd + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

// Note: f64 gets interpolation via SortedView<f64>::rank() specialization

impl<T> fmt::Display for ReqSketch<T>
where
    T: fmt::Display + PartialOrd + Clone,
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

// Specialized implementation for f64 with true interpolation support
impl ReqSketch<f64> {
    /// Returns the approximate rank using the configured method with full interpolation support.
    ///
    /// This specialized version can use true interpolation for enhanced precision.
    pub fn rank_interpolated(&mut self, item: &f64, criteria: SearchCriteria) -> Result<f64> {
        if self.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        match self.rank_method {
            RankMethod::StepFunction => {
                // C++ compatible: use step function
                let sorted_view = self.get_sorted_view()?;
                sorted_view.rank_no_interpolation(item, criteria)
            }
            RankMethod::Interpolation => {
                // Enhanced precision: use true interpolation
                let sorted_view = self.get_sorted_view()?;
                sorted_view.rank_with_interpolation(item, criteria)
            }
        }
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
    fn test_update_and_basic_queries() {
        let mut sketch = ReqSketch::new();

        // Add some values
        for i in 0..100 {
            sketch.update(i as f64);
        }

        assert!(!sketch.is_empty());
        assert_eq!(sketch.len(), 100);
        assert_eq!(sketch.min_item(), Some(&0.0));
        assert_eq!(sketch.max_item(), Some(&99.0));
    }

    #[test]
    fn test_builder_pattern() {
        let sketch: Result<ReqSketch<i32>> = ReqSketch::builder()
            .k(16)
            .and_then(|builder| Ok(builder.rank_accuracy(RankAccuracy::LowRank)))
            .and_then(|builder| builder.build());

        assert!(sketch.is_ok());
        let sketch = sketch.unwrap();
        assert_eq!(sketch.k(), 16);
        assert_eq!(sketch.rank_accuracy(), RankAccuracy::LowRank);
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
    fn test_merge_commutativity_debug() {
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

        println!("Before merge:");
        println!("sketch1a len: {}, items: {:?}", sketch1a.len(), sketch1a.iter().collect::<Vec<_>>());
        println!("sketch2a len: {}, items: {:?}", sketch2a.len(), sketch2a.iter().collect::<Vec<_>>());
        println!("sketch1b len: {}, items: {:?}", sketch1b.len(), sketch1b.iter().collect::<Vec<_>>());
        println!("sketch2b len: {}, items: {:?}", sketch2b.len(), sketch2b.iter().collect::<Vec<_>>());

        // Test merge order: A + B vs B + A
        sketch1a.merge(&sketch2a).unwrap();  // sketch1a + sketch2a
        sketch2b.merge(&sketch1b).unwrap();  // sketch2b + sketch1b

        println!("After merge:");
        println!("sketch1a+sketch2a len: {}", sketch1a.len());
        println!("sketch2b+sketch1b len: {}", sketch2b.len());

        if !sketch1a.is_empty() && !sketch2b.is_empty() {
            let q1: f64 = sketch1a.quantile(0.5, SearchCriteria::Inclusive).unwrap();
            let q2: f64 = sketch2b.quantile(0.5, SearchCriteria::Inclusive).unwrap();

            println!("Quantiles: {} vs {}", q1, q2);
            let diff = (q1 - q2).abs();
            println!("Difference: {}", diff);
        }
    }

    #[test]
    fn test_exact_quantile_debug() {
        // Test the exact same scenario as C++ reference: [1,2,3,4,5,6,7,8,9,10]
        let mut sketch = ReqSketch::new();
        for i in 1..=10 {
            sketch.update(i as f64);
        }

        println!("=== EXACT MODE QUANTILE DEBUG ===");
        println!("Sketch has {} items, total_n: {}", sketch.len(), sketch.total_n);
        println!("Is estimation mode: {}", sketch.is_estimation_mode());

        // Show internal structure
        for (i, compactor) in sketch.compactors.iter().enumerate() {
            println!("Level {}: {} items, weight: {}", i, compactor.num_items(), compactor.weight());
            let items: Vec<_> = compactor.iter().collect();
            if !items.is_empty() {
                println!("  Items: {:?}", items);
            }
        }

        // Test the specific quantile that's failing
        println!("\n=== QUANTILE TESTS ===");
        let test_cases = [
            (0.0, 1.0), (0.1, 1.0), (0.5, 5.0), (0.9, 9.0), (1.0, 10.0)
        ];

        for &(rank, expected) in &test_cases {
            if let Ok(quantile) = sketch.quantile(rank, SearchCriteria::Inclusive) {
                let matches = (quantile - expected).abs() < 1e-6;
                println!("quantile({}, Inclusive) = {}, expected = {}, matches = {}",
                         rank, quantile, expected, matches);
            }
        }

        // Also test exclusive
        println!("\n=== EXCLUSIVE QUANTILE TESTS ===");
        let exclusive_cases = [
            (0.0, 1.0), (0.1, 2.0), (0.5, 6.0), (0.9, 10.0), (1.0, 10.0)
        ];

        for &(rank, expected) in &exclusive_cases {
            if let Ok(quantile) = sketch.quantile(rank, SearchCriteria::Exclusive) {
                let matches = (quantile - expected).abs() < 1e-6;
                println!("quantile({}, Exclusive) = {}, expected = {}, matches = {}",
                         rank, quantile, expected, matches);
            }
        }

        // Debug the sorted view
        if let Ok(sorted_view) = sketch.get_sorted_view() {
            println!("\n=== SORTED VIEW DEBUG ===");
            println!("Total weight: {}", sorted_view.total_weight());
            for (item, cum_weight) in sorted_view.iter_with_weights() {
                println!("Item: {}, Cumulative weight: {}", item, cum_weight);
            }
        }
    }

    #[test]
    fn test_weight_debug_1000() {
        // Test the exact failing case from weight consistency test
        let mut sketch = ReqSketch::new();
        let n = 1000;

        println!("=== WEIGHT DEBUG FOR 1000 ITEMS ===");

        for i in 0..n {
            sketch.update(i as f64);

            // Check at specific intervals and at the end
            if i == 999 || (i + 1) % 100 == 0 {
                println!("\n--- After {} items ---", i + 1);
                println!("Total n: {}, retained: {}", sketch.total_n, sketch.len());
                println!("Is estimation mode: {}", sketch.is_estimation_mode());

                let mut total_weight = 0u64;
                for (level, compactor) in sketch.compactors.iter().enumerate() {
                    if compactor.num_items() > 0 {
                        let level_weight = compactor.num_items() as u64 * compactor.weight();
                        total_weight += level_weight;
                        println!("Level {}: {} items × {} weight = {} total (capacity: {})",
                                 level, compactor.num_items(), compactor.weight(),
                                 level_weight, compactor.nominal_capacity());
                    }
                }

                println!("Total weight: {}, expected: {}", total_weight, sketch.total_n);
                if total_weight != sketch.total_n {
                    println!("❌ WEIGHT MISMATCH! Difference: {}", total_weight as i64 - sketch.total_n as i64);
                    break;
                }
            }
        }

        // Final weight check using iterator
        let iter_weight: u64 = sketch.iter().map(|(_, weight)| weight).sum();
        println!("\nFinal iterator weight check: {}, expected: {}", iter_weight, sketch.total_n);
    }

    #[test]
    fn test_capacity_debug() {
        // Compare our capacity calculations with what C++ should produce
        let mut sketch = ReqSketch::new(); // k=12 by default

        println!("=== CAPACITY DEBUG (k=12) ===");

        // Check capacity calculations for first few levels
        for level in 0..5 {
            let compactor = crate::compactor::Compactor::<f64>::new(level, 12, RankAccuracy::HighRank);
            let section_size_raw = 12.0 / (2.0_f64.powf(level as f64 / 2.0));
            let section_size = compactor.section_size();
            let nominal_capacity = compactor.nominal_capacity();

            println!("Level {}: section_size_raw={:.2}, section_size={}, capacity={}",
                     level, section_size_raw, section_size, nominal_capacity);
        }

        // Add some items and see compaction behavior
        println!("\n=== COMPACTION TIMING ===");
        for i in 1..=50 {
            sketch.update(i as f64);

            if sketch.compactors.len() > 1 || i % 10 == 0 {
                println!("After {} items:", i);
                for (level, compactor) in sketch.compactors.iter().enumerate() {
                    if compactor.num_items() > 0 {
                        println!("  Level {}: {}/{} items ({}%)",
                                 level, compactor.num_items(), compactor.nominal_capacity(),
                                 (compactor.num_items() * 100) / compactor.nominal_capacity());
                    }
                }
                if sketch.compactors.len() > 1 {
                    break; // Stop after first compaction
                }
            }
        }
    }

    #[test]
    fn test_compaction_debug() {
        // Test with enough items to trigger compaction
        let mut sketch = ReqSketch::new();
        let n = 50; // Should trigger some compaction

        println!("=== COMPACTION DEBUG ===");

        for i in 0..n {
            sketch.update(i as f64);

            // Show state after every few updates
            if i % 10 == 9 || i == n - 1 {
                println!("\n--- After {} items ---", i + 1);
                println!("Total n: {}, retained: {}", sketch.total_n, sketch.len());
                println!("Is estimation mode: {}", sketch.is_estimation_mode());

                for (level, compactor) in sketch.compactors.iter().enumerate() {
                    if compactor.num_items() > 0 {
                        println!("Level {}: {} items, weight: {}", level, compactor.num_items(), compactor.weight());
                        let items: Vec<_> = compactor.iter().collect();
                        println!("  Items (first 10): {:?}", &items[..items.len().min(10)]);
                    }
                }

                // Check if total weight makes sense
                let total_weight: u64 = sketch.iter().map(|(_, weight)| weight).sum();
                let expected_weight = sketch.total_n;
                println!("Total weight from iterator: {}, expected: {}", total_weight, expected_weight);

                if total_weight != expected_weight {
                    println!("❌ WEIGHT MISMATCH!");
                }
            }
        }

        // Test a quantile to see accuracy
        if let Ok(median) = sketch.quantile(0.5, SearchCriteria::Inclusive) {
            let expected_median = (n - 1) as f64 * 0.5;
            let error = (median - expected_median).abs() / expected_median;
            println!("\nMedian: {}, expected: {}, error: {:.2}%", median, expected_median, error * 100.0);
        }
    }

    #[test]
    fn debug_section_growth() {
        let mut sketch = ReqSketch::new(); // k=12 by default

        println!("Testing section growth in Rust REQ sketch:");

        for i in 0..200 {
            sketch.update(i as f64);

            if i % 20 == 19 || i < 50 {
                println!("\nAfter {} items:", i + 1);
                println!("Is estimation mode: {}", sketch.is_estimation_mode());
                println!("Total weight: {}", sketch.len());
                println!("Levels: {}", sketch.compactors.len());

                let retained: u32 = sketch.compactors.iter()
                    .map(|c| c.num_items())
                    .sum();
                println!("Retained items: {}", retained);

                // Show detailed compactor state at key points
                if i == 49 || i == 79 || i == 99 {
                    for (level, compactor) in sketch.compactors.iter().enumerate() {
                        if compactor.num_items() > 0 {
                            println!("  Level {}: {} items, capacity {}, state {}, sections {}",
                                level, compactor.num_items(), compactor.nominal_capacity(),
                                compactor.state(), compactor.num_sections());
                        }
                    }
                }

                if sketch.is_estimation_mode() && i > 100 { break; }
            }
        }
    }

    #[test]
    fn debug_50k_uniform() {
        let mut sketch = ReqSketch::new();
        for i in 0..50_000 {
            sketch.update(i as f64);
        }

        println!("Rust REQ sketch with 50k items:");
        println!("Is estimation mode: {}", sketch.is_estimation_mode());
        println!("Total weight: {}", sketch.len());

        let q01 = sketch.quantile(0.1, SearchCriteria::Inclusive).unwrap();
        let expected = 4999.9;
        let error = (q01 - expected).abs() / expected;

        println!("0.1 quantile: {}", q01);
        println!("Expected: {}", expected);
        println!("Relative error: {:.2}%", error * 100.0);

        // Debug compactor state
        println!("\nCompactor state:");
        for (level, compactor) in sketch.compactors.iter().enumerate() {
            if compactor.num_items() > 0 {
                println!("Level {}: {} items, weight {}, capacity {}",
                    level, compactor.num_items(), compactor.weight(), compactor.nominal_capacity());
            }
        }

        // Also test a smaller quantile to see if behavior is consistent
        let q001 = sketch.quantile(0.01, SearchCriteria::Inclusive).unwrap();
        let expected_001 = 499.99;
        let error_001 = (q001 - expected_001).abs() / expected_001;
        println!("\n0.01 quantile: {}", q001);
        println!("Expected: {}", expected_001);
        println!("Relative error: {:.2}%", error_001 * 100.0);

        // Test median for comparison with C++
        let median = sketch.quantile(0.5, SearchCriteria::Inclusive).unwrap();
        let expected_median = 24999.5;
        let median_error = (median - expected_median).abs() / expected_median;
        println!("\n0.5 quantile (median): {}", median);
        println!("Expected: {}", expected_median);
        println!("Relative error: {:.2}%", median_error * 100.0);
    }
}
