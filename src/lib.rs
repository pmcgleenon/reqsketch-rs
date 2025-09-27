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

impl Default for RankAccuracy {
    fn default() -> Self {
        RankAccuracy::HighRank
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
    total_n: u64,
    min_item: Option<T>,
    max_item: Option<T>,
    compactors: Vec<compactor::Compactor<T>>,
    sorted_view_cache: Option<SortedView<T>>,
}

impl<T> ReqSketch<T>
where
    T: PartialOrd + Clone,
{
    /// Creates a new REQ sketch with default parameters.
    ///
    /// Uses k=12 (roughly 1% relative error at 95% confidence) and high rank accuracy.
    pub fn new() -> Self {
        ReqSketchBuilder::new().build().expect("Default parameters should always be valid")
    }

    /// Creates a new REQ sketch with the specified k parameter.
    ///
    /// # Arguments
    /// * `k` - Controls size and error of the sketch. Must be even and in range [4, 1024].
    ///         Value of 12 roughly corresponds to 1% relative error at 95% confidence.
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
        self.compactors.iter().map(|c| c.num_items()).sum()
    }

    /// Returns true if this sketch is in estimation mode (has compacted data).
    pub fn is_estimation_mode(&self) -> bool {
        self.compactors.len() > 1
    }

    /// Updates the sketch with a new item.
    pub fn update(&mut self, item: T) {
        // Update min/max
        match (&self.min_item, &self.max_item) {
            (None, None) => {
                self.min_item = Some(item.clone());
                self.max_item = Some(item.clone());
            }
            (Some(min), Some(max)) => {
                if item < *min {
                    self.min_item = Some(item.clone());
                }
                if item > *max {
                    self.max_item = Some(item.clone());
                }
            }
            _ => unreachable!(),
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
        if other.is_empty() {
            return Ok(());
        }

        // Validate compatibility
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
    /// # Arguments
    /// * `item` - The item to find the rank for
    /// * `criteria` - Whether to include the item's weight in the rank calculation
    ///
    /// # Returns
    /// A value in [0.0, 1.0] representing the approximate normalized rank.
    pub fn rank(&self, item: &T, criteria: SearchCriteria) -> Result<f64> {
        if self.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        let sorted_view = self.get_sorted_view()?;
        sorted_view.rank(item, criteria)
    }

    /// Returns the approximate rank of the given item using inclusive criteria.
    pub fn rank_inclusive(&self, item: &T) -> Result<f64> {
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
    pub fn quantile(&self, rank: f64, criteria: SearchCriteria) -> Result<T> {
        if self.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        if !(0.0..=1.0).contains(&rank) {
            return Err(ReqError::InvalidRank(rank));
        }

        let sorted_view = self.get_sorted_view()?;
        sorted_view.quantile(rank, criteria)
    }

    /// Returns the approximate quantile for the given normalized rank using inclusive criteria.
    pub fn quantile_inclusive(&self, rank: f64) -> Result<T> {
        self.quantile(rank, SearchCriteria::Inclusive)
    }

    /// Returns multiple quantiles for the given normalized ranks.
    pub fn quantiles(&self, ranks: &[f64], criteria: SearchCriteria) -> Result<Vec<T>> {
        ranks
            .iter()
            .map(|&rank| self.quantile(rank, criteria))
            .collect()
    }

    /// Returns the Probability Mass Function (PMF) for the given split points.
    pub fn pmf(&self, split_points: &[T], criteria: SearchCriteria) -> Result<Vec<f64>> {
        if self.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        let sorted_view = self.get_sorted_view()?;
        sorted_view.pmf(split_points, criteria)
    }

    /// Returns the Cumulative Distribution Function (CDF) for the given split points.
    pub fn cdf(&self, split_points: &[T], criteria: SearchCriteria) -> Result<Vec<f64>> {
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
    pub fn sorted_view(&self) -> Result<SortedView<T>> {
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

    fn get_sorted_view(&self) -> Result<SortedView<T>> {
        if self.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        // Always compute a fresh sorted view for now
        // TODO: In the future, we could use interior mutability for caching
        self.compute_sorted_view()
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
        let total_capacity: u32 = self.compactors.iter().map(|c| c.nominal_capacity()).sum();
        self.num_retained() >= total_capacity
    }

    fn compress(&mut self) {
        let mut level = 0;
        while level < self.compactors.len() {
            let needs_compaction = {
                let compactor = &self.compactors[level];
                compactor.num_items() >= compactor.nominal_capacity()
            };

            if needs_compaction {
                // Ensure we have a next level
                if level + 1 >= self.compactors.len() {
                    self.grow();
                }

                // Compact current level and promote to next
                let promoted = self.compactors[level].compact(self.rank_accuracy);
                self.compactors[level + 1].merge_sorted(&promoted);
            }

            level += 1;
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
}

impl<T> Default for ReqSketch<T>
where
    T: PartialOrd + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

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
    fn test_duplicate_values_debug() {
        let mut sketch = ReqSketch::new();

        // Add duplicate values like the failing test case
        for _ in 0..10 {
            sketch.update(0.0);
        }

        println!("Sketch with 10 zeros:");
        println!("  len: {}", sketch.len());
        println!("  estimation_mode: {}", sketch.is_estimation_mode());

        // Test rank-quantile consistency for various ranks
        for rank in [0.1, 0.2, 0.5, 0.8, 0.9] {
            let quantile_inc = sketch.quantile_inclusive(rank).unwrap();
            let rank_back = sketch.rank_inclusive(&quantile_inc).unwrap();

            println!("  rank {} -> quantile {} -> rank {}", rank, quantile_inc, rank_back);

            // Check if they're approximately equal (within reasonable tolerance)
            let diff = (rank - rank_back).abs();
            if diff > 0.1 {
                println!("    INCONSISTENCY: difference = {}", diff);
            }
        }
    }
}
