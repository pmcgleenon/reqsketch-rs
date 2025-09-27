//! Sorted view implementation for efficient quantile queries.

use crate::{ReqError, Result, SearchCriteria};
use std::cmp::Ordering;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A sorted view of all items in the sketch with their cumulative weights.
///
/// This provides an efficient representation for quantile and rank queries
/// by maintaining items in sorted order with precomputed cumulative weights.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SortedView<T> {
    /// Items in sorted order
    items: Vec<T>,
    /// Cumulative weights for each item
    cumulative_weights: Vec<u64>,
    /// Total weight of all items
    total_weight: u64,
}

impl<T> SortedView<T>
where
    T: PartialOrd + Clone,
{
    /// Creates a new sorted view from weighted items.
    ///
    /// # Arguments
    /// * `weighted_items` - Vector of (item, weight) pairs
    ///
    /// The items will be sorted and cumulative weights computed.
    pub fn new(mut weighted_items: Vec<(T, u64)>) -> Self {
        if weighted_items.is_empty() {
            return Self {
                items: Vec::new(),
                cumulative_weights: Vec::new(),
                total_weight: 0,
            };
        }

        // Sort by item value
        weighted_items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let mut items = Vec::with_capacity(weighted_items.len());
        let mut cumulative_weights = Vec::with_capacity(weighted_items.len());
        let mut cumulative_weight = 0u64;

        for (item, weight) in weighted_items {
            cumulative_weight += weight;
            items.push(item);
            cumulative_weights.push(cumulative_weight);
        }

        Self {
            items,
            cumulative_weights,
            total_weight: cumulative_weight,
        }
    }

    /// Returns the number of items in the sorted view.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns true if the sorted view is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Returns the total weight of all items.
    pub fn total_weight(&self) -> u64 {
        self.total_weight
    }

    /// Returns the approximate rank of the given item.
    ///
    /// # Arguments
    /// * `item` - The item to find the rank for
    /// * `criteria` - Whether to include the item's weight in the rank
    ///
    /// # Returns
    /// A normalized rank in [0.0, 1.0]
    pub fn rank(&self, item: &T, criteria: SearchCriteria) -> Result<f64> {
        if self.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        match criteria {
            SearchCriteria::Inclusive => {
                // For consistency with quantile method, use the logic that matches
                // how quantile selects items based on cumulative weights

                // Find the first occurrence of the target item
                for i in 0..self.items.len() {
                    if self.items[i] == *item {
                        // Return the rank corresponding to this item's cumulative weight
                        return Ok(self.cumulative_weights[i] as f64 / self.total_weight as f64);
                    } else if self.items[i] > *item {
                        // Item not found - return the rank just before this position
                        let prev_weight = if i == 0 { 0 } else { self.cumulative_weights[i - 1] };
                        return Ok(prev_weight as f64 / self.total_weight as f64);
                    }
                }

                // All items are smaller than target
                Ok(1.0)
            }
            SearchCriteria::Exclusive => {
                // Find the last occurrence of items < target (strictly less than)
                let mut cumulative_weight = 0u64;
                for i in 0..self.items.len() {
                    if self.items[i] < *item {
                        cumulative_weight = self.cumulative_weights[i];
                    } else {
                        break;
                    }
                }
                Ok(cumulative_weight as f64 / self.total_weight as f64)
            }
        }
    }

    /// Returns the approximate quantile for the given normalized rank.
    ///
    /// # Arguments
    /// * `rank` - A normalized rank in [0.0, 1.0]
    /// * `criteria` - Search criteria for quantile selection
    ///
    /// # Returns
    /// The item at approximately the given rank
    pub fn quantile(&self, rank: f64, criteria: SearchCriteria) -> Result<T> {
        if self.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        if !(0.0..=1.0).contains(&rank) {
            return Err(ReqError::InvalidRank(rank));
        }

        // Handle edge cases
        if rank == 0.0 {
            match criteria {
                SearchCriteria::Inclusive => return Ok(self.items[0].clone()),
                SearchCriteria::Exclusive => return Ok(self.items[0].clone()),
            }
        }
        if rank == 1.0 {
            return Ok(self.items[self.items.len() - 1].clone());
        }

        // Convert rank to target cumulative weight
        let target_weight = match criteria {
            SearchCriteria::Inclusive => (rank * self.total_weight as f64).ceil() as u64,
            SearchCriteria::Exclusive => (rank * self.total_weight as f64) as u64,
        };

        // Find the first item whose cumulative weight >= target_weight
        for i in 0..self.cumulative_weights.len() {
            match criteria {
                SearchCriteria::Inclusive => {
                    if self.cumulative_weights[i] >= target_weight {
                        return Ok(self.items[i].clone());
                    }
                }
                SearchCriteria::Exclusive => {
                    if self.cumulative_weights[i] > target_weight {
                        return Ok(self.items[i].clone());
                    }
                }
            }
        }

        // Fallback to last item
        Ok(self.items[self.items.len() - 1].clone())
    }

    /// Returns the Probability Mass Function (PMF) for the given split points.
    ///
    /// # Arguments
    /// * `split_points` - Array of split points that divide the domain
    /// * `criteria` - Search criteria for boundary handling
    ///
    /// # Returns
    /// Array of probabilities for each interval defined by the split points
    pub fn pmf(&self, split_points: &[T], criteria: SearchCriteria) -> Result<Vec<f64>> {
        if self.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        self.validate_split_points(split_points)?;

        let mut result = Vec::with_capacity(split_points.len() + 1);
        let mut prev_rank = 0.0;

        for split_point in split_points {
            let rank = self.rank(split_point, criteria)?;
            result.push(rank - prev_rank);
            prev_rank = rank;
        }

        // Add the final interval
        result.push(1.0 - prev_rank);

        Ok(result)
    }

    /// Returns the Cumulative Distribution Function (CDF) for the given split points.
    ///
    /// # Arguments
    /// * `split_points` - Array of split points that divide the domain
    /// * `criteria` - Search criteria for boundary handling
    ///
    /// # Returns
    /// Array of cumulative probabilities at each split point
    pub fn cdf(&self, split_points: &[T], criteria: SearchCriteria) -> Result<Vec<f64>> {
        if self.is_empty() {
            return Err(ReqError::EmptySketch);
        }

        self.validate_split_points(split_points)?;

        let mut result = Vec::with_capacity(split_points.len() + 1);
        let mut cumulative = 0.0;

        let pmf = self.pmf(split_points, criteria)?;
        for mass in pmf {
            cumulative += mass;
            result.push(cumulative);
        }

        Ok(result)
    }

    /// Returns an iterator over the items in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    /// Returns an iterator over (item, cumulative_weight) pairs.
    pub fn iter_with_weights(&self) -> impl Iterator<Item = (&T, u64)> {
        self.items.iter().zip(self.cumulative_weights.iter().copied())
    }

    // Private helper methods

    fn validate_split_points(&self, split_points: &[T]) -> Result<()> {
        // Check that split points are monotonically increasing
        for i in 1..split_points.len() {
            if split_points[i - 1] >= split_points[i] {
                return Err(ReqError::InvalidSplitPoints(
                    "Split points must be unique and monotonically increasing".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_view() -> SortedView<i32> {
        let weighted_items = vec![
            (1, 1), (3, 1), (5, 1), (7, 1), (9, 1)
        ];
        SortedView::new(weighted_items)
    }

    #[test]
    fn test_sorted_view_creation() {
        let view = create_test_view();
        assert_eq!(view.len(), 5);
        assert_eq!(view.total_weight(), 5);
        assert!(!view.is_empty());
    }

    #[test]
    fn test_rank_queries() {
        let view = create_test_view();

        // Test exact matches
        assert!((view.rank(&1, SearchCriteria::Inclusive).unwrap() - 0.2).abs() < 1e-10);
        assert!((view.rank(&1, SearchCriteria::Exclusive).unwrap() - 0.0).abs() < 1e-10);

        // Test values between items
        assert!((view.rank(&2, SearchCriteria::Inclusive).unwrap() - 0.2).abs() < 1e-10);
        assert!((view.rank(&6, SearchCriteria::Inclusive).unwrap() - 0.6).abs() < 1e-10);

        // Test edge cases
        assert!((view.rank(&0, SearchCriteria::Inclusive).unwrap() - 0.0).abs() < 1e-10);
        assert!((view.rank(&10, SearchCriteria::Inclusive).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_queries() {
        let view = create_test_view();

        // Test edge cases
        assert_eq!(view.quantile(0.0, SearchCriteria::Inclusive).unwrap(), 1);
        assert_eq!(view.quantile(1.0, SearchCriteria::Inclusive).unwrap(), 9);

        // Test middle values
        let median = view.quantile(0.5, SearchCriteria::Inclusive).unwrap();
        assert!(median >= 3 && median <= 7); // Should be around the middle (values are 1,3,5,7,9)

        // Test various ranks
        let q25 = view.quantile(0.25, SearchCriteria::Inclusive).unwrap();
        let q75 = view.quantile(0.75, SearchCriteria::Inclusive).unwrap();
        assert!(q25 <= median);
        assert!(median <= q75);
    }

    #[test]
    fn test_pmf() {
        let view = create_test_view();
        let split_points = vec![3, 7];

        let pmf = view.pmf(&split_points, SearchCriteria::Inclusive).unwrap();
        assert_eq!(pmf.len(), 3); // 2 split points create 3 intervals

        // Sum should be approximately 1.0
        let sum: f64 = pmf.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cdf() {
        let view = create_test_view();
        let split_points = vec![3, 7];

        let cdf = view.cdf(&split_points, SearchCriteria::Inclusive).unwrap();
        assert_eq!(cdf.len(), 3);

        // CDF should be monotonically increasing
        for i in 1..cdf.len() {
            assert!(cdf[i] >= cdf[i - 1]);
        }

        // Last value should be 1.0
        assert!((cdf[cdf.len() - 1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_view() {
        let view: SortedView<i32> = SortedView::new(vec![]);
        assert!(view.is_empty());
        assert_eq!(view.len(), 0);
        assert_eq!(view.total_weight(), 0);

        // Operations on empty view should return errors
        assert!(view.rank(&5, SearchCriteria::Inclusive).is_err());
        assert!(view.quantile(0.5, SearchCriteria::Inclusive).is_err());
    }
}