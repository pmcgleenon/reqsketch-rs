//! Compactor implementation for REQ sketch levels.
//!
//! Each level in the REQ sketch uses a compactor to maintain a bounded set of items
//! with deterministic compaction when capacity is exceeded.

use crate::{RankAccuracy, Result};
// use rand::Rng;  // Will be used later for randomized compaction
use std::cmp::Ordering;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A compactor maintains items at a specific level of the REQ sketch.
///
/// When the compactor reaches its nominal capacity, it performs compaction
/// by keeping approximately half the items and promoting the rest to the next level.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Compactor<T> {
    /// The level of this compactor (0 = base level)
    lg_weight: u8,
    /// Whether this compactor is configured for high rank accuracy
    rank_accuracy: RankAccuracy,
    /// Raw section size (may be fractional)
    section_size_raw: f32,
    /// Actual section size (rounded to integer)
    section_size: u32,
    /// Number of sections in this compactor
    num_sections: u8,
    /// State for deterministic compaction
    state: u64,
    /// Current items in the compactor
    items: Vec<T>,
    /// Whether items are currently sorted
    is_sorted: bool,
}

impl<T> Compactor<T>
where
    T: PartialOrd + Clone,
{
    /// Creates a new compactor for the given level.
    ///
    /// # Arguments
    /// * `lg_weight` - The level (log weight) of this compactor
    /// * `k` - The k parameter from the parent sketch
    /// * `rank_accuracy` - Rank accuracy configuration
    pub fn new(lg_weight: u8, k: u16, rank_accuracy: RankAccuracy) -> Self {
        let section_size_raw = (k as f32) / (2.0_f32.powf(lg_weight as f32 / 2.0));
        let section_size = nearest_even(section_size_raw);
        let num_sections = 3; // Initial number of sections

        Self {
            lg_weight,
            rank_accuracy,
            section_size_raw,
            section_size,
            num_sections,
            state: 0,
            items: Vec::new(),
            is_sorted: true, // Start sorted (empty)
        }
    }

    /// Returns the level (log weight) of this compactor.
    pub fn lg_weight(&self) -> u8 {
        self.lg_weight
    }

    /// Returns the number of items currently in this compactor.
    pub fn num_items(&self) -> u32 {
        self.items.len() as u32
    }

    /// Returns the nominal capacity of this compactor.
    pub fn nominal_capacity(&self) -> u32 {
        // C++ uses MULTIPLIER = 2
        2 * self.section_size * self.num_sections as u32
    }

    /// Returns the section size of this compactor.
    pub fn section_size(&self) -> u32 {
        self.section_size
    }

    /// Returns whether the items are currently sorted.
    pub fn is_sorted(&self) -> bool {
        self.is_sorted
    }

    /// Appends an item to this compactor.
    pub fn append(&mut self, item: T) {
        self.items.push(item);
        if self.items.len() > 1 {
            self.is_sorted = false;
        }
    }

    /// Merges items from another compactor into this one.
    pub fn merge(&mut self, other: &Self) -> Result<()> {
        self.items.extend_from_slice(&other.items);
        if !other.items.is_empty() {
            self.is_sorted = false;
        }
        Ok(())
    }

    /// Merges pre-sorted items into this compactor.
    pub fn merge_sorted(&mut self, items: &[T]) {
        if items.is_empty() {
            return;
        }

        if self.items.is_empty() {
            self.items.extend_from_slice(items);
            self.is_sorted = true;
            return;
        }

        // If we're sorted and the new items are sorted, we can merge efficiently
        if self.is_sorted {
            let mut merged = Vec::with_capacity(self.items.len() + items.len());
            let mut i = 0;
            let mut j = 0;

            while i < self.items.len() && j < items.len() {
                if self.items[i] <= items[j] {
                    merged.push(self.items[i].clone());
                    i += 1;
                } else {
                    merged.push(items[j].clone());
                    j += 1;
                }
            }

            while i < self.items.len() {
                merged.push(self.items[i].clone());
                i += 1;
            }

            while j < items.len() {
                merged.push(items[j].clone());
                j += 1;
            }

            self.items = merged;
        } else {
            // Just append and sort later
            self.items.extend_from_slice(items);
            self.is_sorted = false;
        }
    }

    /// Sorts the items in this compactor if not already sorted.
    pub fn sort(&mut self) {
        if !self.is_sorted {
            self.items.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            self.is_sorted = true;
        }
    }

    /// Compacts this compactor and returns the promoted items.
    ///
    /// This operation compacts a section of items (not the entire level) and promotes
    /// approximately half of the compacted items to the next level.
    /// The weight is conserved exactly: N compacted items -> N/2 promoted items with 2x weight
    pub fn compact(&mut self, _rank_accuracy: RankAccuracy) -> Vec<T> {
        if self.items.is_empty() {
            return Vec::new();
        }

        // Ensure items are sorted
        self.sort();

        // Calculate sections to compact based on state (C++ logic)
        let secs_to_compact = ((!self.state).trailing_zeros() + 1).min(self.num_sections as u32) as u8;
        let compaction_range = self.compute_compaction_range(secs_to_compact);

        // Must have at least 2 items to compact
        if compaction_range.1 <= compaction_range.0 || (compaction_range.1 - compaction_range.0) < 2 {
            return Vec::new();
        }

        // Ensure enough sections for growth
        self.ensure_enough_sections();

        // Promote every other item from the compaction range
        let promoted = self.promote_evens_or_odds(
            &self.items[compaction_range.0..compaction_range.1],
            _rank_accuracy
        );

        // Remove the entire compaction range (both promoted and non-promoted items)
        // This is the key insight: we remove ALL items in the range, not just the non-promoted ones
        self.items.drain(compaction_range.0..compaction_range.1);

        // Update state
        self.state += 1;

        promoted
    }

    /// Returns an iterator over the items in this compactor.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    /// Returns the weight (2^lg_weight) for items in this compactor.
    pub fn weight(&self) -> u64 {
        1u64 << self.lg_weight
    }

    /// Returns the current state for debugging.
    #[cfg(test)]
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Returns the number of sections for debugging.
    #[cfg(test)]
    pub fn num_sections(&self) -> u8 {
        self.num_sections
    }

    // Private helper methods

    fn ensure_enough_sections(&mut self) -> bool {
        let ssr = self.section_size_raw / 2.0_f32.sqrt();
        let ne = nearest_even(ssr);
        if self.state >= (1u64 << (self.num_sections - 1)) && ne >= 4 { // MIN_K equivalent
            self.section_size_raw = ssr;
            self.section_size = ne;
            self.num_sections *= 2;
            true
        } else {
            false
        }
    }

    fn compute_compaction_range(&self, secs_to_compact: u8) -> (usize, usize) {
        // Implement C++ logic exactly
        let nom_capacity = self.nominal_capacity() as usize;
        let mut non_compact = nom_capacity / 2 + (self.num_sections - secs_to_compact) as usize * self.section_size as usize;

        // Ensure non_compact doesn't exceed items length
        non_compact = non_compact.min(self.items.len());

        // Make compacted region even (ensure even number of items to compact)
        if self.items.len() > non_compact && ((self.items.len() - non_compact) & 1) == 1 {
            non_compact += 1;
            non_compact = non_compact.min(self.items.len()); // Ensure we don't exceed bounds
        }

        // For HRA (High Rank Accuracy) vs LRA (Low Rank Accuracy)
        // This determines which end of the sorted array to compact
        let (low, high) = match self.rank_accuracy {
            RankAccuracy::HighRank => {
                // HRA: compact from the beginning (lower values)
                (0, (self.items.len() - non_compact).min(self.items.len()))
            }
            RankAccuracy::LowRank => {
                // LRA: compact from the end (higher values)
                (non_compact.min(self.items.len()), self.items.len())
            }
        };

        (low, high)
    }

    fn promote_evens_or_odds(&self, items: &[T], _rank_accuracy: RankAccuracy) -> Vec<T> {
        if items.is_empty() {
            return Vec::new();
        }

        // Implement C++ promote_evens_or_odds logic exactly
        // Determine coin flip: for odd state flip coin, for even use random
        let odds = if (self.state & 1) == 1 {
            // For odd state, flip the previous coin value (deterministic)
            !(self.state & 2 == 2) // Use bit 1 as previous coin state
        } else {
            // For even state, use deterministic based on state for reproducibility
            // In C++ this would be random, but for deterministic behavior we use state
            (self.state >> 2) & 1 == 1
        };

        let mut result = Vec::new();
        let mut i = if odds { 1 } else { 0 };

        while i < items.len() {
            result.push(items[i].clone());
            i += 2; // Skip every other item
        }

        result
    }
}

/// Rounds a float to the nearest even integer, matching C++ implementation.
/// C++ version: static_cast<uint32_t>(round(value / 2)) << 1
pub(crate) fn nearest_even(value: f32) -> u32 {
    let result = ((value / 2.0).round() as u32) * 2;

    // Ensure minimum value of 2 (since k must be at least 4, section sizes should be at least 2)
    result.max(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_compactor() {
        let compactor: Compactor<i32> = Compactor::new(0, 12, RankAccuracy::HighRank);
        assert_eq!(compactor.lg_weight(), 0);
        assert_eq!(compactor.num_items(), 0);
        assert!(compactor.is_sorted());
        assert_eq!(compactor.weight(), 1);
    }

    #[test]
    fn test_append_and_sort() {
        let mut compactor = Compactor::new(0, 12, RankAccuracy::HighRank);

        compactor.append(5);
        assert_eq!(compactor.num_items(), 1);
        assert!(compactor.is_sorted()); // Single item is sorted

        compactor.append(3);
        assert_eq!(compactor.num_items(), 2);
        assert!(!compactor.is_sorted()); // Multiple items, not sorted

        compactor.sort();
        assert!(compactor.is_sorted());

        let items: Vec<&i32> = compactor.iter().collect();
        assert_eq!(items, vec![&3, &5]);
    }

    #[test]
    fn test_nearest_even() {
        assert_eq!(nearest_even(1.0), 2); // minimum of 2
        assert_eq!(nearest_even(2.0), 2);
        assert_eq!(nearest_even(3.0), 4); // 3/2=1.5, round(1.5)=2, 2*2=4 (C++ behavior)
        assert_eq!(nearest_even(4.0), 4);
        assert_eq!(nearest_even(4.6), 4);
        assert_eq!(nearest_even(5.6), 6);
        assert_eq!(nearest_even(13.0), 14); // 13/2=6.5, round(6.5)=7, 7*2=14 (C++ behavior)
    }

    #[test]
    fn test_merge_sorted() {
        let mut compactor = Compactor::new(0, 12, RankAccuracy::HighRank);

        compactor.append(1);
        compactor.append(3);
        compactor.append(5);
        compactor.sort();

        let other_items = vec![2, 4, 6];
        compactor.merge_sorted(&other_items);

        assert!(compactor.is_sorted());
        let items: Vec<&i32> = compactor.iter().collect();
        assert_eq!(items, vec![&1, &2, &3, &4, &5, &6]);
    }
}