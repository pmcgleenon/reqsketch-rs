//! Compactor implementation for REQ sketch levels.
//!
//! Each level in the REQ sketch uses a compactor to maintain a bounded set of items
//! with deterministic compaction when capacity is exceeded.

use crate::{RankAccuracy, Result};
// use rand::Rng;  // Will be used later for randomized compaction
use std::cmp::Ordering;

/// C++ compatible nearest_even function
fn nearest_even(value: f32) -> u32 {
    let result = ((value / 2.0).round() as u32) * 2;
    // Ensure minimum value of 2 (since k must be at least 4, section sizes should be at least 2)
    result.max(2)
}

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A compactor maintains items at a specific level of the REQ sketch.
///
/// When the compactor reaches its nominal capacity, it performs compaction
/// by keeping approximately half the items and promoting the rest to the next level.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Compactor<T> {
    // HOT FIELDS - accessed every insert/compaction, grouped for cache locality
    /// Current items in the compactor
    items: Vec<T>,
    /// Whether items are currently sorted
    is_sorted: bool,
    /// State for deterministic compaction
    state: u64,
    /// Reusable scratch buffer for compaction operations
    #[cfg_attr(feature = "serde", serde(skip))]
    scratch_buffer: Vec<T>,

    // WARM FIELDS - used during compaction calculations
    /// Actual section size (rounded to integer)
    section_size: u32,
    /// Number of sections in this compactor
    num_sections: u8,
    /// The level of this compactor (0 = base level)
    lg_weight: u8,

    // COLD FIELDS - configuration, rarely accessed after construction
    /// Whether this compactor is configured for high rank accuracy
    rank_accuracy: RankAccuracy,
    /// Raw section size (may be fractional)
    section_size_raw: f32,
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
        let num_sections = 3u8; // Initial number of sections

        // Pre-reserve capacity to eliminate Vec reallocations
        let nominal: usize = (2 * section_size * num_sections as u32) as usize;

        Self {
            // HOT FIELDS first
            items: Vec::with_capacity(nominal),
            is_sorted: true, // Start sorted (empty)
            state: 0,
            scratch_buffer: Vec::with_capacity(nominal / 2 + 8), // typical promotion size

            // WARM FIELDS
            section_size,
            num_sections,
            lg_weight,

            // COLD FIELDS
            rank_accuracy,
            section_size_raw,
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
    #[inline(always)]
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
    /// Merges sorted items into this compactor using scratch buffer to avoid allocation.
    /// Both this compactor's items and the input must be sorted.
    #[inline(always)]
    pub fn merge_sorted(&mut self, items: &[T]) {
        if items.is_empty() {
            return;
        }

        if self.items.is_empty() {
            self.items.extend_from_slice(items);
            self.is_sorted = true;
            return;
        }

        // Ensure sorted on both inputs by contract
        let total = self.items.len() + items.len();
        self.scratch_buffer.clear();
        if self.scratch_buffer.capacity() < total {
            self.scratch_buffer.reserve(total - self.scratch_buffer.capacity());
        }

        let (mut i, mut j) = (0usize, 0usize);
        let (a, b) = (&self.items, items);

        // Two-pointer merge into scratch buffer
        while i < a.len() && j < b.len() {
            if a[i].partial_cmp(&b[j]).unwrap_or(std::cmp::Ordering::Equal).is_le() {
                self.scratch_buffer.push(a[i].clone());
                i += 1;
            } else {
                self.scratch_buffer.push(b[j].clone());
                j += 1;
            }
        }

        // Add remaining elements
        if i < a.len() {
            self.scratch_buffer.extend_from_slice(&a[i..]);
        }
        if j < b.len() {
            self.scratch_buffer.extend_from_slice(&b[j..]);
        }

        // Swap scratch buffer with items (zero-copy)
        self.items.clear();
        std::mem::swap(&mut self.items, &mut self.scratch_buffer);
        self.is_sorted = true;
    }

    /// Sorts the items in this compactor if not already sorted.
    #[inline(always)]
    pub fn sort(&mut self) {
        if !self.is_sorted {
            // Use unstable sort for better performance (stable not needed for REQ sketch)
            self.items.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.is_sorted = true;
        }
    }

    /// Compacts into the provided output buffer without allocating.
    /// Writes promoted items into `out` and removes the compacted range in-place via `copy_within + truncate`.
    #[inline(always)]
    pub fn compact_into(&mut self, _rank_accuracy: RankAccuracy, out: &mut Vec<T>) {
        if self.items.is_empty() {
            out.clear();
            return;
        }

        // Calculate sections to compact based on state (C++ logic)
        let secs_to_compact = ((!self.state).trailing_zeros() + 1).min(self.num_sections as u32) as u8;
        let compaction_range = self.compute_compaction_range(secs_to_compact);

        // Sort only the compaction range (avoid sorting the whole level)
        self.items[compaction_range.0..compaction_range.1]
            .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.is_sorted = false; // level as a whole might no longer be fully sorted

        // Must have at least 2 items to compact
        if compaction_range.1 <= compaction_range.0 || (compaction_range.1 - compaction_range.0) < 2 {
            out.clear();
            return;
        }

        // Ensure enough sections for growth
        self.ensure_enough_sections();

        // Even/odd choice (same logic you had)
        let odds = if (self.state & 1) == 1 {
            (self.state >> 1) & 1 == 0
        } else {
            ((self.state >> 1) ^ (self.state >> 3) ^ (self.state >> 7)) & 1 == 1
        };

        // Build promoted items directly into output buffer (no alloc)
        out.clear();
        let (start, end) = compaction_range;
        let mut i = start + if odds { 1 } else { 0 };
        while i < end {
            out.push(self.items[i].clone()); // TODO: use Copy fast-path for numeric types
            i += 2;
        }

        // Remove the compacted range in-place by rotating elements left
        let removed = end - start;
        if end < self.items.len() {
            // Use rotate_left to move tail elements to fill the gap
            self.items[start..].rotate_left(removed);
        }
        self.items.truncate(self.items.len() - removed);

        // Update state
        self.state += 1;
    }

    /// Back-compat wrapper (allocates if used). Prefer `compact_into` for performance.
    #[inline(always)]
    pub fn compact(&mut self, _rank_accuracy: RankAccuracy) -> Vec<T> {
        let mut out = Vec::new();
        self.compact_into(_rank_accuracy, &mut out);
        out
    }

    /// Returns an iterator over the items in this compactor.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    /// Returns a slice of items for zero-allocation iteration.
    pub(crate) fn items_slice(&self) -> &[T] {
        &self.items
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
        // Implement exact C++ section growth algorithm
        let ssr = self.section_size_raw / (2.0_f32).sqrt();
        let ne = nearest_even(ssr);

        const MIN_K: u32 = 4; // From C++ req_constants::MIN_K

        if self.state >= (1u64 << (self.num_sections - 1)) && ne >= MIN_K {
            self.section_size_raw = ssr;
            self.section_size = ne;
            self.num_sections <<= 1; // Double the sections
            true
        } else {
            false
        }
    }

    #[inline(always)]
    fn compute_compaction_range(&self, secs_to_compact: u8) -> (usize, usize) {
        // Implement exact C++ logic from compute_compaction_range
        let nom_capacity = self.nominal_capacity() as usize;
        let mut non_compact = nom_capacity / 2 + (self.num_sections - secs_to_compact) as usize * self.section_size as usize;

        // Ensure bounds first to prevent overflow
        non_compact = non_compact.min(self.items.len());

        // Make compacted region even - C++ logic (only if we have enough items)
        if self.items.len() > non_compact && ((self.items.len() - non_compact) & 1) == 1 {
            non_compact += 1;
            non_compact = non_compact.min(self.items.len()); // Ensure bounds again
        }

        // C++ logic: const uint32_t low = hra_ ? 0 : non_compact;
        //           const uint32_t high = hra_ ? num_items_ - non_compact : num_items_;
        let (low, high) = match self.rank_accuracy {
            RankAccuracy::HighRank => {
                // HRA: low = 0, high = num_items - non_compact
                let high = if self.items.len() >= non_compact {
                    self.items.len() - non_compact
                } else {
                    0
                };
                (0, high)
            }
            RankAccuracy::LowRank => {
                // LRA: low = non_compact, high = num_items
                let low = non_compact.min(self.items.len());
                (low, self.items.len())
            }
        };

        (low, high)
    }

    fn promote_evens_or_odds(&mut self, items: &[T], _rank_accuracy: RankAccuracy) -> &Vec<T> {
        // Clear and reuse scratch buffer to avoid allocation
        self.scratch_buffer.clear();

        if items.is_empty() {
            return &self.scratch_buffer;
        }

        // Implement exact C++ coin flip logic
        // From C++ code: if ((state_ & 1) == 1) { coin_ = !coin_; } else { coin_ = random_bit(); }
        let odds = if (self.state & 1) == 1 {
            // For odd state, flip the coin from previous state
            // We need to store coin state, but for now use deterministic pattern
            // that creates similar distribution to random
            (self.state >> 1) & 1 == 0 // Flip based on previous state
        } else {
            // For even state, C++ uses random_bit()
            // Use deterministic but pseudo-random pattern for reproducibility
            // This creates a good distribution while remaining deterministic
            ((self.state >> 1) ^ (self.state >> 3) ^ (self.state >> 7)) & 1 == 1
        };

        // Pre-allocate capacity to avoid reallocations
        let estimated_size = (items.len() + 1) / 2;
        if self.scratch_buffer.capacity() < estimated_size {
            self.scratch_buffer.reserve(estimated_size - self.scratch_buffer.capacity());
        }

        let mut i = if odds { 1 } else { 0 };
        while i < items.len() {
            self.scratch_buffer.push(items[i].clone());
            i += 2; // Skip every other item
        }

        &self.scratch_buffer
    }

    #[inline(always)]
    fn promote_evens_or_odds_simple(&self, items: &[T], _rank_accuracy: RankAccuracy) -> Vec<T> {
        if items.is_empty() {
            return Vec::new();
        }

        // Implement exact C++ coin flip logic
        let odds = if (self.state & 1) == 1 {
            (self.state >> 1) & 1 == 0
        } else {
            ((self.state >> 1) ^ (self.state >> 3) ^ (self.state >> 7)) & 1 == 1
        };

        let mut result = Vec::new();
        let mut i = if odds { 1 } else { 0 };

        while i < items.len() {
            result.push(items[i].clone());
            i += 2; // Skip every other item
        }

        result
    }

    /// Computes the weight contribution of this compactor for rank calculation.
    /// This matches the C++ compute_weight method exactly.
    ///
    /// # Arguments
    /// * `item` - The item to find the weight for
    /// * `inclusive` - Whether to include the item's weight in the calculation
    ///
    /// # Returns
    /// The weight contributed by this compactor (number of items * 2^lg_weight)
    pub fn compute_weight(&mut self, item: &T, inclusive: bool) -> u64 {
        // Ensure items are sorted for binary search (matches C++ behavior)
        if !self.is_sorted {
            self.sort();
        }

        // Perform binary search to find position (matches C++ std::upper_bound/std::lower_bound)
        let position = if inclusive {
            // inclusive: use upper_bound (first position where item < items[pos])
            // This finds the first position where the item would be placed after all equal items
            match self.items.binary_search_by(|probe| probe.partial_cmp(item).unwrap()) {
                Ok(pos) => {
                    // Found exact match, find the last occurrence
                    let mut end_pos = pos;
                    while end_pos + 1 < self.items.len() && self.items[end_pos + 1] == *item {
                        end_pos += 1;
                    }
                    end_pos + 1
                }
                Err(pos) => pos,
            }
        } else {
            // exclusive: use lower_bound (first position where !(items[pos] < item))
            // This finds the first position where the item would be placed before all equal items
            match self.items.binary_search_by(|probe| probe.partial_cmp(item).unwrap()) {
                Ok(pos) => {
                    // Found exact match, find the first occurrence
                    let mut start_pos = pos;
                    while start_pos > 0 && self.items[start_pos - 1] == *item {
                        start_pos -= 1;
                    }
                    start_pos
                }
                Err(pos) => pos,
            }
        };

        // Return distance (number of items) shifted by lg_weight
        // This matches C++: std::distance(begin(), it) << lg_weight_
        (position as u64) << self.lg_weight
    }
}

// Specialized implementations for Copy types (numeric fast-path)
impl<T> Compactor<T>
where
    T: Copy + PartialOrd + Clone,
{
    /// Fast compaction for Copy types - avoids cloning
    pub fn compact_into_fast(&mut self, _rank_accuracy: RankAccuracy, out: &mut Vec<T>) {
        if self.items.is_empty() {
            out.clear();
            return;
        }

        // Calculate sections to compact based on state (C++ logic)
        let secs_to_compact = ((!self.state).trailing_zeros() + 1).min(self.num_sections as u32) as u8;
        let compaction_range = self.compute_compaction_range(secs_to_compact);

        // Sort only the compaction range (avoid sorting the whole level)
        self.items[compaction_range.0..compaction_range.1]
            .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.is_sorted = false; // level as a whole might no longer be fully sorted

        // Must have at least 2 items to compact
        if compaction_range.1 <= compaction_range.0 || (compaction_range.1 - compaction_range.0) < 2 {
            out.clear();
            return;
        }

        // Ensure enough sections for growth
        self.ensure_enough_sections();

        // Even/odd choice (same logic you had)
        let odds = if (self.state & 1) == 1 {
            (self.state >> 1) & 1 == 0
        } else {
            ((self.state >> 1) ^ (self.state >> 3) ^ (self.state >> 7)) & 1 == 1
        };

        // Build promoted items directly into output buffer (NO CLONE for Copy types)
        out.clear();
        let (start, end) = compaction_range;
        let mut i = start + if odds { 1 } else { 0 };
        while i < end {
            out.push(self.items[i]); // Direct copy, no clone() call!
            i += 2;
        }

        // Remove the compacted range in-place: shift tail left and truncate (no allocation).
        let removed = end - start;
        if end < self.items.len() {
            // Use copy for Copy types - no need for unsafe
            self.items.copy_within(end.., start);
        }
        let new_len = self.items.len() - removed;
        self.items.truncate(new_len);

        // Update state
        self.state += 1;
    }

    /// Fast merge for Copy types - avoids cloning in merge operations
    pub fn merge_sorted_fast(&mut self, items: &[T]) {
        if items.is_empty() {
            return;
        }

        if self.items.is_empty() {
            self.items.extend_from_slice(items);
            self.is_sorted = true;
            return;
        }

        // Ensure sorted on both inputs by contract
        let total = self.items.len() + items.len();
        self.scratch_buffer.clear();
        if self.scratch_buffer.capacity() < total {
            self.scratch_buffer.reserve(total - self.scratch_buffer.capacity());
        }

        let (mut i, mut j) = (0usize, 0usize);
        let (a, b) = (&self.items, items);

        // Two-pointer merge into scratch buffer - NO CLONE for Copy types
        while i < a.len() && j < b.len() {
            if a[i].partial_cmp(&b[j]).unwrap_or(std::cmp::Ordering::Equal).is_le() {
                self.scratch_buffer.push(a[i]); // Direct copy
                i += 1;
            } else {
                self.scratch_buffer.push(b[j]); // Direct copy
                j += 1;
            }
        }

        // Add remaining elements - no clone needed
        if i < a.len() {
            self.scratch_buffer.extend_from_slice(&a[i..]);
        }
        if j < b.len() {
            self.scratch_buffer.extend_from_slice(&b[j..]);
        }

        // Swap scratch buffer with items (zero-copy)
        self.items.clear();
        std::mem::swap(&mut self.items, &mut self.scratch_buffer);
        self.is_sorted = true;
    }
}

/// Rounds a float to the nearest even integer, matching C++ implementation.

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