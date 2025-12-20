//! Compactor implementation for REQ sketch levels.
//!
//! Each level in the REQ sketch uses a compactor to maintain a bounded set of items
//! with deterministic compaction when capacity is exceeded.

use crate::{RankAccuracy, Result, TotalOrd};

fn nearest_even(value: f32) -> u32 {
    ((value / 2.0).round() as u32) << 1
}

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A compactor maintains items at a specific level of the REQ sketch.
///
/// When the compactor reaches its nominal capacity, it performs compaction
/// by keeping approximately half the items and promoting the rest to the next level.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: Clone + TotalOrd + PartialEq + serde::Serialize + serde::de::DeserializeOwned"))]
pub struct Compactor<T> {
    /// Current items in the compactor
    items: Vec<T>,
    /// Whether items are currently sorted
    is_sorted: bool,
    /// State for deterministic compaction
    state: u64,
    /// Reusable scratch buffer for compaction operations
    #[cfg_attr(feature = "serde", serde(skip))]
    scratch_buffer: Vec<T>,

    /// Actual section size (rounded to integer)
    section_size: u32,
    /// Number of sections in this compactor
    num_sections: u8,
    /// The level of this compactor (0 = base level)
    lg_weight: u8,

    /// Whether this compactor is configured for high rank accuracy
    rank_accuracy: RankAccuracy,
    /// Raw section size (may be fractional)
    section_size_raw: f32,
    /// Random bit for compaction 
    coin: bool,
}

impl<T> Compactor<T>
where
    T: Clone + TotalOrd + PartialEq,
{
    /// Creates a new compactor for the given level.
    ///
    /// # Arguments
    /// * `lg_weight` - The level (log weight) of this compactor
    /// * `k` - The k parameter from the parent sketch
    /// * `rank_accuracy` - Rank accuracy configuration
    pub fn new(lg_weight: u8, k: u16, rank_accuracy: RankAccuracy) -> Self {
        let section_size_raw = k as f32;
        let section_size = nearest_even(section_size_raw);
        let num_sections = 3u8;

        let nominal: usize = (2 * section_size * num_sections as u32) as usize;

        Self {
            items: Vec::with_capacity(nominal),
            is_sorted: true,
            state: 0,
            scratch_buffer: Vec::with_capacity(nominal / 2 + 8),

            section_size,
            num_sections,
            lg_weight,

            rank_accuracy,
            section_size_raw,
            coin: rand::random::<bool>(),
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
            if a[i].total_cmp(&b[j]).is_le() {
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
            self.items.sort_unstable_by(|a, b| a.total_cmp(b));
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

        // Calculate sections to compact based on state 
        let secs_to_compact = ((!self.state).trailing_zeros() + 1).min(self.num_sections as u32) as u8;
        let compaction_range = self.compute_compaction_range(secs_to_compact);

        // Sort only the compaction range (avoid sorting the whole level)
        self.items[compaction_range.0..compaction_range.1]
            .sort_unstable_by(|a, b| a.total_cmp(b));
        self.is_sorted = false; // level as a whole might no longer be fully sorted

        // Must have at least 2 items to compact
        if compaction_range.1 <= compaction_range.0 || (compaction_range.1 - compaction_range.0) < 2 {
            out.clear();
            return;
        }

        // Ensure enough sections for growth 
        self.ensure_enough_sections();

        if (self.state & 1) == 1 {
            self.coin = !self.coin; // flip coin for odd states
        } else {
            self.coin = rand::random::<bool>(); // random coin flip for even states
        }
        let odds = self.coin;

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
        let ssr = self.section_size_raw / (2.0_f32).sqrt();
        let ne = nearest_even(ssr);

        const MIN_K: u32 = 4; // matches datasketches-cpp

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
        let nom_capacity = self.nominal_capacity() as usize;
        let mut non_compact = nom_capacity / 2 + (self.num_sections - secs_to_compact) as usize * self.section_size as usize;

        // if (((num_items_ - non_compact) & 1) == 1) ++non_compact;
        if ((self.items.len() - non_compact) & 1) == 1 {
            non_compact += 1;
        }

        let (low, high) = match self.rank_accuracy {
            RankAccuracy::HighRank => {
                // HRA: Protect high ranks by compacting LOW sections (low values)
                // This means we compact from [0, num_items - non_compact] (bottom end)
                let high = if self.items.len() >= non_compact {
                    self.items.len() - non_compact
                } else {
                    0
                };
                (0, high)
            }
            RankAccuracy::LowRank => {
                // LRA: Protect low ranks by compacting HIGH sections (high values)
                // This means we compact from [non_compact, num_items] (top end)
                let low = non_compact.min(self.items.len());
                (low, self.items.len())
            }
        };

        // Empty window safety: ensure we have at least 2 items to compact
        if high <= low || (high - low) < 2 {
            return (0, 0); // Signal no compaction needed
        }

        // Ensure minimum section size for meaningful compaction
        if self.items.len() < 2 * self.section_size as usize {
            return (0, 0); // Skip compaction if level too small
        }

        (low, high)
    }


    /// Computes the weight contribution of this compactor for rank calculation.
    ///
    /// # Arguments
    /// * `item` - The item to find the weight for
    /// * `inclusive` - Whether to include the item's weight in the calculation
    ///
    /// # Returns
    /// The weight contributed by this compactor (number of items * 2^lg_weight)
    pub fn compute_weight(&mut self, item: &T, inclusive: bool) -> u64 {
        // Ensure items are sorted for binary search 
        if !self.is_sorted {
            self.sort();
        }

        // Perform binary search to find position
        let position = if inclusive {
            // inclusive: use upper_bound (first position where item < items[pos])
            // This finds the first position where the item would be placed after all equal items
            match self.items.binary_search_by(|probe| {
                probe.total_cmp(item)
            }) {
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
            match self.items.binary_search_by(|probe| {
                probe.total_cmp(item)
            }) {
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
        (position as u64) << self.lg_weight
    }
}

// Specialized implementations for Copy types (numeric fast-path)
impl<T> Compactor<T>
where
    T: Copy + Clone + TotalOrd + PartialEq,
{
    /// Fast compaction for Copy types - avoids cloning
    pub fn compact_into_fast(&mut self, _rank_accuracy: RankAccuracy, out: &mut Vec<T>) {
        if self.items.is_empty() {
            out.clear();
            return;
        }

        // Calculate sections to compact based on state 
        let secs_to_compact = ((!self.state).trailing_zeros() + 1).min(self.num_sections as u32) as u8;
        let compaction_range = self.compute_compaction_range(secs_to_compact);

        // Sort only the compaction range (avoid sorting the whole level)
        self.items[compaction_range.0..compaction_range.1]
            .sort_unstable_by(|a, b| a.total_cmp(b));
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
            if a[i].total_cmp(&b[j]).is_le() {
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
        assert_eq!(nearest_even(0.0), 0); // 0/2=0, round(0)=0, 0<<1=0
        assert_eq!(nearest_even(1.0), 2); // 1/2=0.5, round(0.5)=1, 1<<1=2
        assert_eq!(nearest_even(2.0), 2); // 2/2=1, round(1)=1, 1<<1=2
        assert_eq!(nearest_even(3.0), 4); // 3/2=1.5, round(1.5)=2, 2<<1=4
        assert_eq!(nearest_even(4.0), 4); // 4/2=2, round(2)=2, 2<<1=4
        assert_eq!(nearest_even(4.6), 4); // 4.6/2=2.3, round(2.3)=2, 2<<1=4
        assert_eq!(nearest_even(5.6), 6); // 5.6/2=2.8, round(2.8)=3, 3<<1=6
        assert_eq!(nearest_even(13.0), 14); // 13/2=6.5, round(6.5)=7, 7<<1=14
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
