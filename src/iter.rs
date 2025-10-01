//! Iterator implementations for REQ sketch inspection.

use crate::compactor::Compactor;
use crate::TotalOrd;

/// Iterator over (item, weight) pairs in a REQ sketch.
///
/// Provides access to all items in the sketch along with their weights,
/// which depend on the level of the compactor they're stored in.
///
/// Zero-allocation implementation that works directly with slices.
pub struct ReqSketchIterator<'a, T>
where
    T: Clone + TotalOrd + PartialEq,
{
    compactors: &'a [Compactor<T>],
    current_level: usize,
    current_level_iter: Option<std::slice::Iter<'a, T>>,
    current_weight: u64,
}

impl<'a, T> ReqSketchIterator<'a, T>
where
    T: Clone + TotalOrd + PartialEq,
{
    /// Creates a new iterator over the compactors.
    pub(crate) fn new(compactors: &'a [Compactor<T>]) -> Self {
        let mut iter = Self {
            compactors,
            current_level: 0,
            current_level_iter: None,
            current_weight: 0,
        };
        iter.advance_to_next_level();
        iter
    }

    fn advance_to_next_level(&mut self) {
        while self.current_level < self.compactors.len() {
            let compactor = &self.compactors[self.current_level];
            // Access items slice directly without allocation
            let items_slice = compactor.items_slice();

            if !items_slice.is_empty() {
                self.current_level_iter = Some(items_slice.iter());
                self.current_weight = compactor.weight();
                return;
            }

            self.current_level += 1;
        }

        self.current_level_iter = None;
    }
}

impl<'a, T> Iterator for ReqSketchIterator<'a, T>
where
    T: Clone + TotalOrd + PartialEq,
{
    type Item = (T, u64);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut level_iter) = self.current_level_iter {
                if let Some(item) = level_iter.next() {
                    return Some((item.clone(), self.current_weight));
                }
            }

            // Current level exhausted, move to next
            self.current_level += 1;
            self.advance_to_next_level();

            if self.current_level_iter.is_none() {
                return None;
            }
        }
    }
}

/// Iterator over items in a specific compactor level.
/// Zero-allocation implementation using direct slice iteration.
pub struct CompactorIterator<'a, T>
where
    T: Clone + TotalOrd + PartialEq,
{
    items_iter: std::slice::Iter<'a, T>,
    weight: u64,
}

impl<'a, T> CompactorIterator<'a, T>
where
    T: Clone + TotalOrd + PartialEq,
{
    /// Creates a new iterator for a compactor.
    pub fn new(compactor: &'a Compactor<T>) -> Self {
        Self {
            items_iter: compactor.items_slice().iter(),
            weight: compactor.weight(),
        }
    }
}

impl<'a, T> Iterator for CompactorIterator<'a, T>
where
    T: Clone + TotalOrd + PartialEq,
{
    type Item = (T, u64);

    fn next(&mut self) -> Option<Self::Item> {
        self.items_iter.next().map(|item| (item.clone(), self.weight))
    }
}

#[cfg(test)]
mod tests {
    use crate::ReqSketch;

    #[test]
    fn test_sketch_iterator() {
        let mut sketch = ReqSketch::new();

        // Add some values
        for i in 0..10 {
            sketch.update(i);
        }

        let items: Vec<(i32, u64)> = sketch.iter().collect();

        // All items should be at level 0 (weight 1) since no compaction occurred
        assert_eq!(items.len(), 10);
        for (_, weight) in &items {
            assert_eq!(*weight, 1);
        }

        // Items should be present (order may vary since level 0 might not be sorted)
        let values: Vec<i32> = items.into_iter().map(|(item, _)| item).collect();
        for i in 0..10 {
            assert!(values.contains(&i));
        }
    }

    #[test]
    fn test_empty_iterator() {
        let sketch: ReqSketch<i32> = ReqSketch::new();
        let items: Vec<(i32, u64)> = sketch.iter().collect();
        assert!(items.is_empty());
    }
}