//! Iterator implementations for REQ sketch inspection.

use crate::compactor::Compactor;

/// Iterator over (item, weight) pairs in a REQ sketch.
///
/// Provides access to all items in the sketch along with their weights,
/// which depend on the level of the compactor they're stored in.
pub struct ReqSketchIterator<'a, T>
where
    T: PartialOrd + Clone,
{
    compactors: &'a [Compactor<T>],
    current_level: usize,
    current_item: usize,
}

impl<'a, T> ReqSketchIterator<'a, T>
where
    T: PartialOrd + Clone,
{
    /// Creates a new iterator over the compactors.
    pub(crate) fn new(compactors: &'a [Compactor<T>]) -> Self {
        Self {
            compactors,
            current_level: 0,
            current_item: 0,
        }
    }
}

impl<'a, T> Iterator for ReqSketchIterator<'a, T>
where
    T: PartialOrd + Clone,
{
    type Item = (T, u64);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_level < self.compactors.len() {
            let compactor = &self.compactors[self.current_level];
            let items: Vec<&T> = compactor.iter().collect();

            if self.current_item < items.len() {
                let item = items[self.current_item].clone();
                let weight = compactor.weight();
                self.current_item += 1;
                return Some((item, weight));
            }

            // Move to next level
            self.current_level += 1;
            self.current_item = 0;
        }

        None
    }
}

/// Iterator over items in a specific compactor level.
pub struct CompactorIterator<'a, T>
where
    T: PartialOrd + Clone,
{
    items: Box<dyn Iterator<Item = &'a T> + 'a>,
    weight: u64,
}

impl<'a, T> CompactorIterator<'a, T>
where
    T: PartialOrd + Clone,
{
    /// Creates a new iterator for a compactor.
    pub fn new(compactor: &'a Compactor<T>) -> Self {
        Self {
            items: Box::new(compactor.iter()),
            weight: compactor.weight(),
        }
    }
}

impl<'a, T> Iterator for CompactorIterator<'a, T>
where
    T: PartialOrd + Clone,
{
    type Item = (T, u64);

    fn next(&mut self) -> Option<Self::Item> {
        self.items.next().map(|item| (item.clone(), self.weight))
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