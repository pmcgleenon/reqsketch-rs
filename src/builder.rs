//! Builder pattern implementation for REQ sketch construction.

use crate::{RankAccuracy, ReqError, ReqSketch, Result, TotalOrd};
use std::marker::PhantomData;

/// Builder for constructing REQ sketches with custom parameters.
///
/// Provides a fluent interface for configuring sketch parameters with validation
/// and sensible defaults.
///
/// # Examples
///
/// ```rust
/// use reqsketch::{ReqSketch, RankAccuracy};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Use defaults
/// let sketch: ReqSketch<f64> = ReqSketch::builder().build()?;
///
/// // Custom configuration
/// let sketch: ReqSketch<f64> = ReqSketch::builder()
///     .k(16)?
///     .rank_accuracy(RankAccuracy::LowRank)
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ReqSketchBuilder<T> {
    k: u16,
    rank_accuracy: RankAccuracy,
    _phantom: PhantomData<T>,
}

impl<T> ReqSketchBuilder<T>
where
    T: Clone + TotalOrd + PartialEq,
{
    /// Creates a new builder with default parameters.
    ///
    /// Defaults:
    /// - k = 12 (roughly 1% relative error at 95% confidence)
    /// - rank_accuracy = HighRank
    pub fn new() -> Self {
        Self {
            k: 12,
            rank_accuracy: RankAccuracy::HighRank,
            _phantom: PhantomData,
        }
    }

    /// Sets the k parameter.
    ///
    /// # Arguments
    /// * `k` - Controls size and error of the sketch. Must be even and >= 4.
    ///
    /// # Errors
    /// Returns `ReqError::InvalidK` if k is not even or < 4.
    pub fn k(mut self, k: u16) -> Result<Self> {
        validate_k(k)?;
        self.k = k;
        Ok(self)
    }

    /// Sets the rank accuracy mode.
    ///
    /// # Arguments
    /// * `accuracy` - Whether to optimize for high ranks or low ranks
    pub fn rank_accuracy(mut self, accuracy: RankAccuracy) -> Self {
        self.rank_accuracy = accuracy;
        self
    }


    /// Builds the REQ sketch with the configured parameters.
    pub fn build(self) -> Result<ReqSketch<T>> {
        Ok(ReqSketch {
            k: self.k,
            rank_accuracy: self.rank_accuracy,
            total_n: 0,
            max_nom_size: 0,
            num_retained: 0,
            compactors: Vec::new(),
            promotion_buf: Vec::with_capacity(self.k as usize), // Pre-allocate based on k
            min_item: None,
            max_item: None,
            sorted_view_cache: None,
        })
    }

    /// Returns the currently configured k value.
    pub fn get_k(&self) -> u16 {
        self.k
    }

    /// Returns the currently configured rank accuracy mode.
    pub fn get_rank_accuracy(&self) -> RankAccuracy {
        self.rank_accuracy
    }

}

impl<T> Default for ReqSketchBuilder<T>
where
    T: Clone + TotalOrd + PartialEq,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Validates the k parameter.
pub(crate) fn validate_k(k: u16) -> Result<()> {
    if k < 4 || k % 2 != 0 {
        Err(ReqError::InvalidK(k))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_builder() {
        let builder: ReqSketchBuilder<f64> = ReqSketchBuilder::new();
        assert_eq!(builder.get_k(), 12);
        assert_eq!(builder.get_rank_accuracy(), RankAccuracy::HighRank);
    }

    #[test]
    fn test_builder_k_validation() {
        // Valid k values
        assert!(ReqSketchBuilder::<f64>::new().k(4).is_ok());
        assert!(ReqSketchBuilder::<f64>::new().k(12).is_ok());
        assert!(ReqSketchBuilder::<f64>::new().k(1024).is_ok());
        assert!(ReqSketchBuilder::<f64>::new().k(2048).is_ok()); // large values now allowed
        assert!(ReqSketchBuilder::<f64>::new().k(u16::MAX - 1).is_ok()); // max even u16

        // Invalid k values
        assert!(ReqSketchBuilder::<f64>::new().k(3).is_err()); // too small
        assert!(ReqSketchBuilder::<f64>::new().k(5).is_err()); // odd
        assert!(ReqSketchBuilder::<f64>::new().k(u16::MAX).is_err()); // max u16 is odd
    }

    #[test]
    fn test_builder_fluent_interface() {
        let sketch = ReqSketchBuilder::<i32>::new()
            .k(16)
            .expect("Valid k value")
            .rank_accuracy(RankAccuracy::LowRank)
            .build()
            .expect("Builder should succeed");

        assert_eq!(sketch.k(), 16);
        assert_eq!(sketch.rank_accuracy(), RankAccuracy::LowRank);
    }

    #[test]
    fn test_validate_k() {
        assert!(validate_k(4).is_ok());
        assert!(validate_k(12).is_ok());
        assert!(validate_k(1024).is_ok());
        assert!(validate_k(2048).is_ok()); // large values now allowed
        assert!(validate_k(u16::MAX - 1).is_ok()); // max even u16

        assert!(validate_k(3).is_err());
        assert!(validate_k(5).is_err());
        assert!(validate_k(u16::MAX).is_err()); // max u16 is odd
    }
}