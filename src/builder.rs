//! Builder pattern implementation for REQ sketch construction.

use crate::{RankAccuracy, RankMethod, ReqError, ReqSketch, Result};
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
/// // Use defaults
/// let sketch: ReqSketch<f64> = ReqSketch::builder().build().unwrap();
///
/// // Custom configuration
/// let sketch: ReqSketch<f64> = ReqSketch::builder()
///     .k(16).unwrap()
///     .rank_accuracy(RankAccuracy::LowRank)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ReqSketchBuilder<T> {
    k: u16,
    rank_accuracy: RankAccuracy,
    rank_method: RankMethod,
    _phantom: PhantomData<T>,
}

impl<T> ReqSketchBuilder<T>
where
    T: PartialOrd + Clone,
{
    /// Creates a new builder with default parameters.
    ///
    /// Defaults:
    /// - k = 12 (roughly 1% relative error at 95% confidence)
    /// - rank_accuracy = HighRank
    /// - rank_method = StepFunction (C++ compatible)
    pub fn new() -> Self {
        Self {
            k: 12,
            rank_accuracy: RankAccuracy::HighRank,
            rank_method: RankMethod::StepFunction,
            _phantom: PhantomData,
        }
    }

    /// Sets the k parameter.
    ///
    /// # Arguments
    /// * `k` - Controls size and error of the sketch. Must be even and in range [4, 1024].
    ///
    /// # Errors
    /// Returns `ReqError::InvalidK` if k is not even or outside valid range.
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

    /// Sets the rank calculation method.
    ///
    /// # Arguments
    /// * `method` - The rank calculation method to use
    ///   - `StepFunction`: C++ compatible step function (default)
    ///   - `Interpolation`: Enhanced precision with linear interpolation
    pub fn rank_method(mut self, method: RankMethod) -> Self {
        self.rank_method = method;
        self
    }

    /// Builds the REQ sketch with the configured parameters.
    pub fn build(self) -> Result<ReqSketch<T>> {
        Ok(ReqSketch {
            k: self.k,
            rank_accuracy: self.rank_accuracy,
            rank_method: self.rank_method,
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

    /// Returns the currently configured rank method.
    pub fn get_rank_method(&self) -> RankMethod {
        self.rank_method
    }
}

impl<T> Default for ReqSketchBuilder<T>
where
    T: PartialOrd + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Validates the k parameter.
pub(crate) fn validate_k(k: u16) -> Result<()> {
    if k < 4 || k > 1024 || k % 2 != 0 {
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

        // Invalid k values
        assert!(ReqSketchBuilder::<f64>::new().k(3).is_err()); // too small
        assert!(ReqSketchBuilder::<f64>::new().k(5).is_err()); // odd
        assert!(ReqSketchBuilder::<f64>::new().k(1025).is_err()); // too large
    }

    #[test]
    fn test_builder_fluent_interface() {
        let sketch = ReqSketchBuilder::<i32>::new()
            .k(16)
            .unwrap()
            .rank_accuracy(RankAccuracy::LowRank)
            .build()
            .unwrap();

        assert_eq!(sketch.k(), 16);
        assert_eq!(sketch.rank_accuracy(), RankAccuracy::LowRank);
    }

    #[test]
    fn test_validate_k() {
        assert!(validate_k(4).is_ok());
        assert!(validate_k(12).is_ok());
        assert!(validate_k(1024).is_ok());

        assert!(validate_k(3).is_err());
        assert!(validate_k(5).is_err());
        assert!(validate_k(1025).is_err());
    }
}