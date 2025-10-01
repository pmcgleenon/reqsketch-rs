//! Error types for the REQ sketch library.

use std::fmt;

/// Result type alias for REQ sketch operations.
pub type Result<T> = std::result::Result<T, ReqError>;

/// Error types that can occur during REQ sketch operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ReqError {
    /// The sketch is empty and cannot perform the requested operation.
    EmptySketch,

    /// Invalid parameter k - must be even and >= 4.
    InvalidK(u16),

    /// Invalid rank - must be in range [0.0, 1.0].
    InvalidRank(f64),

    /// Incompatible sketches for merge operation.
    IncompatibleSketches(String),

    /// Internal cache is invalid and needs refresh.
    CacheInvalid,

    /// Split points are not properly sorted or contain invalid values.
    InvalidSplitPoints(String),

    /// Operation not supported for this type.
    UnsupportedOperation(String),

    /// Serialization/deserialization error.
    #[cfg(feature = "serde")]
    SerializationError(String),
}

impl fmt::Display for ReqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReqError::EmptySketch => write!(f, "Cannot perform operation on empty sketch"),
            ReqError::InvalidK(k) => write!(f, "Invalid k parameter: {}. Must be even and >= 4", k),
            ReqError::InvalidRank(rank) => write!(f, "Invalid rank: {}. Must be in range [0.0, 1.0]", rank),
            ReqError::IncompatibleSketches(msg) => write!(f, "Incompatible sketches: {}", msg),
            ReqError::CacheInvalid => write!(f, "Internal cache is invalid"),
            ReqError::InvalidSplitPoints(msg) => write!(f, "Invalid split points: {}", msg),
            ReqError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            #[cfg(feature = "serde")]
            ReqError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for ReqError {}