use thiserror::Error;

/// Main error type for HPC-Core operations
#[derive(Error, Debug)]
pub enum ClError {
    #[error("OpenCL error code {0}")]
    Api(i32),
    
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },
    
    #[error("Invalid state transition")]
    InvalidState,
    
    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),
}

/// Result type alias for HPC-Core operations
pub type Result<T> = std::result::Result<T, ClError>;

/// Macro for checking OpenCL error codes
#[macro_export]
macro_rules! cl_try {
    ($expr:expr) => {
        let err = unsafe { $expr };
        if err != 0 {
            return Err($crate::error::ClError::Api(err));
        }
    };
}

// Re-export for backwards compatibility
#[allow(unused_imports)]
pub(crate) use cl_try;

impl From<opencl3::error_codes::ClError> for ClError {
    fn from(err: opencl3::error_codes::ClError) -> Self {
        ClError::Api(err.0)
    }
}

impl From<i32> for ClError {
    fn from(code: i32) -> Self {
        ClError::Api(code)
    }
}