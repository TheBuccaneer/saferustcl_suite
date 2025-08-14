//! HPC-Core: High-Performance Computing utilities for OpenCL
//! 
//! This crate provides safe wrappers and utilities for GPU computing.

// Core modules (always available)
mod error;
mod buffer;

// Re-export core types
pub use error::{ClError, Result};
pub use buffer::{GpuBuffer, GpuEventGuard};
pub use buffer::state::{State, Queued, InFlight, Ready};

// Feature-gated modules
#[cfg(feature = "metrics")]
pub mod metrics;
#[cfg(feature = "metrics")]
pub use metrics::{record, summary, ALLOCS, ALLOC_BYTES};

#[cfg(feature = "memtrace")]
pub mod memtracer;
#[cfg(feature = "memtrace")]
pub use memtracer::{
    start, flush_csv, reset,
    Dir, Operation, CopyToken, TracingScope,
    is_auto_trace_enabled, enable_auto_trace, disable_auto_trace,
    AbortEvent, AbortTokenGuard, set_abort_token, clear_abort_token,
    log_abort, log_transfer, now_us,
};

// FFI callback for memtrace
#[cfg(feature = "memtrace")]
use std::ffi::c_void;

#[cfg(feature = "memtrace")]
pub extern "C" fn memtrace_callback(
    _evt: opencl3::types::cl_event,
    _status: opencl3::types::cl_int,
    user_data: *mut c_void,
) {
    // SAFETY: Pointer was obtained via Box::into_raw, so it is non-null and uniquely owned.
    let tok: Box<CopyToken> = unsafe { Box::from_raw(user_data.cast()) };
    tok.finish();
}