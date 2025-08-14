//! Type-state pattern for compile-time state checking

/// Sealed trait pattern for state types
mod sealed {
    pub trait Sealed {}
}

/// State trait for GPU buffer states
pub trait State: sealed::Sealed + std::fmt::Debug + Send + Sync {}

/// Buffer is queued and ready for operations
#[derive(Debug, Clone, Copy)]
pub struct Queued;
impl sealed::Sealed for Queued {}
impl State for Queued {}

/// Buffer operation is in flight
#[derive(Debug, Clone, Copy)]
pub struct InFlight;
impl sealed::Sealed for InFlight {}
impl State for InFlight {}

/// Buffer is ready for use
#[derive(Debug, Clone, Copy)]
pub struct Ready;
impl sealed::Sealed for Ready {}
impl State for Ready {}