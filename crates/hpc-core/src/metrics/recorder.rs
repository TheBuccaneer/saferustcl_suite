#![cfg(feature = "metrics")]

use once_cell::sync::Lazy;
use std::{
    sync::Mutex,
    time::Instant,
};

/// Global timing records
pub static TIMES: Lazy<Mutex<Vec<(&'static str, u128)>>> =
    Lazy::new(|| Mutex::new(Vec::with_capacity(1024)));

/// Record timing for an operation
#[inline]
pub fn record(name: &'static str, start: Instant) {
    let dur = start.elapsed().as_micros();
    TIMES.lock().unwrap().push((name, dur));
}

/// Record timing with explicit duration
#[inline]
pub fn record_duration(name: &'static str, duration_us: u128) {
    TIMES.lock().unwrap().push((name, duration_us));
}