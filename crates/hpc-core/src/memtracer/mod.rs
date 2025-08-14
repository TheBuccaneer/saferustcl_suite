#![cfg(feature = "memtrace")]

//! Memory transfer tracing utilities

mod copytoken;
mod aborttoken;

pub use copytoken::{CopyToken, start, log_transfer};
pub use aborttoken::{
    AbortEvent, log_abort, 
    set_abort_token, clear_abort_token, AbortTokenGuard,
    CURRENT_ABORT,
};

use once_cell::sync::Lazy;
use std::{
    collections::HashMap,
    fs::File,
    io::Write,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    time::Instant,
};

/// Transfer direction
#[derive(Clone, Copy, Debug)]
pub enum Dir {
    H2D,
    D2H,
    Kernel,
}

/// Operation type (semantically clearer)
#[derive(Clone, Copy, Debug)]
pub enum Operation {
    H2D,
    D2H,
    Kernel,
}

impl Operation {
    pub fn as_str(self) -> &'static str {
        match self {
            Operation::H2D => "H2D",
            Operation::D2H => "D2H",
            Operation::Kernel => "KRN",
        }
    }
}

impl Dir {
    pub fn as_str(self) -> &'static str {
        match self {
            Dir::H2D => "H2D",
            Dir::D2H => "D2H",
            Dir::Kernel => "KRN",
        }
    }
}

/// Phase of operation
#[derive(Clone, Copy, Debug)]
pub enum Phase {
    Transfer,
    Kernel,
    Abort,
}

impl Phase {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Transfer => "Transfer",
            Phase::Kernel => "Kernel",
            Phase::Abort => "Abort",
        }
    }
}

/// Global start time reference
pub static T0: Lazy<Instant> = Lazy::new(Instant::now);

/// Auto-trace enable flag
pub static AUTO_TRACE: AtomicBool = AtomicBool::new(true);

/// Enable auto-tracing
#[inline]
pub fn enable_auto_trace() {
    AUTO_TRACE.store(true, Ordering::Relaxed);
}

/// Disable auto-tracing
#[inline]
pub fn disable_auto_trace() {
    AUTO_TRACE.store(false, Ordering::Relaxed);
}

/// Check if auto-tracing is enabled
#[inline]
pub fn is_auto_trace_enabled() -> bool {
    AUTO_TRACE.load(Ordering::Relaxed)
}

/// Log record
#[derive(Debug)]
pub struct Record {
    pub t_start_us: u64,
    pub t_end_us: u64,
    pub bytes: usize,
    pub dir: Dir,
    pub idle_us: u64,
    pub abort_token: Option<String>,
    pub phase: Phase,
    pub tx_id: Option<u64>,
    pub cause: Option<String>,
    pub retries: Option<u32>,
    pub conflict_sz: Option<usize>,
}

/// Global log storage
pub static LOG: Lazy<Mutex<Vec<Record>>> =
    Lazy::new(|| Mutex::new(Vec::with_capacity(4096)));

#[cfg(feature = "memtrace")]
pub fn flush_csv() {
    let log = LOG.lock().unwrap();

    // A) Transfer/Kernel Events → memtrace.csv
    let mut f = File::create("memtrace.csv").expect("memtrace.csv nicht anlegbar");
    writeln!(f, "t_start_us,t_end_us,bytes,dir,idle_us,abort_token,phase").unwrap();
    for r in log.iter().filter(|r| !matches!(r.phase, Phase::Abort)) {
        let dir = match r.dir { Dir::H2D => "H2D", Dir::D2H => "D2H", Dir::Kernel => "Kernel" };
        let phase = match r.phase { Phase::Kernel => "Kernel", Phase::Transfer => "Transfer", Phase::Abort => "Abort" };
        writeln!(
            f,
            "{},{},{},{},{},{},{}",
            r.t_start_us,
            r.t_end_us,
            r.bytes,
            dir,
            r.idle_us,
            r.abort_token.as_deref().unwrap_or(""),
            phase
        ).unwrap();
    }

    // B) Abort-Events (aggregiert) → memtrace_abort.csv
    #[derive(Default, Clone)]
    struct Agg {
        count: u64,
        retries_sum: u64,
        conflict_sum: u64,
        conflict_min: usize,
        conflict_max: usize,
        first_us: u64,
        last_us: u64,
    }

    let mut agg: HashMap<(String, String), Agg> = HashMap::new();
    for r in log.iter().filter(|r| matches!(r.phase, Phase::Abort)) {
        let token = r.abort_token.as_deref().unwrap_or("").to_string();
        let cause = r.cause.as_deref().unwrap_or("").to_string();
        let entry = agg.entry((token, cause)).or_insert_with(|| Agg {
            conflict_min: usize::MAX,
            ..Default::default()
        });
        entry.count += 1;
        entry.retries_sum += r.retries.unwrap_or(0) as u64;
        let c = r.conflict_sz.unwrap_or(0);
        entry.conflict_sum += c as u64;
        if c < entry.conflict_min { entry.conflict_min = c; }
        if c > entry.conflict_max { entry.conflict_max = c; }
        if entry.first_us == 0 || r.t_start_us < entry.first_us { entry.first_us = r.t_start_us; }
        if r.t_end_us > entry.last_us { entry.last_us = r.t_end_us; }
    }

    let mut fa = File::create("memtrace_abort.csv").expect("memtrace_abort.csv nicht anlegbar");
    writeln!(fa, "abort_token,cause,count,retries_avg,conflict_avg,conflict_min,conflict_max,first_us,last_us").unwrap();
    for ((token, cause), a) in agg.iter() {
        let r_avg = if a.count > 0 { a.retries_sum as f64 / a.count as f64 } else { 0.0 };
        let c_avg = if a.count > 0 { a.conflict_sum as f64 / a.count as f64 } else { 0.0 };
        let c_min = if a.conflict_min == usize::MAX { 0 } else { a.conflict_min };
        writeln!(
            fa,
            "{},{},{},{:.3},{:.3},{},{},{},{}",
            token, cause, a.count, r_avg, c_avg, c_min, a.conflict_max, a.first_us, a.last_us
        ).unwrap();
    }

    // Optional: Voll-Log der Aborts → memtrace_abort_full.csv (nur wenn Feature aktiv)
    #[cfg(feature = "memtrace_full")]
    {
        let mut ff = File::create("memtrace_abort_full.csv").expect("memtrace_abort_full.csv nicht anlegbar");
        writeln!(ff, "tx_id,cause,retries,conflict_sz,t_start_us,t_end_us,abort_token").unwrap();
        for r in log.iter().filter(|r| matches!(r.phase, Phase::Abort)) {
            writeln!(
                ff,
                "{},{},{},{},{},{},{}",
                r.tx_id.unwrap_or(0),
                r.cause.as_deref().unwrap_or(""),
                r.retries.unwrap_or(0),
                r.conflict_sz.unwrap_or(0),
                r.t_start_us,
                r.t_end_us,
                r.abort_token.as_deref().unwrap_or("")
            ).unwrap();
        }
    }

    // C) Summary → memtrace_summary.txt
    let total_events = log.len();
    let total_idle: u64 = log.iter().map(|r| r.idle_us).sum();
    let bytes_h2d: u64 = log.iter()
        .filter(|r| matches!(r.dir, Dir::H2D))
        .map(|r| r.bytes as u64).sum();
    let bytes_d2h: u64 = log.iter()
        .filter(|r| matches!(r.dir, Dir::D2H))
        .map(|r| r.bytes as u64).sum();
    let aborts = log.iter().filter(|r| matches!(r.phase, Phase::Abort)).count();

    let mut fs = File::create("memtrace_summary.txt").expect("memtrace_summary.txt nicht anlegbar");
    writeln!(fs, "events_total: {}", total_events).unwrap();
    writeln!(fs, "idle_total_us: {}", total_idle).unwrap();
    writeln!(fs, "bytes_h2d: {}", bytes_h2d).unwrap();
    writeln!(fs, "bytes_d2h: {}", bytes_d2h).unwrap();
    writeln!(fs, "aborts: {}", aborts).unwrap();
}

/// Reset all logs
pub fn reset() {
    LOG.lock().unwrap().clear();
}

/// RAII scope for temporarily changing trace state
#[derive(Debug)]
pub struct TracingScope {
    prev: bool,
}

impl TracingScope {
    #[inline]
    pub fn new(enable: bool) -> Self {
        let prev = AUTO_TRACE.swap(enable, Ordering::Relaxed);
        TracingScope { prev }
    }

    #[inline]
    pub fn enabled() -> Self {
        Self::new(true)
    }

    #[inline]
    pub fn disabled() -> Self {
        Self::new(false)
    }
}

impl Drop for TracingScope {
    fn drop(&mut self) {
        AUTO_TRACE.store(self.prev, Ordering::Relaxed);
    }
}

/// Get current time in microseconds since T0
#[inline]
pub fn now_us() -> u64 {
    Instant::now().duration_since(*T0).as_micros() as u64
}

#[cfg(feature = "memtrace")]
pub fn trace_abort(tx_id: u64, cause: &str, retries: u32, conflict_sz: u32, abort_token: &str) {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let t_us = now.as_micros() as u64;
    LOG.lock().unwrap().push(Record {
        t_start_us: t_us,
        t_end_us:   t_us,
        bytes: 0,
        dir: Dir::Kernel,
        idle_us: 0,
        abort_token: Some(abort_token.to_string()),
        phase: Phase::Abort,
        tx_id: Some(tx_id),
        cause: Some(cause.to_string()),
        retries: Some(retries),
        conflict_sz: Some(conflict_sz as usize),
    });
}
