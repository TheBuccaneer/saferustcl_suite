#![cfg(feature = "memtrace")]

use std::time::Instant;
use super::{LOG, Record, Dir, Phase, T0, AUTO_TRACE, CURRENT_ABORT};

/// Token for tracking copy operations
pub struct CopyToken {
    start: Instant,
    bytes: usize,
    dir: Dir,
    finished: bool,
}

impl CopyToken {
    /// Finish and log the operation
    pub fn finish(mut self) {
        self.log_once();
    }

    fn log_once(&mut self) {
        if self.finished {
            return;
        }
        
        let s = self.start.duration_since(*T0).as_micros() as u64;
        let e = Instant::now().duration_since(*T0).as_micros() as u64;
        
        let mut log = LOG.lock().unwrap();
        let prev_end = log.last().map(|r| r.t_end_us).unwrap_or(0);
        let idle = if s > prev_end { s - prev_end } else { 0 };
        let abort = CURRENT_ABORT.lock().unwrap().clone();

        let phase = match self.dir {
            Dir::Kernel => Phase::Kernel,
            _ => Phase::Transfer,
        };

        log.push(Record {
            t_start_us: s,
            t_end_us: e,
            bytes: self.bytes,
            dir: self.dir,
            idle_us: idle,
            abort_token: abort,
            phase,
            tx_id: None,
            cause: None,
            retries: None,
            conflict_sz: None,
        });

        self.finished = true;
    }
}

impl Drop for CopyToken {
    fn drop(&mut self) {
        if AUTO_TRACE.load(std::sync::atomic::Ordering::Relaxed) {
            self.log_once();
        }
    }
}

/// Start tracking a transfer
pub fn start(dir: Dir, bytes: usize) -> CopyToken {
    CopyToken {
        start: Instant::now(),
        bytes,
        dir,
        finished: false,
    }
}

/// Log a transfer with explicit timing
pub fn log_transfer(t_start_us: u64, t_end_us: u64, bytes: usize, dir: Dir) {
    if !AUTO_TRACE.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    
    let abort = CURRENT_ABORT.lock().unwrap().clone();
    let mut log = LOG.lock().unwrap();
    let prev_end = log.last().map(|r| r.t_end_us).unwrap_or(0);
    let idle = if t_start_us > prev_end { t_start_us - prev_end } else { 0 };
    
    log.push(Record {
        t_start_us,
        t_end_us,
        bytes,
        dir,
        idle_us: idle,
        abort_token: abort,
        phase: if matches!(dir, Dir::Kernel) { Phase::Kernel } else { Phase::Transfer },
        tx_id: None,
        cause: None,
        retries: None,
        conflict_sz: None,
    });
}