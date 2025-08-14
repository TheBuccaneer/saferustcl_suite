#![cfg(feature = "metrics")]

use once_cell::sync::Lazy;
use std::{
    collections::HashMap,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

// Roh‑Latenzen

static TIMES: Lazy<Mutex<Vec<(&'static str, u128)>>> =
    Lazy::new(|| Mutex::new(Vec::new()));

/// Im Wrapper aufrufen: `record("enqueue_write", Instant::now());`
pub fn record(name: &'static str, start: Instant) {
    let dur = start.elapsed().as_micros();
    TIMES.lock().unwrap().push((name, dur));
}

// Buffer‑Allokationen

pub static ALLOCS:      AtomicUsize = AtomicUsize::new(0);
pub static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);


pub fn summary() {
    // API‑Latenzen hrouping
    let mut map: HashMap<&str, Vec<u128>> = HashMap::new();
    {
        let mut times = TIMES.lock().unwrap();
        for (name, us) in times.drain(..) {
            map.entry(name).or_default().push(us);
        }
    }

    println!("── metrics summary ──");
    for (name, mut v) in map {
    v.sort_unstable();
    let mean = v.iter().sum::<u128>() / v.len() as u128;
    let p95  = v[((v.len() * 95) / 100).saturating_sub(1)];

    println!("{:<18} mean={:>5} µs   p95={:>5} µs", name, mean, p95);

    //TO-DO NEEDS FIXING THROUGHPUT WRONG
    if name == "enqueue_write" {
        // Approximate throughput from total bytes & total time
        let total_us: u128 = v.iter().sum();
        let gbps = (crate::ALLOC_BYTES.load(Ordering::Relaxed) as f64)
                 / (total_us as f64) / 1e3; // GiB/s
        println!("    ↳ throughput ≈ {:.2} GiB/s", gbps);
    }
}

    /* Allokations‑Zähler */
    let allocs = ALLOCS.load(Ordering::Relaxed);
    let bytes  = ALLOC_BYTES.load(Ordering::Relaxed);
    println!("GPU allocations: {}   ({} MiB)", allocs, bytes / 1024 / 1024);
}
