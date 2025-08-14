// examples/stm_abort.rs
//
// Deterministisch: Barrier-Sync + per-Thread RNG-Seed.
// CLI: --threads, --conflict, (--ops ODER --duration), --seed
// Default: --ops 1_000_000. Bei Angabe beider gewinnt --ops.
// Aborts werden optional via feature "memtrace" geloggt.

use std::env;
use std::str::FromStr;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Barrier,
};
use std::thread;
use std::time::{Duration, Instant};

// ---- CLI ----

#[derive(Clone, Copy, Debug)]
enum Conflict {
    Low,
    Med,
    High,
}
impl FromStr for Conflict {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "low" => Ok(Conflict::Low),
            "med" | "medium" => Ok(Conflict::Med),
            "high" => Ok(Conflict::High),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Ops(u64),
    Duration(u64),
}

#[derive(Debug)]
struct Config {
    threads: usize,
    conflict: Conflict,
    mode: Mode,
    seed: u64,
}

fn parse_args() -> Config {
    let mut threads = 4usize;
    let mut conflict = Conflict::Low;
    let mut duration_s: Option<u64> = None;
    let mut ops: Option<u64> = None;
    let mut seed = 1u64;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--threads" => {
                if let Some(v) = args.next() {
                    threads = v.parse().unwrap_or(4);
                }
            }
            "--conflict" => {
                if let Some(v) = args.next() {
                    conflict = v.parse().unwrap_or(Conflict::Low);
                }
            }
            "--duration" => {
                if let Some(v) = args.next() {
                    duration_s = v.parse().ok();
                }
            }
            "--ops" => {
                if let Some(v) = args.next() {
                    ops = v.parse().ok();
                }
            }
            "--seed" => {
                if let Some(v) = args.next() {
                    seed = v.parse().unwrap_or(1);
                }
            }
            _ => {}
        }
    }

    // Priorität: ops > duration > default ops
    let mode = if let Some(n) = ops {
        Mode::Ops(n.max(1))
    } else if let Some(s) = duration_s {
        Mode::Duration(s.max(1))
    } else {
        Mode::Ops(1_000_000)
    };

    Config { threads, conflict, mode, seed }
}

// ---- sehr einfacher, deterministischer PRNG ----
#[derive(Clone)]
struct XorShift64 {
    state: u64,
}
impl XorShift64 {
    fn new(seed: u64) -> Self { Self { state: seed.max(1) } }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        let v = self.next_u32();
        (v as f32) / (u32::MAX as f32)
    }
}

// ---- Dummy-STM-Workload ----

fn main() {
    #[cfg(feature="memtrace")]
    eprintln!("memtrace: ENABLED");
    #[cfg(not(feature="memtrace"))]
    eprintln!("memtrace: DISABLED");

    let cfg = parse_args();

    // Konfliktwahrscheinlichkeit grob (feintuning später)
    let p_conflict = match cfg.conflict {
        Conflict::Low => 0.02_f32,
        Conflict::Med => 0.15_f32,
        Conflict::High => 0.40_f32,
    };

    // Zähler
    let aborts = Arc::new(AtomicU64::new(0));
    let commits = Arc::new(AtomicU64::new(0));

    // Barrier für synchronen Start
    let barrier = Arc::new(Barrier::new(cfg.threads));

    // Für Duration-Modus
    let stop = Arc::new(AtomicBool::new(false));
    let stop_at = match cfg.mode {
        Mode::Duration(s) => Some(Instant::now() + Duration::from_secs(s)),
        _ => None,
    };

    // Für Ops-Modus: faire Verteilung
    let (ops_base, ops_extra) = match cfg.mode {
        Mode::Ops(n) => (n / cfg.threads as u64, n % cfg.threads as u64),
        _ => (0, 0),
    };

    eprintln!(
        "stm_abort: threads={}, conflict={:?}, mode={:?}, seed={}",
        cfg.threads, cfg.conflict, cfg.mode, cfg.seed
    );

    let mut handles = Vec::with_capacity(cfg.threads);
    for tid in 0..cfg.threads {
        let barrier = barrier.clone();
        let aborts = Arc::clone(&aborts);
        let commits = Arc::clone(&commits);
        let stop_flag = stop.clone();
        let local_mode = match cfg.mode {
            Mode::Ops(_) => {
                let per_thread = ops_base + if (tid as u64) < ops_extra { 1 } else { 0 };
                Mode::Ops(per_thread)
            }
            Mode::Duration(s) => Mode::Duration(s),
        };
        let local_stop_at = stop_at; // Copy (Option<Instant> ist Copy)

        // deterministischer Seed je Thread
        let thread_seed = cfg.seed
            ^ (((tid as u64) + 1) << 32)
            ^ 0x9E37_79B9_7F4A_7C15u64;
        let mut rng = XorShift64::new(thread_seed);

        let h = thread::spawn(move || {
            // synchroner Start
            barrier.wait();

            // Fortschritt
            let mut done: u64 = 0;
            let mut last = Instant::now();

            match local_mode {
                Mode::Ops(ops) => {
                    for _ in 0..ops {
                        // Arbeit simulieren
                        spin_for_ns(1500 + (rng.next_u32() % 1500) as u64);

                        // Konfliktsampling
                        if rng.next_f32() < p_conflict {
                            aborts.fetch_add(1, Ordering::Relaxed);
                            // deterministischer Backoff
                            spin_for_ns(10_000 + ((tid as u64) * 1_000));

                            #[cfg(feature = "memtrace")]
                            hpc_core::memtracer::trace_abort(
                                /*tx_id*/ 0,
                                /*cause*/ "conflict",
                                /*retries*/ 1,
                                /*conflict_sz*/ 1,
                                /*abort_token*/ "stm",
                            );
                        } else {
                            commits.fetch_add(1, Ordering::Relaxed);
                        }

                        // Fortschritt ausgeben (ca. 1×/s)
                        done += 1;
                        if last.elapsed().as_secs() >= 1 {
                            eprintln!("progress tid={} done={}", tid, done);
                            last = Instant::now();
                        }
                    }
                }
                Mode::Duration(_) => {
                    let deadline = local_stop_at.expect("deadline missing");
                    while Instant::now() < deadline && !stop_flag.load(Ordering::Relaxed) {
                        spin_for_ns(1500 + (rng.next_u32() % 1500) as u64);

                        if rng.next_f32() < p_conflict {
                            aborts.fetch_add(1, Ordering::Relaxed);
                            spin_for_ns(10_000 + ((tid as u64) * 1_000));

                            #[cfg(feature = "memtrace")]
                            hpc_core::memtracer::trace_abort(0, "conflict", 1, 1, "stm");
                        } else {
                            commits.fetch_add(1, Ordering::Relaxed);
                        }

                        // Fortschritt ausgeben (ca. 1×/s)
                        done += 1;
                        if last.elapsed().as_secs() >= 1 {
                            eprintln!("progress tid={} done={}", tid, done);
                            last = Instant::now();
                        }
                    }
                }
            }
        });
        handles.push(h);
    }

    // Duration-Modus: sauber stoppen
    if let Mode::Duration(s) = cfg.mode {
        let dur = Duration::from_secs(s);
        let t0 = Instant::now();
        while t0.elapsed() < dur {
            thread::sleep(Duration::from_millis(5));
        }
        stop.store(true, Ordering::Relaxed);
    }

    for h in handles {
        let _ = h.join();
    }

    let a = aborts.load(Ordering::Relaxed);
    let c = commits.load(Ordering::Relaxed);

    println!("STM run finished.");
    println!("aborts_total: {}", a);
    println!("commits_total: {}", c);

    #[cfg(feature = "memtrace")]
    {
        hpc_core::memtracer::flush_csv();
        println!("memtrace.csv / memtrace_summary.txt geschrieben (falls Events vorhanden).");
    }
}


// sehr kleiner, portabler Busy-Wait (für deterministische Mikro-Sleeps)
#[inline(always)]
fn spin_for_ns(nanos: u64) {
    let start = Instant::now();
    while start.elapsed().as_nanos() < nanos as u128 {
        core::hint::spin_loop();
    }
}
