// examples/abort_token.rs


use std::thread::sleep;
use std::time::Duration;

#[cfg(feature = "memtrace")]
use hpc_core::memtracer::{self, Dir, AbortTokenGuard};

fn main() {
    #[cfg(feature = "memtrace")]
    // 1) Auto-Trace aktivieren
    memtracer::enable_auto_trace();

    // 2) Ohne Token loggen
    {
        #[cfg(feature = "memtrace")]
        let _t = memtracer::start(Dir::H2D, 1024);
        sleep(Duration::from_millis(10));
    }

    // 3) Mit dauerhaft gesetztem Token
    #[cfg(feature = "memtrace")]
    memtracer::set_abort_token("permanent-123");
    {
        #[cfg(feature = "memtrace")]
        let _t = memtracer::start(Dir::D2H, 2048);
        sleep(Duration::from_millis(15));
    }
    #[cfg(feature = "memtrace")]
    memtracer::clear_abort_token();

    // 4) Mit temporärem Token (Guard)
    {
        #[cfg(feature = "memtrace")]
        let _guard = AbortTokenGuard::new("guarded-XYZ");
        #[cfg(feature = "memtrace")]
        let _t = memtracer::start(Dir::Kernel, 4096);
        sleep(Duration::from_millis(5));
    }

    #[cfg(feature = "memtrace")]
{
    let s = memtracer::now_us();
    let ev = memtracer::AbortEvent {
        tx_id: 1,
        cause: "test_conflict".into(),
        retries: 2,
        conflict_sz: 64,
        t_start_us: s,
        t_end_us: s + 100,
    };
    memtracer::log_abort(&ev);
}

    // 5) Flush ins CSV
    #[cfg(feature = "memtrace")]
    memtracer::flush_csv();

    println!("memtrace.csv geschrieben – bitte Datei prüfen.");


}