// crates/hpc-core/examples/vec_add.rs
// 2025 Thomas Bicanic – MIT License
//
// Vektoraddition mit Safe-RustCL-Wrapper (Typ-State + Metrics + MemTrace)

use bytemuck::{cast_slice, cast_slice_mut};
use hpc_core::{ClError, GpuBuffer, Queued, Ready};

use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::Kernel,
    platform::get_platforms,
    program::Program,
};

#[cfg(feature = "metrics")]
use hpc_core::summary;
#[cfg(feature = "memtrace")]
use hpc_core::{start as trace_start, Dir, flush_csv};

fn main() -> Result<(), ClError> {
    // 1) OpenCL-Setup
    let platform   = get_platforms()?.remove(0);
    let device_id  = platform.get_devices(CL_DEVICE_TYPE_GPU)?[0];
    let device     = Device::new(device_id);
    let context    = Context::from_device(&device)?;
    let queue      = CommandQueue::create(&context, device.id(), CL_QUEUE_PROFILING_ENABLE)?;

    // 2) Host-Daten vorbereiten
    let n           = 1 << 22;                          // 4 Mi Elemente
    let size_bytes  = n * std::mem::size_of::<f32>();
    let h_a         = vec![1.0_f32; n];
    let h_b         = vec![2.0_f32; n];
    let mut h_out   = vec![0.0_f32; n];

    // 3) Device-Puffer über Wrapper anlegen
    let a_dev   = GpuBuffer::<Queued>::new(&context, size_bytes)?;
    let b_dev   = GpuBuffer::<Queued>::new(&context, size_bytes)?;
    let out_dev = GpuBuffer::<Queued>::new(&context, size_bytes)?;

// error[E0463]: missing field `cl_mem` in initializer of `GpuBuffer<Ready>`

    // 4) Host→Device (A)
    #[cfg(feature = "memtrace")]
    let tok_a = trace_start(Dir::H2D, size_bytes);
    let (a_if, guard_a) = a_dev.enqueue_write(&queue, cast_slice(&h_a))?;
    let a_ready: GpuBuffer<Ready> = a_if.into_ready(guard_a);
    #[cfg(feature = "memtrace")]
    tok_a.finish();

    // 5) Host→Device (B)
    #[cfg(feature = "memtrace")]
    let tok_b = trace_start(Dir::H2D, size_bytes);
    let (b_if, guard_b) = b_dev.enqueue_write(&queue, cast_slice(&h_b))?;
    let b_ready: GpuBuffer<Ready> = b_if.into_ready(guard_b);
    #[cfg(feature = "memtrace")]
    tok_b.finish();

    // 6) Host→Device (Out-Initialisierung)
    #[cfg(feature = "memtrace")]
    let tok_o = trace_start(Dir::H2D, size_bytes);
    let (o_if, guard_o) = out_dev.enqueue_write(&queue, cast_slice(&h_out))?;
    let out_ready: GpuBuffer<Ready> = o_if.into_ready(guard_o);
    #[cfg(feature = "memtrace")]
    tok_o.finish();

    // 7) Kernel starten
    #[cfg(feature = "memtrace")]
    let tok_k = trace_start(Dir::Kernel, 0);
    let src     = include_str!("../examples/vec_add.cl");
    let program = Program::create_and_build_from_source(&context, src, "")
        .map_err(|_| ClError::Api(-3))?;
    let kernel  = Kernel::create(&program, "vec_add")?;
    kernel.set_arg(0, a_ready.raw())?;
    kernel.set_arg(1, b_ready.raw())?;
    kernel.set_arg(2, out_ready.raw())?;
    let global = [n, 1, 1];
    queue.enqueue_nd_range_kernel(
        kernel.get(), 1,
        std::ptr::null(), global.as_ptr(),
        std::ptr::null(), &[],
    )?;
    queue.finish()?;  // warte auf Kernel
    #[cfg(feature = "memtrace")]
    tok_k.finish();

    // 8) Device→Host (Out lesen)
    #[cfg(feature = "memtrace")]
    let tok_d = trace_start(Dir::D2H, size_bytes);
    let (read_if, guard_read) = out_ready.enqueue_read(&queue, cast_slice_mut(&mut h_out))?;
    let _final: GpuBuffer<Ready> = read_if.into_ready(guard_read);
    #[cfg(feature = "memtrace")]
    tok_d.finish();

    // 9) Verifikation
    assert!(h_out.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    println!("vec_add OK, first element = {}", h_out[0]);

    // 10) Abschlussberichte
    #[cfg(feature = "metrics")]
    summary();
    #[cfg(feature = "memtrace")]
    flush_csv();

    Ok(())
}
