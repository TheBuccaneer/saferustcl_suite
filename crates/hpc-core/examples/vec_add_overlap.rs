// examples/vec_add_overlap_fast.rs
// 2025 Thomas Bicanic – MIT License
//
// Vektor-Addition mit Overlap: genau 3 MemTrace-Einträge (H2D, Kernel, D2H)

// Ganz oben in vec_add_overlap_fast.rs:
#[cfg(feature = "memtrace")]
use hpc_core::{start as trace_start, Dir, flush_csv};

#[cfg(not(feature = "memtrace"))]
mod memtrace_stubs {
    pub enum Dir { H2D, Kernel, D2H }
    pub struct Token;
    impl Token { pub fn finish(self) {} }
    pub fn trace_start(_d: Dir, _b: usize) -> Token { Token }
    pub fn flush_csv() {}
}
#[cfg(not(feature = "memtrace"))]
use memtrace_stubs::{trace_start, Dir, flush_csv};


use bytemuck::{cast_slice, cast_slice_mut};
#[cfg(feature = "metrics")]
use hpc_core::summary;




use hpc_core::{ClError};
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    event::Event,
    kernel::Kernel,
    memory::{Buffer, CL_MEM_READ_WRITE},
    platform::get_platforms,
    program::Program,
    types::{CL_NON_BLOCKING, CL_BLOCKING},
};

fn main() -> Result<(), ClError> {
    // 1) Setup & Build
    let platform = get_platforms()?.remove(0);
    let device   = Device::new(platform.get_devices(CL_DEVICE_TYPE_GPU)?[0]);
    let context  = Context::from_device(&device)?;
    // Zwei Queues: eine für Transfers, eine für Compute
    let props      = CL_QUEUE_PROFILING_ENABLE;
    let queue_xfer = CommandQueue::create(&context, device.id(), props)?;
    let queue_comp = CommandQueue::create(&context, device.id(), props)?;

    // 2) Parameter & Host-Puffer
    let n = std::env::args()
        .nth(1)
        .unwrap_or_else(|| (1 << 22).to_string())
        .parse::<usize>()
        .expect("need element count");
    let size_b = n * std::mem::size_of::<f32>();
    let h_a   = vec![1.0_f32; n];
    let h_b   = vec![2.0_f32; n];
    let mut h_out = vec![0.0_f32; n];

    // 3) Program & Kernel laden
    let src     = include_str!("../examples/vec_add.cl");
    let program = Program::create_and_build_from_source(&context, src, "")
        .map_err(|_| ClError::Api(-3))?;
    let kernel  = Kernel::create(&program, "vec_add")?;

    // 4) Device-Buffers anlegen
    let mut a_dev   = Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, n, std::ptr::null_mut())?;
    let mut b_dev   = Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, n, std::ptr::null_mut())?;
    let mut out_dev = Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, n, std::ptr::null_mut())?;
    kernel.set_arg(0, &a_dev)?;
    kernel.set_arg(1, &b_dev)?;
    kernel.set_arg(2, &out_dev)?;

    // 5) H2D: A+B Upload
    let tok_h2d = trace_start(Dir::H2D, size_b * 2);
    let _evt_a: Event = queue_xfer.enqueue_write_buffer(&mut a_dev,   CL_NON_BLOCKING, 0, cast_slice(&h_a),   &[])?;
    let evt_b: Event = queue_xfer.enqueue_write_buffer(&mut b_dev,   CL_NON_BLOCKING, 0, cast_slice(&h_b),   &[])?;
    queue_xfer.finish()?;
    tok_h2d.finish();

    // 6) Kernel
    let tok_k = trace_start(Dir::Kernel, 0);
    let raw_evt_b = evt_b.get();
    let global = [n, 1, 1];
    queue_comp.enqueue_nd_range_kernel(
        kernel.get(), 1,
        std::ptr::null(), global.as_ptr(),
        std::ptr::null(), &[raw_evt_b],
    )?;
    queue_comp.finish()?;
    tok_k.finish();

    // 7) D2H: Ergebnis-Download
    let tok_d2h = trace_start(Dir::D2H, size_b);
    queue_xfer.enqueue_read_buffer(&mut out_dev, CL_BLOCKING, 0, cast_slice_mut(&mut h_out), &[evt_b.get()])?;
    queue_xfer.finish()?;
    tok_d2h.finish();

    // 8) Reports
    flush_csv();

    // 9) Verification
    assert!(h_out.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    println!("vec_add_overlap_fast OK for {} elements", n);

    Ok(())
}
