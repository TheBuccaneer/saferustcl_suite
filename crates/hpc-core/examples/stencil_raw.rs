// examples/stencil_raw.rs
// 2025 Thomas Bicanic – MIT License
//
// 2D Jacobi-Stencil with openCL3 (Raw-Version)

use bytemuck::{cast_slice, cast_slice_mut};
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::Kernel,
    memory::{Buffer, CL_MEM_READ_WRITE},
    platform::get_platforms,
    program::Program,
    types::CL_BLOCKING,
};
use hpc_core::ClError;

#[cfg(feature = "metrics")]
use hpc_core::summary;
#[cfg(feature = "memtrace")]
use hpc_core::{start as trace_start, Dir, flush_csv};


use std::env;

fn main() -> Result<(), ClError> {
    // 1) OpenCL-Setup
    let platform  = get_platforms()?.remove(0);
    let device_id = platform.get_devices(CL_DEVICE_TYPE_GPU)?[0];
    let device    = Device::new(device_id);
    let context   = Context::from_device(&device)?;
    let queue     = CommandQueue::create(&context, device.id(), CL_QUEUE_PROFILING_ENABLE)?;

    // 2) Parameter
    let args: Vec<String> = env::args().collect();
    let width  = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1026);
    let height = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1026);
    let total  = width * height;

    #[cfg(feature = "memtrace")]
    let size_bytes = total * std::mem::size_of::<f32>();


    let mut h_src = vec![0.0f32; total];
    let mut h_dst = vec![0.0f32; total];
    for y in 0..height {
        for x in 0..width {
            h_src[y*width + x] = (y*width + x) as f32;
        }
    }

    // 3) Device-Buffers
    let mut src_buf = Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, total, std::ptr::null_mut())?;
    let mut dst_buf = Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, total, std::ptr::null_mut())?;

    // 4) Host→Device
    #[cfg(feature = "memtrace")]
    let tok_h2d = trace_start(Dir::H2D, size_bytes * 2);
    queue.enqueue_write_buffer(&mut src_buf, CL_BLOCKING, 0, cast_slice(&h_src), &[])?;
    queue.enqueue_write_buffer(&mut dst_buf, CL_BLOCKING, 0, cast_slice(&h_dst), &[])?;
    queue.finish()?;

    #[cfg(feature = "memtrace")]
    tok_h2d.finish();

    // 5) Kernel
    #[cfg(feature = "memtrace")]
    let tok_k = trace_start(Dir::Kernel, 0);

    let src_cl  = include_str!("../examples/stencil.cl");
    let program = Program::create_and_build_from_source(&context, src_cl, "")
    .map_err(|_s| ClError::Api(-3))?;
    let kernel  = Kernel::create(&program, "jacobi")?;
    kernel.set_arg(0, &src_buf)?;
    kernel.set_arg(1, &dst_buf)?;
    kernel.set_arg(2, &(width as i32))?;
    kernel.set_arg(3, &(height as i32))?;
    let global = [width, height, 1];
    queue.enqueue_nd_range_kernel(
        kernel.get(), 2,
        std::ptr::null(), global.as_ptr(),
        std::ptr::null(), &[],
    )?;
    queue.finish()?;
    #[cfg(feature = "memtrace")]
    tok_k.finish();

    // 6) Device→Host
    #[cfg(feature = "memtrace")]
    let tok_d2h = trace_start(Dir::D2H, size_bytes);
    queue.enqueue_read_buffer(&dst_buf, CL_BLOCKING, 0, cast_slice_mut(&mut h_dst), &[])?;
    queue.finish()?;
    #[cfg(feature = "memtrace")]
    tok_d2h.finish();

    // 7) Verifikation
    for y in 1..(height-1) {
        for x in 1..(width-1) {
            let idx = y*width + x;
            let up    = h_src[idx - width];
            let down  = h_src[idx + width];
            let left  = h_src[idx - 1];
            let right = h_src[idx + 1];
            let expected = 0.25 * (up + down + left + right);
            assert!((h_dst[idx] - expected).abs() < 1e-6);
        }
    }
    println!("stencil_raw OK for {}×{} grid", width, height);

    // 8) Reports
    #[cfg(feature = "metrics")]
    summary();
    #[cfg(feature = "memtrace")]
    flush_csv();

    Ok(())
}