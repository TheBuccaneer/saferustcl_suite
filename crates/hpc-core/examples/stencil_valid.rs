/* 
    examples.stencil_valid.rs
    2025 Tihomir Thomas Bicanic – MIT License

    // verificats examples/stencil.cl 
*/


use hpc_core::ClError;
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
use std::env;

fn main() -> Result<(), ClError> {

    // usual setup for normale opencl3
    let platforms = get_platforms()?;
    let platform  = &platforms[0];
    let device_id = platform.get_devices(CL_DEVICE_TYPE_GPU)?[0];
    let device    = Device::new(device_id);
    let context   = Context::from_device(&device)?;
    let queue     = CommandQueue::create(&context, device.id(), CL_QUEUE_PROFILING_ENABLE)?;

    // parameter given as parameters
    let args: Vec<String> = env::args().collect();
    let width  = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    let height = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let total = width * height;

    
    let mut src = Vec::with_capacity(total);
    for y in 0..height {
        for x in 0..width {
            src.push((y as f32) * (width as f32) + (x as f32));
        }
    }
    let mut dst = vec![0.0f32; total];

    // Device-Puffer
    let  mut src_buf = Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, total, std::ptr::null_mut())?;
    let mut dst_buf = Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, total, std::ptr::null_mut())?;

    // Write data on GPU
    queue.enqueue_write_buffer(&mut src_buf, CL_BLOCKING, 0, cast_slice(&src), &[])?;
    queue.enqueue_write_buffer(&mut dst_buf, CL_BLOCKING, 0, cast_slice(&dst), &[])?;
    queue.finish()?;

    
    let src_cl  = include_str!("./stencil.cl");
    let program = Program::create_and_build_from_source(&context, src_cl, "")
    .map_err(|_s| ClError::Api(-3))?;
    let kernel  = Kernel::create(&program, "jacobi")?;
    kernel.set_arg(0, &src_buf)?;
    kernel.set_arg(1, &dst_buf)?;
    kernel.set_arg(2, &(width as i32))?;
    kernel.set_arg(3, &(height as i32))?;

    // ND-Range starten (2D)
    let global_dims = [width, height, 1];
    queue.enqueue_nd_range_kernel(
        kernel.get(), 2,
        std::ptr::null(), global_dims.as_ptr(),
        std::ptr::null(), &[],
    )?;
    queue.finish()?;


    queue.enqueue_read_buffer(&mut dst_buf, CL_BLOCKING, 0, cast_slice_mut(&mut dst), &[])?;
    queue.finish()?;

    // verification
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
                if (dst[idx] - src[idx]).abs() > 1e-6 {
                    panic!(
                        "Randabgleich fehlgeschlagen bei ({}, {}): dst={}, src={}",
                        x, y, dst[idx], src[idx]
                    );
                }
            } else {
                let up    = src[idx - width];
                let down  = src[idx + width];
                let left  = src[idx - 1];
                let right = src[idx + 1];
                let expected = 0.25 * (up + down + left + right);
                if (dst[idx] - expected).abs() > 1e-6 {
                    panic!(
                        "Stencil-Verifikation fehlgeschlagen bei ({}, {}): expected={}, got={}",
                        x, y, expected, dst[idx]
                    );
                }
            }
        }
    }

    println!("Stencil-Validierung erfolgreich für {}×{} Grid", width, height);
    Ok(())
}
