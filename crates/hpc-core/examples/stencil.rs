// examples/stencil.rs
// 2025 Thomas Bicanic – MIT License
//
// 2D Jacobi-Stencil mit Safe-RustCL-Wrapper (GpuBuffer),
// exakt 3 MemTrace-Einträge: H2D, Kernel, D2H.

use bytemuck::{cast_slice, cast_slice_mut};
use hpc_core::{ClError, GpuBuffer, Queued, Ready};

#[cfg(feature = "metrics")]
use hpc_core::summary;

#[cfg(feature = "memtrace")]
use hpc_core::{start as trace_start, Dir, flush_csv, TracingScope};

use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::Kernel,
    platform::get_platforms,
    program::Program,
};

fn main() -> Result<(), ClError> {
    // 1) Setup
    let platform  = get_platforms()?.remove(0);
    let device_id = platform.get_devices(CL_DEVICE_TYPE_GPU)?[0];
    let device    = Device::new(device_id);
    let context   = Context::from_device(&device)?;
    let queue     = CommandQueue::create(&context, device.id(), CL_QUEUE_PROFILING_ENABLE)?;

    // 2) Params + Host data
    let args: Vec<String> = std::env::args().collect();
    let width  = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1026);
    let height = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1026);
    let total      = width * height;
    let size_bytes = total * std::mem::size_of::<f32>();
    let mut h_src = vec![0.0f32; total];
    let mut h_dst = vec![0.0f32; total];
    for i in 0..total { 
        h_src[i] = i as f32; 
    }

    // 3) Device buffers via wrapper
    let src_dev = GpuBuffer::<Queued>::new(&context, size_bytes)?;
    let dst_dev = GpuBuffer::<Queued>::new(&context, size_bytes)?;
    let src_ready: GpuBuffer<Ready>;
    let dst_ready: GpuBuffer<Ready>;

    // 4) Host→Device als ein logischer H2D-Block
    #[cfg(feature = "memtrace")]
    let tok_h2d = trace_start(Dir::H2D, size_bytes * 2);
    
    #[cfg(feature = "memtrace")]
    let _scope = TracingScope::new(false);
    
    let (si, gi) = src_dev.enqueue_write(&queue, cast_slice(&h_src))?;
    src_ready = si.into_ready(gi);
    let (di, gd) = dst_dev.enqueue_write(&queue, cast_slice(&h_dst))?;
    dst_ready = di.into_ready(gd);
    
    #[cfg(feature = "memtrace")]
    drop(_scope); // Re-aktiviere Auto-Tracing
    
    #[cfg(feature = "memtrace")]
    tok_h2d.finish();

    // 5) Kernel
    #[cfg(feature = "memtrace")]
    let tok_k = trace_start(Dir::Kernel, 0);
    
    let src_cl  = include_str!("../examples/stencil.cl");
    let program = Program::create_and_build_from_source(&context, src_cl, "")
        .map_err(|_| ClError::Api(-3))?;
    let kernel = Kernel::create(&program, "jacobi")?;
    kernel.set_arg(0, src_ready.raw())?;
    kernel.set_arg(1, dst_ready.raw())?;
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

    // 6) Device→Host (D2H) - Auto-Tracing funktioniert normal
    let (ri, gr) = dst_ready.enqueue_read(&queue, cast_slice_mut(&mut h_dst))?;
    let _        = ri.into_ready(gr);

    // 7) Reports
    #[cfg(feature = "memtrace")]
    flush_csv();
    #[cfg(feature = "metrics")]
    summary();
    

    // 8) Verification
    for y in 1..height-1 {
        for x in 1..width-1 {
            let idx = y * width + x;
            let expected = 0.25 * (
                h_src[idx - width] + h_src[idx + width] +
                h_src[idx - 1    ] + h_src[idx + 1    ]
            );
            assert!((h_dst[idx] - expected).abs() < 1e-6, 
                "Mismatch at ({},{}) idx={}: got={}, expected={}", 
                x, y, idx, h_dst[idx], expected);
        }
    }
    println!("stencil OK for {}×{} grid.", width, height);
    Ok(())
}