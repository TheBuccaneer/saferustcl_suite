// examples/bandwidth_optimized.rs  
// 2025 - optimized bandwith-Test mit Out-of-Order Queue
// same structure as basic

use bytemuck::{cast_slice, cast_slice_mut};
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE},
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    memory::{Buffer, CL_MEM_READ_WRITE},
    platform::get_platforms,
    types::CL_NON_BLOCKING,
};
use hpc_core::ClError;
use std::{env, time::Instant, ptr};

fn main() -> Result<(), ClError> {
    // 1) OpenCL Setup mit Out-of-Order Queue
    let platform = get_platforms()?.remove(0);
    let device_id = platform.get_devices(CL_DEVICE_TYPE_GPU)?[0];
    let device = Device::new(device_id);
    let context = Context::from_device(&device)?;
    
    let queue_flags = CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    let queue = CommandQueue::create(&context, device.id(), queue_flags)?;

    // 2) Parameter
    let args: Vec<String> = env::args().collect();
    let size_mb = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let iterations = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let num_buffers = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    
    let total_floats = (size_mb * 1024 * 1024) / 4;
    let chunk_size = total_floats / num_buffers;
    let chunk_bytes = chunk_size * 4;
    
    println!("Testing OPTIMIZED RAW bandwidth:");
    println!("  Total: {} MB, {} buffers of {:.1} MB each", 
             size_mb, num_buffers, (chunk_bytes as f64) / (1024.0 * 1024.0));
    println!("  {} iterations", iterations);

    // 3) Host data
    let mut host_data = vec![0.0f32; total_floats];
    let mut result_data = vec![0.0f32; total_floats];
    
    for i in 0..total_floats {
        host_data[i] = i as f32;
    }

    // 4) GPU-Buffer
    println!("Allocating {} GPU buffers...", num_buffers);
    let mut device_buffers: Vec<Buffer<f32>> = Vec::new();
    for _ in 0..num_buffers {
        device_buffers.push(Buffer::<f32>::create(
            &context, CL_MEM_READ_WRITE, chunk_size, ptr::null_mut())?);
    }
    println!("✓ GPU buffers allocated");

    // 5) H2D Benchmark
    let mut h2d_total_time = 0.0;
    
    for iter in 0..iterations {
        let start = Instant::now();
        let mut events = Vec::new();
        
        for chunk in 0..num_buffers {
            let start_idx = chunk * chunk_size;
            let end_idx = start_idx + chunk_size;
            let chunk_data = &host_data[start_idx..end_idx];
            
            let evt = queue.enqueue_write_buffer(
                &mut device_buffers[chunk],
                CL_NON_BLOCKING,
                0,
                cast_slice(chunk_data),
                &[]
            )?;
            events.push(evt);
        }
        
        for evt in events {
            evt.wait()?;
        }
        
        let elapsed = start.elapsed().as_secs_f64();
        h2d_total_time += elapsed;
        
        if iter == 0 {
            println!("First H2D (optimized raw): {:.3} ms", elapsed * 1000.0);
        }
    }

    // 6) D2H Benchmark
    println!("Preparing buffers for D2H test...");
    println!("✓ Buffers ready for D2H");
    
    let mut d2h_total_time = 0.0;
    
    for iter in 0..iterations {  
        let start = Instant::now();
        let mut events = Vec::new();
        
        for chunk in 0..num_buffers {
            let start_idx = chunk * chunk_size;
            let end_idx = start_idx + chunk_size;
            let chunk_result = &mut result_data[start_idx..end_idx];
            
            let evt = queue.enqueue_read_buffer(
                &device_buffers[chunk],
                CL_NON_BLOCKING,
                0,
                cast_slice_mut(chunk_result),
                &[]
            )?;
            events.push(evt);
        }
        
        for evt in events {
            evt.wait()?;
        }
        
        let elapsed = start.elapsed().as_secs_f64();
        d2h_total_time += elapsed;
        
        if iter == 0 {
            println!("First D2H (optimized raw): {:.3} ms", elapsed * 1000.0);
        }
    }

    // 7) Pure Transfer Tests
    println!("\nTesting pure transfer speed (single large buffer)...");
    
    let mut pure_h2d_time = 0.0;
    let mut pure_d2h_time = 0.0;
    
    // Pure H2D
    for iter in 0..5 {
        let mut big_buffer = Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, total_floats, ptr::null_mut())?;
        let start = Instant::now();
        let evt = queue.enqueue_write_buffer(&mut big_buffer, CL_NON_BLOCKING, 0, cast_slice(&host_data), &[])?;
        evt.wait()?;
        let elapsed = start.elapsed().as_secs_f64();
        pure_h2d_time += elapsed;
        
        if iter == 0 {
            println!("First pure H2D: {:.3} ms", elapsed * 1000.0);
        }
    }
    
    // Pure D2H
    let mut big_buffer = Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, total_floats, ptr::null_mut())?;
    let evt = queue.enqueue_write_buffer(&mut big_buffer, CL_NON_BLOCKING, 0, cast_slice(&host_data), &[])?;
    evt.wait()?;
    
    for iter in 0..5 {
        let start = Instant::now();
        let evt = queue.enqueue_read_buffer(&big_buffer, CL_NON_BLOCKING, 0, cast_slice_mut(&mut result_data), &[])?;
        evt.wait()?;
        let elapsed = start.elapsed().as_secs_f64();
        pure_d2h_time += elapsed;
        
        if iter == 0 {
            println!("First pure D2H: {:.3} ms", elapsed * 1000.0);
        }
    }

    // 8) Results
    let h2d_avg = h2d_total_time / iterations as f64;
    let d2h_avg = d2h_total_time / iterations as f64;
    let pure_h2d_avg = pure_h2d_time / 5.0;
    let pure_d2h_avg = pure_d2h_time / 5.0;
    
    let total_size_gb = (size_mb as f64) / 1024.0;
    let h2d_bandwidth = total_size_gb / h2d_avg;
    let d2h_bandwidth = total_size_gb / d2h_avg;
    let pure_h2d_bandwidth = total_size_gb / pure_h2d_avg;
    let pure_d2h_bandwidth = total_size_gb / pure_d2h_avg;
    
    println!("\n=== OPTIMIZED RAW RESULTS ===");
    println!("Data size: {:.2} GB ({} buffers)", total_size_gb, num_buffers);
    println!("Multi-buffer H2D: {:.3} ms, {:.2} GB/s", h2d_avg * 1000.0, h2d_bandwidth);
    println!("Multi-buffer D2H: {:.3} ms, {:.2} GB/s", d2h_avg * 1000.0, d2h_bandwidth);
    println!("Pure single H2D:  {:.3} ms, {:.2} GB/s", pure_h2d_avg * 1000.0, pure_h2d_bandwidth);
    println!("Pure single D2H:  {:.3} ms, {:.2} GB/s", pure_d2h_avg * 1000.0, pure_d2h_bandwidth);
    
    let pcie3_x16 = 12.8;
    println!("\n=== OPTIMIZED RAW EFFICIENCY ===");
    println!("Pure optimized efficiency: H2D {:.1}%, D2H {:.1}%", 
             (pure_h2d_bandwidth / pcie3_x16) * 100.0,
             (pure_d2h_bandwidth / pcie3_x16) * 100.0);
    
    println!("\n=== OVERHEAD ANALYSIS ===");
    println!("Multi-buffer overhead: H2D {:.1}%, D2H {:.1}%", 
             ((pure_h2d_bandwidth - h2d_bandwidth) / pure_h2d_bandwidth) * 100.0,
             ((pure_d2h_bandwidth - d2h_bandwidth) / pure_d2h_bandwidth) * 100.0);
    
    // 9) Verification
    for i in (0..total_floats).step_by(total_floats / 100) {
        assert!((host_data[i] - result_data[i]).abs() < 1e-6);
    }
    println!("✓ Data integrity verified");
    
    Ok(())
}