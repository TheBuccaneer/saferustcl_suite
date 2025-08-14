// examples/bandwidth_wrapper_fixed.rs
// 2025 - Fair bandwith with wrapper test

use bytemuck::{cast_slice, cast_slice_mut};
use hpc_core::{ClError, GpuBuffer, Queued, Ready};
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE},
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    platform::get_platforms,
};
use std::{env, time::Instant};

#[cfg(feature = "metrics")]
use hpc_core::summary;

#[cfg(feature = "memtrace")]
use hpc_core::{flush_csv, TracingScope};

fn main() -> Result<(), ClError> {
    // 1) OpenCL Setup
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
    
    println!("Testing FIXED WRAPPER bandwidth:");
    println!("  Total: {} MB, {} buffers of {:.1} MB each", 
             size_mb, num_buffers, (chunk_bytes as f64) / (1024.0 * 1024.0));
    println!("  {} iterations", iterations);

    // 3) Host-Daten - wie Raw-Version
    let mut host_data = vec![0.0f32; total_floats];
    let mut result_data = vec![0.0f32; total_floats];
    
    for i in 0..total_floats {
        host_data[i] = i as f32;
    }

    // 4) GPU-Buffer NUR EINMAL allokieren (wie Raw-Version)
    println!("Allocating {} GPU buffers...", num_buffers);
    let mut gpu_buffers: Vec<GpuBuffer<Queued>> = Vec::new();
    
    #[cfg(feature = "memtrace")]
    let _trace_scope = TracingScope::disabled();
    
    for _ in 0..num_buffers {
        gpu_buffers.push(GpuBuffer::new(&context, chunk_bytes)?);
    }
    
    println!("✓ GPU buffers allocated");

    // 5) H2D Benchmark - faire Messung ohne Re-Allokation
    let mut h2d_total_time = 0.0;
    
    for iter in 0..iterations {
        // Neue Buffer für diese Iteration (unvermeidbar wegen Type-State)
        let mut iter_buffers: Vec<GpuBuffer<Queued>> = Vec::new();
        for _ in 0..num_buffers {
            iter_buffers.push(GpuBuffer::new(&context, chunk_bytes)?);
        }
        
        // ✅ Timer startet nach der Allokation
        let start = Instant::now();
        
        // Parallel H2D für alle Chunks
        for (i, gpu_buf) in iter_buffers.into_iter().enumerate() {
            let start_idx = i * chunk_size;
            let end_idx = start_idx + chunk_size;
            let chunk_data = &host_data[start_idx..end_idx];
            
            let (in_flight, guard) = gpu_buf.enqueue_write(&queue, cast_slice(chunk_data))?;
            // Guard automatisch warten lassen
            drop(guard);
            drop(in_flight);
        }
        
        let elapsed = start.elapsed().as_secs_f64();
        h2d_total_time += elapsed;
        
        if iter == 0 {
            println!("First H2D (fixed wrapper): {:.3} ms", elapsed * 1000.0);
        }
    }

    // 6) D2H Benchmark - Buffer vorbereiten AUSSERHALB der Messung
    println!("Preparing buffers for D2H test...");
    let mut prepared_buffers: Vec<GpuBuffer<Ready>> = Vec::new();
    
    for i in 0..num_buffers {
        let gpu_buf = GpuBuffer::new(&context, chunk_bytes)?;
        let start_idx = i * chunk_size;
        let end_idx = start_idx + chunk_size;
        let chunk_data = &host_data[start_idx..end_idx];
        
        let (in_flight, guard) = gpu_buf.enqueue_write(&queue, cast_slice(chunk_data))?;
        let ready = in_flight.into_ready(guard);
        prepared_buffers.push(ready);
    }
    
    let mut d2h_total_time = 0.0;
    
    for iter in 0..iterations {
        // Neue Ready-Buffer für diese Iteration
        let mut iter_ready_buffers: Vec<GpuBuffer<Ready>> = Vec::new();
        if iter > 0 {
            // Für weitere Iterationen neu vorbereiten
            for i in 0..num_buffers {
                let gpu_buf = GpuBuffer::new(&context, chunk_bytes)?;
                let start_idx = i * chunk_size;
                let end_idx = start_idx + chunk_size;
                let chunk_data = &host_data[start_idx..end_idx];
                
                let (in_flight, guard) = gpu_buf.enqueue_write(&queue, cast_slice(chunk_data))?;
                let ready = in_flight.into_ready(guard);
                iter_ready_buffers.push(ready);
            }
        } else {
            iter_ready_buffers = prepared_buffers;
            prepared_buffers = Vec::new(); // Move out
        }
        
        // ✅ Timer startet bei reinen D2H-Transfers
        let start = Instant::now();
        
        for (i, ready_buf) in iter_ready_buffers.into_iter().enumerate() {
            let start_idx = i * chunk_size;
            let end_idx = start_idx + chunk_size;
            let chunk_result = &mut result_data[start_idx..end_idx];
            
            let (in_flight, guard) = ready_buf.enqueue_read(&queue, cast_slice_mut(chunk_result))?;
            // Warten durch Guard-Drop
            drop(guard);
            drop(in_flight);
        }
        
        let elapsed = start.elapsed().as_secs_f64();
        d2h_total_time += elapsed;
        
        if iter == 0 {
            println!("First D2H (fixed wrapper): {:.3} ms", elapsed * 1000.0);
        }
    }

    // 7) Reine Transfer-Geschwindigkeit (ohne Allokationen)  
    println!("\nTesting pure transfer speed (single large buffer)...");
    
    let mut pure_h2d_time = 0.0;
    let mut pure_d2h_time = 0.0;
    
    // Pure H2D - mehrere Messungen mit frischen Buffern
    for iter in 0..5 {
        let big_buffer = GpuBuffer::new(&context, total_floats * 4)?;
        let start = Instant::now();
        let (in_flight, guard) = big_buffer.enqueue_write(&queue, cast_slice(&host_data))?;
        drop(guard); // Warten
        drop(in_flight);
        let elapsed = start.elapsed().as_secs_f64();
        pure_h2d_time += elapsed;
        
        if iter == 0 {
            println!("First pure H2D: {:.3} ms", elapsed * 1000.0);
        }
    }
    
    // Pure D2H - Buffer mit Daten vorbereiten
    for iter in 0..5 {
        // Jede Iteration braucht einen frischen Buffer (wegen Move-Semantik)
        let big_buffer = GpuBuffer::new(&context, total_floats * 4)?;
        let (in_flight, guard) = big_buffer.enqueue_write(&queue, cast_slice(&host_data))?;
        let ready_for_d2h = in_flight.into_ready(guard);
        
        let start = Instant::now();
        let (in_flight, guard) = ready_for_d2h.enqueue_read(&queue, cast_slice_mut(&mut result_data))?;
        drop(guard); // Warten
        drop(in_flight);
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
    
    println!("\n=== FIXED WRAPPER RESULTS ===");
    println!("Data size: {:.2} GB", total_size_gb);
    println!("Multi-buffer H2D: {:.3} ms, {:.2} GB/s", h2d_avg * 1000.0, h2d_bandwidth);
    println!("Multi-buffer D2H: {:.3} ms, {:.2} GB/s", d2h_avg * 1000.0, d2h_bandwidth);
    println!("Pure single H2D:  {:.3} ms, {:.2} GB/s", pure_h2d_avg * 1000.0, pure_h2d_bandwidth);
    println!("Pure single D2H:  {:.3} ms, {:.2} GB/s", pure_d2h_avg * 1000.0, pure_d2h_bandwidth);
    
    let pcie3_x16 = 12.8;
    println!("\n=== WRAPPER EFFICIENCY ===");
    println!("Pure wrapper efficiency: H2D {:.1}%, D2H {:.1}%", 
             (pure_h2d_bandwidth / pcie3_x16) * 100.0,
             (pure_d2h_bandwidth / pcie3_x16) * 100.0);
    
    // 9) Overhead-Analyse
    println!("\n=== OVERHEAD ANALYSIS ===");
    println!("Multi-buffer overhead: H2D {:.1}%, D2H {:.1}%", 
             ((pure_h2d_bandwidth - h2d_bandwidth) / pure_h2d_bandwidth) * 100.0,
             ((pure_d2h_bandwidth - d2h_bandwidth) / pure_d2h_bandwidth) * 100.0);
    
    // 10) Verification
    for i in (0..total_floats).step_by(total_floats / 100) {
        assert!((host_data[i] - result_data[i]).abs() < 1e-6);
    }
    println!("✓ Data integrity verified");
    
    #[cfg(feature = "metrics")]
    {
        println!("\n=== METRICS ===");
        summary();
    }
    
    Ok(())
}