// Benchmark for Jacobi 4-point stencil – fair comparison with buffer allocated in every iteration.
// bench is buffer centric

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use hpc_core::{GpuBuffer, Queued, Ready};
use bytemuck::cast_slice;
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    memory::{Buffer, CL_MEM_READ_WRITE},
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::Kernel,
    platform::get_platforms,
    program::Program,
    types::CL_BLOCKING,
};

use std::ptr;
use std::time::Duration;

const NX: usize = 1024;
const NY: usize = 1024;
const N_BYTES: usize = NX * NY * std::mem::size_of::<f32>();
const N_ITERS: usize = 10;

fn bench_stencil(c: &mut Criterion) {
    let mut g = c.benchmark_group("jacobi4");

    g.throughput(Throughput::Bytes((NX * NY * 4 * 2 * N_ITERS) as u64));

    // Raw version with opencl3 
    g.bench_function("raw_jacobi_1024x1024_10iter_fair", |b| {
        b.iter_batched(
            || {
                let platform = get_platforms().unwrap().remove(0);
                let dev_id   = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap()[0];
                let device   = Device::new(dev_id);
                let ctx      = Context::from_device(&device).unwrap();
                let queue    = CommandQueue::create(&ctx, device.id(), CL_QUEUE_PROFILING_ENABLE).unwrap();
                let src      = include_str!("../examples/stencil.cl");
                let program  = Program::create_and_build_from_source(&ctx, src, "").unwrap();
                let kern     = Kernel::create(&program, "jacobi").unwrap();

               
                let mut buf_src = Buffer::<f32>::create(&ctx, CL_MEM_READ_WRITE, NX*NY, ptr::null_mut()).unwrap();
                queue.enqueue_write_buffer(&mut buf_src, CL_BLOCKING, 0, cast_slice(&vec![1.0_f32; NX*NY]), &[]).unwrap();
                
                (ctx, queue, kern, buf_src)
            },
            |(ctx, queue, kern, mut src_buf)| {
                
                for _ in 0..N_ITERS {
                    let mut dst_buf = Buffer::<f32>::create(&ctx, CL_MEM_READ_WRITE, NX*NY, ptr::null_mut()).unwrap();
                    
                    kern.set_arg(0, &src_buf).unwrap();
                    kern.set_arg(1, &dst_buf).unwrap();
                    kern.set_arg(2, &(NX as i32)).unwrap();
                    kern.set_arg(3, &(NY as i32)).unwrap();
                    
                    let evt = queue
                        .enqueue_nd_range_kernel(kern.get(), 2, std::ptr::null(), [NX,NY,1].as_ptr(), std::ptr::null(), &[])
                        .unwrap();
                    evt.wait().unwrap();
                    
                    // dst wird zu src für nächste Iteration (wie beim Wrapper)
                    src_buf = dst_buf;
                }
            },
            BatchSize::SmallInput,
        )
    });

    // ============================================================================
    // WRAPPER VERSION - Unverändert (allokiert bereits pro Iteration)
    // ============================================================================
    g.bench_function("wrapper_jacobi_1024x1024_10iter", |b| {
        b.iter_batched(
            || {
                let platform  = get_platforms().unwrap().remove(0);
                let dev_id    = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap()[0];
                let device    = Device::new(dev_id);
                let context   = Context::from_device(&device).unwrap();
                let queue     = CommandQueue::create(&context, device.id(), CL_QUEUE_PROFILING_ENABLE).unwrap();

                let src      = include_str!("../examples/stencil.cl");
                let program  = Program::create_and_build_from_source(&context, src, "").unwrap();
                let kern     = Kernel::create(&program, "jacobi").unwrap();

                let init = vec![1.0_f32; NX * NY];
                let (ping_buf, g) = GpuBuffer::<Queued>::new(&context, N_BYTES).unwrap()
                    .enqueue_write(&queue, cast_slice(&init)).unwrap();
                let ping_ready: GpuBuffer<Ready> = ping_buf.into_ready(g);

                (context, queue, kern, ping_ready)
            },
            |(context, queue, kern, mut ping)| {
                
                for _ in 0..N_ITERS {
                    let mut dst_if = GpuBuffer::<Queued>::new(&context, N_BYTES).unwrap().launch();

                    kern.set_arg(0, ping.raw()).unwrap();
                    kern.set_arg(1, dst_if.raw_mut()).unwrap();
                    kern.set_arg(2, &(NX as i32)).unwrap();
                    kern.set_arg(3, &(NY as i32)).unwrap();

                    let global = [NX, NY, 1];
                    let evt = queue
                        .enqueue_nd_range_kernel(kern.get(), 2, std::ptr::null(), global.as_ptr(), std::ptr::null(), &[])
                        .unwrap();
                    evt.wait().unwrap();

                    let ready_dst = dst_if.complete(evt);
                    ping = ready_dst;
                }
            },
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(20))
        .sample_size(30)
        .configure_from_args()
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_stencil
}
criterion_main!(benches);