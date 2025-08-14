//benches vec_add example. Not used in work. Not up to date



use criterion::{Criterion, criterion_group, BenchmarkId, criterion_main};
use hpc_core::{ClError, GpuBuffer, Queued};
use bytemuck::cast_slice;
use bytemuck::cast_slice_mut;
use opencl3::{
    context::Context, command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, program::Program, kernel::Kernel,
    platform::get_platforms, device::{Device, CL_DEVICE_TYPE_GPU},
};
use std::time::Duration;

const SIZES: &[usize] = &[
    1 << 10,
    1 << 14,
    1 << 18,
    1 << 22,
];

fn bench_vec_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec_add");

    for &bytes in SIZES {
        // wie viele f32-Elemente passen in `bytes`?
        let elements = bytes / std::mem::size_of::<f32>();
        let id = BenchmarkId::from_parameter(bytes);

        group.bench_with_input(id, &elements, |b, &n| {
            // einmalige Setup-Arbeit pro Input-Größe
            // (wird nicht in die Messung einbezogen)
            let platform = get_platforms().unwrap().remove(0);
            let dev_id   = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap()[0];
            let device   = Device::new(dev_id);
            let context  = Context::from_device(&device).unwrap();
            let queue    = CommandQueue::create(
                &context, device.id(), CL_QUEUE_PROFILING_ENABLE,
            ).unwrap();

            // Kernel laden & kompilieren
            let src     = include_str!("../examples/vec_add.cl");
            let program = Program::create_and_build_from_source(&context, src, "")
                .map_err(|_| ClError::Api(-3)).unwrap();
            let kernel = Kernel::create(&program, "vec_add").unwrap();
            // Buffer anlegen
            let size_bytes = n * std::mem::size_of::<f32>();
            let a_buf = GpuBuffer::<Queued>::new(&context, size_bytes).unwrap();
            let b_buf = GpuBuffer::<Queued>::new(&context, size_bytes).unwrap();
            let out_buf = GpuBuffer::<Queued>::new(&context, size_bytes).unwrap();

            // Host-Daten vorbereiten
            let h_a = vec![1.0_f32; n];
            let h_b = vec![2.0_f32; n];

            // Preload auf die GPU
            let (a_if, g_a) = a_buf.enqueue_write(&queue, cast_slice(&h_a)).unwrap();
            let (b_if, g_b) = b_buf.enqueue_write(&queue, cast_slice(&h_b)).unwrap();
            let a_ready = a_if.into_ready(g_a);
            let b_ready = b_if.into_ready(g_b);

            kernel.set_arg(0, a_ready.raw()).unwrap();
            kernel.set_arg(1, b_ready.raw()).unwrap();
            kernel.set_arg(2, out_buf.raw()).unwrap();

            let global_work_size = [n, 1, 1];

            // Hier beginnt die eigentliche Messung:
            b.iter(|| {
                // Alle drei Puffer müssen vor jedem Durchlauf neu gestartet werden,
                // sonst wird nur das erste Mal kopiert / gelayert.
                let evt = queue.enqueue_nd_range_kernel(
                    kernel.get(),
                    1,
                    std::ptr::null(),
                    global_work_size.as_ptr(),
                    std::ptr::null(),
                    &[],
                ).unwrap();
                evt.wait().unwrap();

                // Read-Back (optional, wenn du das mitmessen willst)
                let mut h_out = vec![0.0_f32; n];
                queue.enqueue_read_buffer(
                    out_buf.raw(),
                    opencl3::types::CL_BLOCKING,
                    0,
                    cast_slice_mut(&mut h_out),
                    &[],
                ).unwrap();
                queue.finish().unwrap();
            });
        });
    }

    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        // erlaube bis zu 30 s Messdauer
        .measurement_time(Duration::from_secs(20))
        // behalte sample_size=100 (Default)
}


// Diese Zeilen sind notwendig, damit Criterion den Benchmark ausführt
criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_vec_add
}
criterion_main!(benches);