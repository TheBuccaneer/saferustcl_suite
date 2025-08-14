use hpc_core::{GpuBuffer, Queued, InFlight, Ready};
use opencl3::{
    platform::get_platforms,
    device::{Device, CL_DEVICE_TYPE_GPU},
    context::Context,
    command_queue::CommandQueue,
};

#[test]
fn opencl_typestate_transitions_work() {
    let platform = get_platforms().unwrap().remove(0);
    let device_ids = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap();
    let device = Device::new(device_ids[0]);
    let context = Context::from_device(&device).unwrap();
    let queue = CommandQueue::create(&context, device.id(), 0).unwrap();

    let host_data = vec![0u8; 4];

    let (inflight, guard) = GpuBuffer::<Queued>::new(&context, 4)
        .unwrap()
        .enqueue_write(&queue, &host_data)
        .unwrap();

    let _ready: GpuBuffer<Ready> = inflight.into_ready(guard);
}
