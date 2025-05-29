use std::sync::{Arc, LazyLock};

use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        BufferContents, BufferUsage, Subbuffer,
    },
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
        Queue, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator},
    VulkanLibrary,
};

#[macro_export]
macro_rules! assert_eq_with_tol {
    ($a:expr, $b:expr, $tol:expr) => {
        if ($a - $b).abs() > $tol {
            panic!(
                "assertion failed: `(left == right)`\n  left: `{:?}`\n right: `{:?}`",
                $a, $b
            );
        }
    };
    ($a:expr, $b:expr) => {
        assert_eq_with_tol!($a, $b, 1e-6);
    };
}

type CachedCore = (
    Arc<Device>,
    Arc<Queue>,
    Arc<StandardCommandBufferAllocator>,
    Arc<StandardDescriptorSetAllocator>,
    Arc<StandardMemoryAllocator>, // memory allocator is Sync
);

static DEVICE_CORE: LazyLock<CachedCore> = LazyLock::new(|| {
    let start_time = std::time::Instant::now();
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .unwrap();
    println!("instance: {:?}", start_time.elapsed());

    let device_extensions = DeviceExtensions::empty();

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();
    println!(
        "physical device: {:?}",
        start_time.elapsed()
    );
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            enabled_features: {
                let mut features = DeviceFeatures::default();
                features.vulkan_memory_model = true;
                features.shader_int8 = true;
                features.shader_int64 = true;
                features.shader_float64 = true;
                features
            },
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();
    println!("device: {:?}", start_time.elapsed());
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));
    println!("command buffer allocator: {:?}", start_time.elapsed());
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));
    println!("descriptor set allocator: {:?}", start_time.elapsed());
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    println!("memory allocator: {:?}", start_time.elapsed());
    (
        device,
        queues.next().unwrap(),
        command_buffer_allocator,
        descriptor_set_allocator,
        memory_allocator,
    )
});

pub fn get_device() -> (
    Arc<Device>,
    Arc<Queue>,
    Arc<StandardCommandBufferAllocator>,
    Arc<StandardDescriptorSetAllocator>,
    Arc<SubbufferAllocator>,
) {
    let start_time = std::time::Instant::now();
    let (device, queue, command_buffer_allocator, descriptor_set_allocator, memory_allocator) =
        DEVICE_CORE.clone();

    let buffer_allocator = Arc::new(SubbufferAllocator::new(
        memory_allocator,
        SubbufferAllocatorCreateInfo {
            buffer_usage: BufferUsage::TRANSFER_DST
                | BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_SRC,
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,

            ..Default::default()
        },
    ));
    println!("get_device: {:?}", start_time.elapsed());
    (
        device,
        queue,
        command_buffer_allocator,
        descriptor_set_allocator,
        buffer_allocator,
    )
}

pub fn move_gpu<T: BufferContents + Copy>(
    data: &[T],
    subbuffer_allocator: &Arc<SubbufferAllocator>,
) -> Subbuffer<[T]> {
    let buffer = subbuffer_allocator
        .allocate_slice(data.len() as u64)
        .unwrap();
    buffer.write().unwrap().copy_from_slice(&data);
    buffer
}
