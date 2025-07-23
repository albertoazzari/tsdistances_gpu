use std::sync::{Arc, LazyLock};

use vulkano::{
    VulkanLibrary,
    buffer::{
        BufferContents, BufferUsage, Subbuffer,
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
    },
    command_buffer::{
        AutoCommandBufferBuilder, CommandBuffer, CopyBufferInfo,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator},
};

use crate::cpu;

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

#[derive(Clone)]
pub struct SubBuffersAllocator {
    gpu: Arc<SubbufferAllocator>,
    cpu: Arc<SubbufferAllocator>,
}

type CachedCore = (
    Arc<Device>,
    Arc<Queue>,
    Arc<StandardCommandBufferAllocator>,
    Arc<StandardDescriptorSetAllocator>,
    Arc<StandardMemoryAllocator>, // memory allocator is Sync
);

static DEVICE_CORE: LazyLock<CachedCore> = LazyLock::new(|| {
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .unwrap();

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
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            enabled_features: {
                let mut features = DeviceFeatures::default();
                features.shader_int8 = true;
                features.shader_int64 = true;
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
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
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
    SubBuffersAllocator,
) {
    let (device, queue, command_buffer_allocator, descriptor_set_allocator, memory_allocator) =
        DEVICE_CORE.clone();

    let gpu_buffer_allocator = Arc::new(SubbufferAllocator::new(
        memory_allocator.clone(),
        SubbufferAllocatorCreateInfo {
            buffer_usage: BufferUsage::TRANSFER_DST
                | BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_SRC,
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    ));

    let cpu_buffer_allocator = Arc::new(SubbufferAllocator::new(
        memory_allocator,
        SubbufferAllocatorCreateInfo {
            buffer_usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
    ));

    (
        device,
        queue,
        command_buffer_allocator,
        descriptor_set_allocator,
        SubBuffersAllocator {
            gpu: gpu_buffer_allocator,
            cpu: cpu_buffer_allocator,
        },
    )
}

pub fn move_gpu<T: BufferContents + Copy, L>(
    data: &[T],
    subbuffer_allocator: &SubBuffersAllocator,
    command_buffer: &mut AutoCommandBufferBuilder<L>,
    alignment: usize,
) -> Subbuffer<[T]> {
    let padded_len = data.len().div_ceil(alignment) * alignment;

    let cpu_buffer = subbuffer_allocator
        .cpu
        .allocate_slice(padded_len as u64)
        .unwrap();
    cpu_buffer.write().unwrap()[0..data.len()].copy_from_slice(&data);

    let gpu_buffer = subbuffer_allocator
        .gpu
        .allocate_slice(padded_len as u64)
        .unwrap();

    command_buffer
        .copy_buffer(CopyBufferInfo::buffers(
            cpu_buffer.clone(),
            gpu_buffer.clone(),
        ))
        .unwrap();

    gpu_buffer
}

pub fn move_cpu<T: BufferContents + Copy, L>(
    subbuffer_allocator: &SubBuffersAllocator,
    gpu_buffer: &Subbuffer<[T]>,
    command_buffer: &mut AutoCommandBufferBuilder<L>,
) -> Subbuffer<[T]> {
    let cpu_buffer = subbuffer_allocator
        .cpu
        .allocate_slice(gpu_buffer.len() as u64)
        .unwrap();
    command_buffer
        .copy_buffer(CopyBufferInfo::buffers(
            gpu_buffer.clone(),
            cpu_buffer.clone(),
        ))
        .unwrap();
    cpu_buffer
}
