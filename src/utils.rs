use std::{cell::OnceCell, sync::Arc};

use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        BufferContents, BufferUsage, Subbuffer,
    },
    command_buffer::{self, allocator::StandardCommandBufferAllocator},
    descriptor_set::{self, allocator::StandardDescriptorSetAllocator},
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

pub fn get_device() -> (
    Arc<Device>,
    Arc<Queue>,
    Arc<StandardCommandBufferAllocator>,
    Arc<StandardDescriptorSetAllocator>,
    Arc<SubbufferAllocator>,
) {
    let instance_cell = OnceCell::new();
    let instance = instance_cell.get_or_init(|| {
        let library = VulkanLibrary::new().unwrap();
        Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .unwrap()
    });
    let device_extensions = DeviceExtensions {
        ..DeviceExtensions::empty()
    };

    let device_cell: OnceCell<(Arc<Device>, Arc<Queue>)> = OnceCell::new();
    let (device, queue) = device_cell.get_or_init(|| {
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
            physical_device.clone(),
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
        (device, queues.next().unwrap())
    });

    let allocator = OnceCell::new();
    let (command_buffer, descriptor_set, buffer) = allocator.get_or_init(|| {
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let buffer_allocator = Arc::new(SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::TRANSFER_DST
                    | BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        ));
        (
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
            buffer_allocator.clone(),
        )
    });

    (
        device.clone(),
        queue.clone(),
        command_buffer.clone(),
        descriptor_set.clone(),
        buffer.clone(),
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
