use std::{cell::OnceCell, sync::Arc};

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo, PrimaryAutoCommandBuffer},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceExtensions, QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocatePreference, MemoryTypeFilter, StandardMemoryAllocator,
    },
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

pub fn get_device() -> (Arc<PhysicalDevice>, DeviceExtensions, u32) {
    let cell = OnceCell::new();
    let instance = cell.get_or_init(|| {
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
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };
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
    (physical_device, device_extensions, queue_family_index)
}

pub fn move_gpu<T: BufferContents + Copy>(
    data: &[T],
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    device: Arc<Device>,
) -> Subbuffer<[T]> {
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    // Create CPU-accessible source buffer
    let buffer_host = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data.iter().cloned(),
    )
    .unwrap_or_else(|e| {
        panic!(
            "Failed to create host buffer of len {}\n {:?}",
            data.len(),
            e
        );
    });

    // Create GPU-side destination buffer with TRANSFER_SRC for later readback
    let buffer_device = Buffer::new_slice(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST
                | BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            allocate_preference: MemoryAllocatePreference::AlwaysAllocate,
            ..Default::default()
        },
        data.len() as u64,
    )
    .unwrap_or_else(|e| {
        panic!(
            "Failed to create device buffer of len {}\n {:?}",
            data.len(),
            e
        );
    });

    builder
        .copy_buffer(CopyBufferInfo::buffers(buffer_host, buffer_device.clone()))
        .unwrap();

    buffer_device
}

pub fn move_cpu<T: BufferContents + Copy>(
    buffer_device: Subbuffer<[T]>,
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    device: Arc<Device>,
) -> Subbuffer<[T]> {
    // Create a host-visible buffer for receiving the data
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let buffer_host = Buffer::new_slice(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST, // Changed from TRANSFER_SRC to TRANSFER_DST
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        buffer_device.len(),
    )
    .unwrap_or_else(|e| {
        panic!(
            "Failed to create host buffer for reading back data\n {:?}",
            e
        );
    });

    builder
        .copy_buffer(CopyBufferInfo::buffers(buffer_device, buffer_host.clone()))
        .unwrap();

    buffer_host
}
