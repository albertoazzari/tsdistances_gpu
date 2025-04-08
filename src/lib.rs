use std::{fmt::Error, sync::Arc};

use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::{self, GpuFuture},
};

const SHADER: &[u8] = include_bytes!(env!("tsdistances_gpu.spv"));

#[cfg(test)]
mod tests {

    #[test]
    fn test_device() {
        let (physical_device, _, queue_family_index) = super::utils::get_device();
        println!(
            "Physical device: {:?}",
            physical_device.properties().device_name
        );
        println!("Queue family index: {:?}", queue_family_index);
        println!(
            "Device type: {:?}",
            physical_device.properties().device_type
        );
    }

    #[test]
    fn test_shader() {
        let shader = super::SHADER;
        let (physical_device, device_extensions, queue_family_index) = super::utils::get_device();
        let result = super::compute_shader(
            shader,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &mut [0.0; 10],
            physical_device.clone(),
            device_extensions,
            queue_family_index,
        );
        assert!(result.is_ok(), "Failed to run compute shader: {:?}", result);
    }
}

mod utils {
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
            AllocationCreateInfo, MemoryAllocatePreference, MemoryTypeFilter,
            StandardMemoryAllocator,
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
}

pub fn compute_shader(
    shader: &[u8],
    input_a: &[f32],
    input_b: &[f32],
    diagonal: &mut [f32],
    physical_device: Arc<PhysicalDevice>,
    device_extensions: DeviceExtensions,
    queue_family_index: u32,
) -> Result<(), Error> {
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();
    let queue = queues.next().unwrap();
    let pipeline = {
        mod cs {
            use std::sync::Arc;

            use vulkano::{
                device::Device,
                shader::{ShaderModule, ShaderModuleCreateInfo},
                Validated, VulkanError,
            };

            pub fn load(
                device: Arc<Device>,
                shader: &[u8],
            ) -> Result<Arc<ShaderModule>, Validated<VulkanError>> {
                // convert from &[u8] to &[u32]
                unsafe {
                    let shader = std::slice::from_raw_parts(
                        shader.as_ptr() as *const u32,
                        shader.len() / std::mem::size_of::<u32>(),
                    );
                    ShaderModule::new(device, ShaderModuleCreateInfo::new(shader)).map_err(|e| {
                        eprintln!("Failed to load shader module: {:?}", e);
                        e
                    })
                }
            }
        }
        let cs = cs::load(device.clone(), shader)
            .unwrap()
            .entry_point("main_cs")
            .unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap()
    };

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let gpu_a = utils::move_gpu(input_a, &mut builder, device.clone());
    let gpu_b = utils::move_gpu(input_b, &mut builder, device.clone());
    let diagonal = utils::move_gpu(diagonal, &mut builder, device.clone());

    let layout = &pipeline.layout().set_layouts()[0];
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let set = DescriptorSet::new(
        descriptor_set_allocator,
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, diagonal.clone()),
            WriteDescriptorSet::buffer(1, gpu_a.clone()),
            WriteDescriptorSet::buffer(2, gpu_b.clone()),
        ],
        [],
    )
    .unwrap();

    builder
        .bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set,
        )
        .unwrap();

    unsafe { builder.dispatch([1024, 1, 1]) }.unwrap();

    let diagonal = utils::move_cpu(diagonal, &mut builder, device.clone());

    let command_buffer = builder.build().unwrap();

    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let data_buffer_content = diagonal.read().unwrap();
    for n in 0..diagonal.len() {
        if n < 10 {
            println!(
                "Diagonal[{}]: {}",
                n,
                data_buffer_content[n as usize]
            );
        }
        assert_eq_with_tol!(data_buffer_content[n as usize], (input_a[n as usize] * input_b[n as usize]).log2());
    }
    println!("Success");
    Ok(())
}
