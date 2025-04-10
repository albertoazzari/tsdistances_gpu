#![cfg_attr(target_arch = "spirv", no_std)]

#[cfg(target_arch = "spirv")]
pub mod gpu {
    // HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
    // #![deny(warnings)]

    use glam::UVec3;
    use spirv_std::num_traits::Float;
    use spirv_std::{glam, spirv};

    #[spirv(compute(threads(64)))]
    pub fn erp(
        #[spirv(global_invocation_id)] id: UVec3,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input_a: &[f32],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] input_b: &[f32],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] diagonal: &mut [f32],
    ) {
        let index = id.x as usize;
        diagonal[index] = (input_a[index] * input_b[index]).log2();
    }
}
#[cfg(not(target_arch = "spirv"))]
mod tests;

#[cfg(not(target_arch = "spirv"))]
mod utils;

#[cfg(not(target_arch = "spirv"))]
mod cpu {

    use crate::assert_eq_with_tol;
    use crate::utils;
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

    const SHADER: &[u8] = include_bytes!(env!("tsdistances.spv"));

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
                        ShaderModule::new(device, ShaderModuleCreateInfo::new(shader)).map_err(
                            |e| {
                                eprintln!("Failed to load shader module: {:?}", e);
                                e
                            },
                        )
                    }
                }
            }
            let cs = cs::load(device.clone(), shader)
                .unwrap()
                .entry_point("gpu::erp")
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
                println!("Diagonal[{}]: {}", n, data_buffer_content[n as usize]);
            }
            assert_eq_with_tol!(
                data_buffer_content[n as usize],
                (input_a[n as usize] * input_b[n as usize]).log2()
            );
        }
        println!("Success");
        Ok(())
    }
}
