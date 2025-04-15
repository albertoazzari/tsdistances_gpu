#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(unexpected_cfgs)]

pub mod kernels;

#[cfg(not(target_arch = "spirv"))]
mod shader_load;
#[cfg(not(target_arch = "spirv"))]
mod tests;
#[cfg(not(target_arch = "spirv"))]
mod utils;
#[cfg(not(target_arch = "spirv"))]
mod warps;


#[cfg(not(target_arch = "spirv"))]
mod cpu {

    use crate::assert_eq_with_tol;
    use crate::kernels::erp_distance::cpu::ERPImpl;
    use crate::utils;
    use crate::warps::diamond_partitioning_gpu;
    use crate::warps::GpuBatchMode;
    use std::{fmt::Error, sync::Arc};

    use vulkano::device::Queue;
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

    // const SHADER: &[u8] = include_bytes!(env!("tsdistances.spv"));

    pub fn erp<'a, M: GpuBatchMode>(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
        gap_penalty: f64,
    ) -> M::ReturnType {
    
        diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            ERPImpl {
                gap_penalty: gap_penalty as f32,
            },
            a,
            b,
            f32::INFINITY,
        )
    }
}
