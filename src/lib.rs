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
    use crate::kernels::{erp_distance::cpu::ERPImpl, lcss_distance::cpu::LCSSImpl};
    use crate::warps::diamond_partitioning_gpu;
    use crate::warps::GpuBatchMode;
    use std::sync::Arc;

    use vulkano::device::Queue;
    use vulkano::{
        command_buffer::allocator::StandardCommandBufferAllocator,
        descriptor_set::allocator::StandardDescriptorSetAllocator, device::Device,
    };

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

    pub fn lcss<'a, M: GpuBatchMode>(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
        epsilon: f64,
    ) -> M::ReturnType {
        diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            LCSSImpl {
                epsilon: epsilon as f32,
            },
            a,
            b,
            f32::INFINITY,
        )
    }
}
