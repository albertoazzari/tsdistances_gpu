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
    use crate::kernels::adtw_distance::cpu::ADTWImpl;
    use crate::kernels::dtw_distance::cpu::DTWImpl;
    use crate::kernels::msm_distance::cpu::MSMImpl;
    use crate::kernels::twe_distance::cpu::TWEImpl;
    use crate::kernels::wdtw_distance::cpu::WDTWImpl;
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
        let similarity = diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            LCSSImpl {
                epsilon: epsilon as f32,
            },
            a,
            b,
            0.0,
        );
        let min_len = M::get_sample_length(&a.clone()).min(M::get_sample_length(&b.clone())) as f64;
        M::apply_fn(similarity, |s| 1.0 - s / min_len)
    }

    pub fn dtw<'a, M: GpuBatchMode>(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
    ) -> M::ReturnType {
        diamond_partitioning_gpu::<_, M>(device, queue, sba, dsa, DTWImpl {}, a, b, f32::INFINITY)
    }

    pub fn wdtw<'a, M: GpuBatchMode>(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
        weights: &[f64],
    ) -> M::ReturnType {
        let weights = weights.iter().map(|x| *x as f32).collect::<Vec<f32>>();

        diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            WDTWImpl { weights: weights},
            a,
            b,
            f32::INFINITY,
        )
    }

    pub fn msm<'a, M: GpuBatchMode>(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
    ) -> M::ReturnType {
        diamond_partitioning_gpu::<_, M>(device, queue, sba, dsa, MSMImpl {}, a, b, f32::INFINITY)
    }

    pub fn twe<'a, M: GpuBatchMode>(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
        stiffness: f64,
        penalty: f64,
    ) -> M::ReturnType {
        diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            TWEImpl {
                stiffness: stiffness as f32,
                penalty: penalty as f32,
            },
            a,
            b,
            f32::INFINITY,
        )
    }

    pub fn adtw<'a, M: GpuBatchMode>(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
        w: f64,
    ) -> M::ReturnType {
        diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            ADTWImpl { w: w as f32 },
            a,
            b,
            f32::INFINITY,
        )
    }
}
