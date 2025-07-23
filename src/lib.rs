#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(unexpected_cfgs)]

pub mod kernels;
pub type Float = f32;
// pub type Float = f64;

#[cfg(not(target_arch = "spirv"))]
mod shader_load;
#[cfg(not(target_arch = "spirv"))]
pub mod utils;
#[cfg(not(target_arch = "spirv"))]
pub mod warps;

#[cfg(not(target_arch = "spirv"))]
pub mod cpu {
    use crate::kernels::adtw_distance::cpu::ADTWImpl;
    use crate::kernels::dtw_distance::cpu::DTWImpl;
    use crate::kernels::erp_distance::cpu::ERPImpl;
    use crate::kernels::lcss_distance::cpu::LCSSImpl;
    use crate::kernels::msm_distance::cpu::MSMImpl;
    use crate::kernels::twe_distance::cpu::TWEImpl;
    use crate::kernels::wdtw_distance::cpu::WDTWImpl;
    use crate::warps::diamond_partitioning_gpu;
    use crate::warps::GpuBatchMode;
    use crate::Float;
    use crate::utils::SubBuffersAllocator;
    use std::sync::Arc;

    use vulkano::buffer::allocator::SubbufferAllocator;
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
        sa: SubBuffersAllocator,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
        gap_penalty: Float,
    ) -> M::ReturnType {
        diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            sa,
            ERPImpl {
                gap_penalty: gap_penalty as Float,
            },
            a,
            b,
            Float::INFINITY,
        )
    }

    pub fn lcss<'a, M: GpuBatchMode>(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        sa: SubBuffersAllocator,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
        epsilon: Float,
    ) -> M::ReturnType {
        let similarity = diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            sa,
            LCSSImpl {
                epsilon: epsilon as Float,
            },
            a,
            b,
            0.0,
        );
        let min_len =
            M::get_sample_length(&a.clone()).min(M::get_sample_length(&b.clone())) as Float;
        M::apply_fn(similarity, |s| 1.0 - s / min_len)
    }

    pub fn dtw<'a, M: GpuBatchMode>(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        sa: SubBuffersAllocator,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
    ) -> M::ReturnType {
        let res = diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            sa,
            DTWImpl {},
            a,
            b,
            Float::INFINITY,
        );
        res
    }

    pub fn wdtw<'a, M: GpuBatchMode>(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        sa: SubBuffersAllocator,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
        weights: &[Float],
    ) -> M::ReturnType {
        let weights = weights.iter().map(|x| *x as Float).collect::<Vec<Float>>();

        diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            sa,
            WDTWImpl { weights: weights },
            a,
            b,
            Float::INFINITY,
        )
    }

    pub fn msm<'a, M: GpuBatchMode>(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        sa: SubBuffersAllocator,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
    ) -> M::ReturnType {
        diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            sa,
            MSMImpl {},
            a,
            b,
            Float::INFINITY,
        )
    }

    pub fn twe<'a, M: GpuBatchMode>(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        sa: SubBuffersAllocator,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
        stiffness: Float,
        penalty: Float,
    ) -> M::ReturnType {
        diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            sa,
            TWEImpl {
                stiffness: stiffness as Float,
                penalty: penalty as Float,
            },
            a,
            b,
            Float::INFINITY,
        )
    }

    pub fn adtw<'a, M: GpuBatchMode>(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        sa: SubBuffersAllocator,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
        w: Float,
    ) -> M::ReturnType {
        diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            sa,
            ADTWImpl { w: w as Float },
            a,
            b,
            Float::INFINITY,
        )
    }
}
