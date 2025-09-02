#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(unexpected_cfgs)]

pub mod kernels;

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
    use crate::utils::SubBuffersAllocator;
    use crate::warps::GpuBatchMode;
    use crate::warps::diamond_partitioning_gpu;
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
        sa: SubBuffersAllocator,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
        gap_penalty: f32,
    ) -> M::ReturnType {
        diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            sa,
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
        sa: SubBuffersAllocator,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
        epsilon: f32,
    ) -> M::ReturnType {
        let similarity = diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            sa,
            LCSSImpl {
                epsilon,
            },
            a,
            b,
            0.0,
        );
        let min_len =
            M::get_sample_length(&a.clone()).min(M::get_sample_length(&b.clone())) as f32;
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
        // let start_time = std::time::Instant::now();
        let res = diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            sa,
            DTWImpl {},
            a,
            b,
            f32::INFINITY,
        );
        // println!(
        //     "GPU - DTW distance computed in {} ms",
        //     start_time.elapsed().as_millis()
        // );
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
        weights: &[f32],
    ) -> M::ReturnType {

        diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            sa,
            WDTWImpl { weights: weights.to_vec() },
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
            f32::INFINITY,
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
        stiffness: f32,
        penalty: f32,
    ) -> M::ReturnType {
        diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            sa,
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
        sa: SubBuffersAllocator,
        a: M::InputType<'a>,
        b: M::InputType<'a>,
        w: f32,
    ) -> M::ReturnType {
        diamond_partitioning_gpu::<_, M>(
            device,
            queue,
            sba,
            dsa,
            sa,
            ADTWImpl { w: w as f32 },
            a,
            b,
            f32::INFINITY,
        )
    }
}
