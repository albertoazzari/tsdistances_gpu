use crate::Float;
pub struct GpuMatrix<'a> {
    diagonal: &'a mut [Float],
    diagonal_offset: usize,
    mask: usize,
}

impl GpuMatrix<'_> {
    #[inline(always)]
    fn get_diagonal_cell(&self, _diag_row: usize, diag_offset: isize) -> Float {
        self.diagonal[self.diagonal_offset + (diag_offset as usize & self.mask)]
    }

    #[inline(always)]
    fn set_diagonal_cell(&mut self, _diag_row: usize, diag_offset: isize, value: Float) {
        self.diagonal[self.diagonal_offset + (diag_offset as usize & self.mask)] = value;
    }
}

macro_rules! warp_kernel_spec {
    ($(
        fn $name:ident[$impl_struct:ident](
            $a:ident[$a_offset:ident],
            $b:ident[$b_offset:ident],
            $i:ident,
            $j:ident,
            $x:ident,
            $y:ident,
            $z:ident,
            [$($param1:ident: $ty1:ty)?],
            [$($param2:ident: $ty2:ty)?],
            [$($param3:ident: $ty3:ty)?],
            [$($param4:ident: $ty4:ty)?],
            [$($vec5:ident: $ty5:ty)?]
        ) $body:block
    )*) => {
        $(
            pub mod $name {
                use crate::Float;
                #[cfg(not(target_arch = "spirv"))]
                pub mod cpu {
                    use std::sync::Arc;
                    use vulkano::buffer::Subbuffer;
                    use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
                    use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
                    use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
                    use vulkano::device::Device;
                    use crate::{kernels::kernel_trait::{GpuKernelImpl, BatchInfo}, Float};
                    use vulkano::pipeline::{Pipeline, PipelineBindPoint};

                    pub struct $impl_struct {
                        $(pub $param1: $ty1,)?
                        $(pub $param2: $ty2,)?
                        $(pub $param3: $ty3,)?
                        $(pub $param4: $ty4,)?
                        $(pub $vec5:  Vec<$ty5>,)?
                    }

                    pub struct KernelParams {
                        $(pub $vec5:  Subbuffer<[$ty5]>)?
                    }

                    impl GpuKernelImpl for $impl_struct {

                        type KernelParams = KernelParams;

                        fn build_kernel_params(
                            &self,
                            _allocator: Arc<vulkano::buffer::allocator::SubbufferAllocator>,
                            max_work_group_size: usize,
                        ) -> Self::KernelParams {
                            KernelParams {
                                $($vec5: crate::utils::move_gpu(&self.$vec5, &_allocator, max_work_group_size))?
                            }
                        }

                        fn dispatch(
                            &self,
                            device: Arc<Device>,
                            dsa: Arc<StandardDescriptorSetAllocator>,
                            builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                            first_coord: i64,
                            row: u64,
                            diamonds_count: u64,
                            a_start: u64,
                            b_start: u64,
                            a_len: u64,
                            b_len: u64,
                            max_subgroup_threads: u64,
                            a: &Subbuffer<[Float]>,
                            b: &Subbuffer<[Float]>,
                            diagonal: &mut Subbuffer<[Float]>,
                            batch_info: Option<BatchInfo>,
                            _kernel_params: &Self::KernelParams,
                        ) {
                            let (kernel_name, padded_a_len, padded_b_len, threads_count, a_count, b_count, diagonal_stride) = match batch_info {
                                Some(batch_info) =>
                                {
                                    (
                                        concat!("kernels::", stringify!($name), "::batch_call"),
                                        batch_info.padded_a_len,
                                        batch_info.padded_b_len,
                                        (batch_info.a_count * batch_info.b_count * diamonds_count * max_subgroup_threads) as u32,
                                        batch_info.a_count,
                                        batch_info.b_count,
                                        batch_info.diagonal_stride,
                                    )
                                },
                                None => {
                                    (
                                        concat!("kernels::", stringify!($name), "::single_call"),
                                        0u64,
                                        0u64,
                                        (diamonds_count * max_subgroup_threads) as u32,
                                        1,
                                        1,
                                        (diamonds_count * max_subgroup_threads) as u64,
                                    )
                                },
                            };

                            let pipeline = crate::shader_load::get_shader_entry_pipeline(device.clone(), kernel_name);
                            let layout = &pipeline.layout().set_layouts()[0];

                            let set = DescriptorSet::new(
                                dsa.clone(),
                                layout.clone(),
                                [
                                    WriteDescriptorSet::buffer(0, diagonal.clone()),
                                    WriteDescriptorSet::buffer(1, a.clone()),
                                    WriteDescriptorSet::buffer(2, b.clone()),
                                    $(WriteDescriptorSet::buffer(3, _kernel_params.$vec5.clone()),)?
                                ],
                                [],
                            )
                            .unwrap();

                            let kernel_constants = super::KernelConstants {
                                    first_coord,
                                    row,
                                    diamonds_count,
                                    a_start,
                                    b_start,
                                    a_len,
                                    b_len,
                                    a_count,
                                    b_count,
                                    diagonal_stride,
                                    max_subgroup_threads,
                                    $(param1: self.$param1,)?
                                    $(param2: self.$param2,)?
                                    $(param3: self.$param3,)?
                                    $(param4: self.$param4,)?
                                    padded_a_len,
                                    padded_b_len,
                            };

                            builder
                                .bind_pipeline_compute(pipeline.clone())
                                .unwrap()
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Compute,
                                    pipeline.layout().clone(),
                                    0,
                                    set,
                                )
                                .unwrap()
                                .push_constants(
                                    pipeline.layout().clone(),
                                    0,
                                    kernel_constants
                                )
                                .unwrap();

                            let max_threads_x = device
                                .physical_device()
                                .properties()
                                .max_compute_work_group_size[0];
                            
                            // print kernel_name, padded_a_len, padded_b_len, threads_count, a_count, b_count, diagonal_stride
                            println!(
                                "Dispatching kernel: {}, padded_a_len: {}, padded_b_len: {}, threads_count: {}, a_count: {}, b_count: {}, diagonal_stride: {}, max_threads_x: {}, threads_count: {}",
                                kernel_name, padded_a_len, padded_b_len, threads_count, a_count, b_count, diagonal_stride, max_threads_x, threads_count
                            );

                            unsafe { builder.dispatch([threads_count.div_ceil(max_threads_x), 1u32, 1u32]) }.unwrap();
                        }
                    }
                }

                #[derive(Clone, Copy, bytemuck::AnyBitPattern)]
                #[repr(C)]
                #[allow(unused)]
                pub struct KernelConstants {
                    first_coord: i64,
                    row: u64,
                    diamonds_count: u64,
                    a_start: u64,
                    b_start: u64,
                    a_len: u64,
                    b_len: u64,
                    a_count: u64,
                    b_count: u64,
                    diagonal_stride: u64,
                    max_subgroup_threads: u64,
                    $(param1: $ty1,)?
                    $(param2: $ty2,)?
                    $(param3: $ty3,)?
                    $(param4: $ty4,)?
                    padded_a_len: u64,
                    padded_b_len: u64,
                }

                #[cfg(target_arch = "spirv")]
                use spirv_std::{glam::UVec3, spirv, num_traits::Float as SpirvFloat};

                #[cfg(target_arch = "spirv")]
                #[inline(always)]
                fn warp_kernel_inner(
                    mut matrix: super::GpuMatrix,
                    d_offset: u64,
                    a_start: u64,
                    b_start: u64,
                    diag_mid: i64,
                    diag_count: u64,
                    warp: u64,
                    max_subgroup_threads: u64,
                    $a: &[Float],
                    $b: &[Float],
                    $a_offset: usize,
                    $b_offset: usize,
                    $($param1: $ty1,)?
                    $($param2: $ty2,)?
                    $($param3: $ty3,)?
                    $($param4: $ty4,)?
                    $($vec5: &[$ty5],)?
                ) {
                    let mut i = a_start;
                    let mut j = b_start;
                    let mut s = diag_mid;
                    let mut e = diag_mid;

                    for d in 2..diag_count {
                        let k = (warp * 2) as i64 + s;
                        if k <= e {
                            let $i = i - warp;
                            let $j = j + warp;

                            let $x = matrix.get_diagonal_cell((d_offset + d - 1) as usize, (k - 1) as isize);
                            let $y = matrix.get_diagonal_cell((d_offset + d - 2) as usize, k as isize);
                            let $z = matrix.get_diagonal_cell((d_offset + d - 1) as usize, (k + 1) as isize);


                            let value = {
                                $body
                            };

                            matrix.set_diagonal_cell((d_offset + d) as usize, k as isize, value);
                        }
                        // Warp synchronize
                        unsafe { spirv_std::arch::workgroup_memory_barrier_with_group_sync() };

                        if d <= max_subgroup_threads {
                            i += 1;
                            s -= 1;
                            e += 1;
                        } else {
                            j += 1;
                            s += 1;
                            e -= 1;
                        }
                    }
                }

                #[cfg(target_arch = "spirv")]
                #[inline(always)]
                fn warp_kernel(
                    global_id: u64,
                    first_coord: i64,
                    row: u64,
                    diamonds_count: u64,
                    a_start: u64,
                    b_start: u64,
                    a_len: u64,
                    b_len: u64,
                    max_subgroup_threads: u64,
                    diagonal: &mut [Float],
                    diagonal_offset: u64,
                    diagonal_len: u64,
                    $a: &[Float],
                    $b: &[Float],
                    $a_offset: usize,
                    $b_offset: usize,
                    $($param1: $ty1,)?
                    $($param2: $ty2,)?
                    $($param3: $ty3,)?
                    $($param4: $ty4,)?
                    $($vec5: &[$ty5],)?
                ) {
                    let warp_id: u64 = global_id % max_subgroup_threads;
                    let diamond_id = global_id / max_subgroup_threads;

                    if diamond_id >= diamonds_count {
                        return;
                    }

                    let diag_start = first_coord + ((diamond_id * max_subgroup_threads) as i64) * 2;
                    let d_a_start = a_start - diamond_id * max_subgroup_threads;
                    let d_b_start = b_start + diamond_id * max_subgroup_threads;

                    let alen = a_len - d_a_start;
                    let blen = b_len - d_b_start;

                    let matrix = super::GpuMatrix {
                        diagonal,
                        diagonal_offset: diagonal_offset as usize,
                        mask: diagonal_len as usize - 1,
                    };

                    warp_kernel_inner(
                        matrix,
                        row * max_subgroup_threads,
                        d_a_start,
                        d_b_start,
                        diag_start + (max_subgroup_threads as i64),
                        (max_subgroup_threads * 2 + 1).min(alen + blen + 1),
                        warp_id,
                        max_subgroup_threads,
                        $a,
                        $b,
                        $a_offset,
                        $b_offset,
                        $($param1,)?
                        $($param2,)?
                        $($param3,)?
                        $($param4,)?
                        $($vec5,)?
                    );
                }

                #[cfg(target_arch = "spirv")]
                #[spirv(compute(threads(1)))]
                pub fn single_call(
                    #[spirv(global_invocation_id)] global_id: UVec3,
                    #[spirv(push_constant)] constants: &KernelConstants,
                    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] diagonal: &mut [Float],
                    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] $a: &[Float],
                    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] $b: &[Float],
                    $(#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] vec5: &[$ty5],)?
                ) {

                    $(let $param1 = constants.param1;)?
                    $(let $param2 = constants.param2;)?
                    $(let $param3 = constants.param3;)?
                    $(let $param4 = constants.param4;)?
                    $(let $vec5 = vec5;)?

                    let $a_offset = 0;
                    let $b_offset = 0;

                    let global_id = global_id.x as u64;
                    warp_kernel(
                        global_id,
                        constants.first_coord,
                        constants.row,
                        constants.diamonds_count,
                        constants.a_start,
                        constants.b_start,
                        constants.a_len,
                        constants.b_len,
                        constants.max_subgroup_threads,
                        diagonal,
                        0,
                        diagonal.len() as u64,
                        $a,
                        $b,
                        $a_offset,
                        $b_offset,
                        $($param1,)?
                        $($param2,)?
                        $($param3,)?
                        $($param4,)?
                        $($vec5,)?
                    );
                }
                #[cfg(target_arch = "spirv")]
                #[spirv(compute(threads(1)))]
                pub fn batch_call(
                    #[spirv(global_invocation_id)] global_id: UVec3,
                    #[spirv(push_constant)] constants: &KernelConstants,
                    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] diagonal: &mut [Float],
                    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] $a: &[Float],
                    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] $b: &[Float],
                    $(#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] vec5: &[$ty5],)?
                ) {

                    $(let $param1 = constants.param1;)?
                    $(let $param2 = constants.param2;)?
                    $(let $param3 = constants.param3;)?
                    $(let $param4 = constants.param4;)?
                    $(let $vec5 = vec5;)?


                    let global_id = global_id.x as u64;
                    let threads_stride = constants.diamonds_count * constants.max_subgroup_threads;

                    let pair_index = global_id / threads_stride;
                    let instance_id = global_id % threads_stride;

                    let a_index = pair_index / constants.b_count as u64;
                    let b_index = pair_index % constants.b_count as u64;

                    let diagonal_offset = pair_index * constants.diagonal_stride;

                    let $a_offset = a_index as usize * constants.padded_a_len as usize;
                    let $b_offset = b_index as usize * constants.padded_b_len as usize;

                    warp_kernel(
                        instance_id,
                        constants.first_coord,
                        constants.row,
                        constants.diamonds_count,
                        constants.a_start,
                        constants.b_start,
                        constants.a_len,
                        constants.b_len,
                        constants.max_subgroup_threads,
                        diagonal,
                        diagonal_offset,
                        constants.diagonal_stride,
                        $a,
                        $b,
                        $a_offset,
                        $b_offset,
                        $($param1,)?
                        $($param2,)?
                        $($param3,)?
                        $($param4,)?
                        $($vec5,)?
                    );
                }
            }
        )*
    };
}

#[cfg(not(target_arch = "spirv"))]
pub mod kernel_trait {
    use std::sync::Arc;
    use vulkano::buffer::Subbuffer;
    use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
    use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
    use vulkano::device::Device;

    use crate::Float;

    pub struct BatchInfo {
        pub padded_a_len: u64,
        pub padded_b_len: u64,
        pub a_count: u64,
        pub b_count: u64,
        pub diagonal_stride: u64,
    }

    pub trait GpuKernelImpl {
        type KernelParams;

        fn build_kernel_params(
            &self,
            allocator: Arc<vulkano::buffer::allocator::SubbufferAllocator>,
            max_work_group_size: usize,
        ) -> Self::KernelParams;

        fn dispatch(
            &self,
            device: Arc<Device>,
            stsa: Arc<StandardDescriptorSetAllocator>,
            builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
            first_coord: i64,
            row: u64,
            diamonds_count: u64,
            a_start: u64,
            b_start: u64,
            a_len: u64,
            b_len: u64,
            max_subgroup_threads: u64,
            a: &Subbuffer<[Float]>,
            b: &Subbuffer<[Float]>,
            diagonal: &mut Subbuffer<[Float]>,
            batch_info: Option<BatchInfo>,
            kernel_params: &Self::KernelParams,
        );
    }
}

const MSM_C: Float = 1.0;
#[inline(always)]
pub fn msm_cost_function(x: Float, y: Float, z: Float) -> Float {
    MSM_C + (y.min(z) - x).max(x - y.max(z)).max(0.0)
}

warp_kernel_spec! {
    fn erp_distance[ERPImpl](a[a_offset], b[b_offset], i, j, x, y, z, [gap_penalty: Float], [], [], [], []) {
        (y + (a[a_offset + i as usize] - b[b_offset + j as usize]).abs())
        .min((z + (a[a_offset + i as usize] - gap_penalty).abs()).min(x + (b[b_offset + j as usize] - gap_penalty).abs()))
    }
    fn lcss_distance[LCSSImpl](a[a_offset], b[b_offset], i, j, x, y, z, [epsilon: Float], [], [], [], []) {
        let dist = (a[a_offset + i as usize] - b[b_offset + j as usize]).abs();
        (dist <= epsilon) as i32 as Float * (y + 1.0) + (dist > epsilon) as i32 as Float * x.max(z)
    }
    fn dtw_distance[DTWImpl](a[a_offset], b[b_offset], i, j, x, y, z, [], [], [], [], []) {
        let dist = (a[a_offset + i as usize] - b[b_offset + j as usize]).powi(2);
        dist + z.min(x.min(y))
    }
    fn wdtw_distance[WDTWImpl](a[a_offset], b[b_offset], i, j, x, y, z, [], [], [], [], [weights: Float]) {
        let dist = (a[a_offset + i as usize] - b[b_offset + j as usize]).powi(2) * weights[(i as i32 - j as i32).abs() as usize];
        dist + x.min(y.min(z))
    }
    fn msm_distance[MSMImpl](a[a_offset], b[b_offset], i, j, x, y, z, [], [], [], [], []) {
        (y + (a[a_offset + i as usize] - b[b_offset + j as usize]).abs())
        .min(
            z + super::msm_cost_function(a[a_offset + i as usize], if i == 0 {0.0} else {a[a_offset + i as usize - 1]}, b[b_offset + j as usize]),
        )
        .min(
            x + super::msm_cost_function(b[b_offset + j as usize], a[a_offset + i as usize], if j == 0 {0.0} else {b[b_offset + j as usize - 1]}),
        )
    }
    fn twe_distance[TWEImpl](a[a_offset], b[b_offset], i, j, x, y, z, [stiffness: Float], [penalty: Float], [], [], []) {
        let delete_addition = penalty + stiffness;
        // deletion in a
        let del_a =
        z + (if i == 0 {0.0} else {a[a_offset + i as usize - 1]} - a[a_offset + i as usize]).abs() + delete_addition;

        // deletion in b
        let del_b =
            x + (if j == 0 {0.0} else {b[b_offset + j as usize - 1]} - b[b_offset + j as usize]).abs() + delete_addition;

        // match
        let match_current = (a[a_offset + i as usize] - b[b_offset + j as usize]).abs();
        let match_previous = (if i == 0 {0.0} else {a[a_offset + i as usize - 1]}
            - if j == 0 {0.0} else {b[b_offset + j as usize - 1]})
        .abs();
        let match_a_b = y
            + match_current
            + match_previous
            + stiffness * (2.0 * (i as isize - j as isize).abs() as Float);

        del_a.min(del_b.min(match_a_b))
    }
    fn adtw_distance[ADTWImpl](a[a_offset], b[b_offset], i, j, x, y, z, [w: Float], [], [], [], []) {
        let dist = (a[a_offset + i as usize] - b[b_offset + j as usize]).powi(2);
                dist + (z + w).min((x + w).min(y))
    }
}
