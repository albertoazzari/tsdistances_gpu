use std::sync::Arc;

use crate::{
    kernels::kernel_trait::{BatchInfo, GpuKernelImpl},
    utils::{move_cpu, move_gpu},
};
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{Device, Queue},
    sync::GpuFuture,
};

pub trait GpuBatchMode {
    const IS_BATCH: bool;

    type ReturnType;
    type InputType<'a>;

    fn get_samples_count(input: &Self::InputType<'_>) -> usize;
    fn new_return(alen: usize, blen: usize) -> Self::ReturnType;
    fn set_return(ret: &mut Self::ReturnType, i: usize, j: usize, value: f32);
    fn build_padded(input: &Self::InputType<'_>, pad_stride: usize) -> Vec<f32>;
    fn get_sample_length(input: &Self::InputType<'_>) -> usize;
    fn get_padded_len(sample_length: usize, pad_stride: usize) -> usize {
        next_multiple_of_n(sample_length, pad_stride)
    }
    fn get_subslice<'a>(
        input: &Self::InputType<'a>,
        start: usize,
        len: usize,
    ) -> Self::InputType<'a>;
    fn apply_fn(ret: Self::ReturnType, func: impl Fn(f64) -> f64) -> Self::ReturnType;
    fn join_results(results: Vec<Self::ReturnType>) -> Self::ReturnType;
}

pub struct SingleBatchMode;
impl GpuBatchMode for SingleBatchMode {
    const IS_BATCH: bool = false;

    type ReturnType = f64;
    type InputType<'a> = &'a [f64];

    fn get_samples_count(_input: &Self::InputType<'_>) -> usize {
        1
    }

    fn new_return(_: usize, _: usize) -> Self::ReturnType {
        0.0
    }

    fn set_return(ret: &mut Self::ReturnType, _: usize, _: usize, value: f32) {
        *ret = value as f64;
    }

    fn build_padded(input: &Self::InputType<'_>, pad_stride: usize) -> Vec<f32> {
        let padded_len = Self::get_padded_len(Self::get_sample_length(input), pad_stride);
        let mut padded = vec![0.0; padded_len];
        for (padded, input) in padded.iter_mut().zip(input.iter()) {
            *padded = *input as f32;
        }

        padded
    }

    fn get_sample_length(input: &Self::InputType<'_>) -> usize {
        input.len()
    }

    fn apply_fn(ret: Self::ReturnType, func: impl Fn(f64) -> f64) -> Self::ReturnType {
        func(ret)
    }

    fn get_subslice<'a>(input: &Self::InputType<'a>, _: usize, _: usize) -> Self::InputType<'a> {
        &input
    }

    fn join_results(results: Vec<Self::ReturnType>) -> Self::ReturnType {
        results[0]
    }
}

pub struct MultiBatchMode;

impl GpuBatchMode for MultiBatchMode {
    const IS_BATCH: bool = true;

    type ReturnType = Vec<Vec<f64>>;

    type InputType<'a> = &'a [Vec<f64>];

    fn get_samples_count(input: &Self::InputType<'_>) -> usize {
        input.len()
    }

    fn new_return(alen: usize, blen: usize) -> Self::ReturnType {
        vec![vec![0.0; blen]; alen]
    }

    fn set_return(ret: &mut Self::ReturnType, i: usize, j: usize, value: f32) {
        ret[i][j] = value as f64;
    }

    fn build_padded(input: &Self::InputType<'_>, pad_stride: usize) -> Vec<f32> {
        let single_padded_len = Self::get_padded_len(Self::get_sample_length(input), pad_stride);
        let mut padded = vec![0.0; input.len() * single_padded_len];
        for i in 0..input.len() {
            for j in 0..input[i].len() {
                padded[i * single_padded_len + j] = input[i][j] as f32;
            }
        }

        padded
    }

    fn get_sample_length(input: &Self::InputType<'_>) -> usize {
        input.first().map_or(0, |x| x.len())
    }

    fn apply_fn(mut ret: Self::ReturnType, func: impl Fn(f64) -> f64) -> Self::ReturnType {
        for i in 0..ret.len() {
            for j in 0..ret[i].len() {
                ret[i][j] = func(ret[i][j]);
            }
        }
        ret
    }

    fn get_subslice<'a>(
        input: &Self::InputType<'a>,
        start: usize,
        len: usize,
    ) -> Self::InputType<'a> {
        // println!("GINO start: {}, len: {}, a_len: {}", start, len, input.len());
        &input[start..(start + len)]
    }

    fn join_results(results: Vec<Self::ReturnType>) -> Self::ReturnType {
        results.into_iter().flatten().collect()
    }
}

fn compute_diag_len<M: GpuBatchMode>(sample_length: usize, pad_stride: usize) -> usize {
    2 * (M::get_padded_len(sample_length, pad_stride) + 1).next_power_of_two()
}

pub fn diamond_partitioning_gpu<'a, G: GpuKernelImpl, M: GpuBatchMode>(
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    params: G,
    a: M::InputType<'a>,
    b: M::InputType<'a>,
    init_val: f32,
) -> M::ReturnType {
    let (a, b) = if M::get_sample_length(&a) > M::get_sample_length(&b) {
        (b, a)
    } else {
        (a, b)
    };

    let max_subgroup_threads: usize = device
        .physical_device()
        .properties()
        .max_subgroup_size
        .unwrap() as usize;

    let a_sample_length = M::get_sample_length(&a);

    let diag_len = compute_diag_len::<M>(a_sample_length, max_subgroup_threads);

    // Limit the maximum size of a buffer to 2gb
    // a_count * b_count * diag_len should not exceed 2gb / size_of::<f32>())
    let max_buffer_size = device
        .physical_device()
        .properties()
        .max_storage_buffer_range as usize;

    // M::get_sample_length(&b) * diag_len
    let max_a_batch_size = max_buffer_size / (diag_len * M::get_samples_count(&b));

    if max_a_batch_size == 0 {
        println!("WARNING: The input is too large to be processed by the GPU, you could experience a runtime crash.");
    }

    let a_batch_size = max_a_batch_size.min(M::get_samples_count(&a)).max(1);
    let mut start = 0;
    let a_len = M::get_samples_count(&a);
    let mut distances = Vec::new();

    while start < a_len {
        let len = a_batch_size.min(a_len - start);
        // println!("start: {}, len: {}, a_len: {}", start, len, a_len);
        let a = M::get_subslice(&a, start, len);
        let a_padded = M::build_padded(&a, max_subgroup_threads);
        let b_padded = M::build_padded(&b, max_subgroup_threads);

        distances.push(diamond_partitioning_gpu_::<G, M>(
            device.clone(),
            queue.clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
            &params,
            max_subgroup_threads,
            M::get_sample_length(&a),
            M::get_sample_length(&b),
            a_padded,
            b_padded,
            init_val,
            M::IS_BATCH,
        ));
        start += len;
    }
    M::join_results(distances)
}

#[inline(always)]
fn diamond_partitioning_gpu_<G: GpuKernelImpl, M: GpuBatchMode>(
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    params: &G,
    max_subgroup_threads: usize,
    a_sample_len: usize,
    b_sample_len: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    init_val: f32,
    is_batch: bool,
) -> M::ReturnType {
    let padded_a_len = M::get_padded_len(a_sample_len, max_subgroup_threads);
    let padded_b_len = M::get_padded_len(b_sample_len, max_subgroup_threads);

    let a_count = a.len() / padded_a_len;
    let b_count = b.len() / padded_b_len;

    let diag_len = compute_diag_len::<M>(a_sample_len, max_subgroup_threads);

    let mut diagonal = vec![init_val; a_count * b_count * diag_len];

    for i in 0..(a_count * b_count) {
        diagonal[i * diag_len] = 0.0;
    }

    let a_diamonds = padded_a_len.div_ceil(max_subgroup_threads);
    let b_diamonds = padded_b_len.div_ceil(max_subgroup_threads);
    let rows_count = (padded_a_len + padded_b_len).div_ceil(max_subgroup_threads) - 1;

    let mut diamonds_count = 1;
    let mut first_coord = -(max_subgroup_threads as isize);
    let mut a_start = 0;
    let mut b_start = 0;

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let a_gpu = move_gpu(&a, &mut builder, device.clone());
    let b_gpu = move_gpu(&a, &mut builder, device.clone());
    let mut diagonal = move_gpu(&diagonal, &mut builder, device.clone());

    // Number of kernel calls
    for i in 0..rows_count {
        params.dispatch(
            device.clone(),
            descriptor_set_allocator.clone(),
            &mut builder,
            first_coord as i64,
            i as u64,
            diamonds_count as u64,
            a_start as u64,
            b_start as u64,
            a_sample_len as u64,
            b_sample_len as u64,
            max_subgroup_threads as u64,
            &a_gpu,
            &b_gpu,
            &mut diagonal,
            if is_batch {
                Some(BatchInfo {
                    padded_a_len: padded_a_len as u64,
                    padded_b_len: padded_b_len as u64,
                })
            } else {
                None
            },
        );

        if i < (a_diamonds - 1) {
            diamonds_count += 1;
            first_coord -= max_subgroup_threads as isize;
            a_start += max_subgroup_threads;
        } else if i < (b_diamonds - 1) {
            first_coord += max_subgroup_threads as isize;
            b_start += max_subgroup_threads;
        } else {
            diamonds_count -= 1;
            first_coord += max_subgroup_threads as isize;
            b_start += max_subgroup_threads;
        }
    }

    fn index_mat_to_diag(i: usize, j: usize) -> (usize, isize) {
        (i + j, (j as isize) - (i as isize))
    }

    let (_, cx) = index_mat_to_diag(a_sample_len, b_sample_len);

    let diagonal = move_cpu(diagonal, &mut builder, device.clone());
    let command_buffer = builder.build().unwrap();
    let future = vulkano::sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();

    let mut res = M::new_return(a_count, b_count);

    let diagonal = diagonal.read().unwrap();

    for i in 0..a_count {
        for j in 0..b_count {
            let diag_offset = (i * b_count + j) * diag_len;
            M::set_return(
                &mut res,
                i,
                j,
                diagonal[diag_offset + ((cx as usize) & (diag_len - 1))],
            );
        }
    }

    res
}

fn next_multiple_of_n(x: usize, n: usize) -> usize {
    (x + n - 1) / n * n
}
