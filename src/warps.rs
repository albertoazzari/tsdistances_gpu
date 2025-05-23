use std::sync::Arc;

use crate::{
    kernels::kernel_trait::{BatchInfo, GpuKernelImpl},
    utils::{move_cpu, move_gpu},
    Precision,
};
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{Device, Queue},
    memory::allocator::StandardMemoryAllocator,
    sync::GpuFuture,
};

pub trait GpuBatchMode {
    const IS_BATCH: bool;

    type ReturnType;
    type InputType<'a>: Copy;

    fn get_samples_count(input: &Self::InputType<'_>) -> usize;
    fn new_return(alen: usize, blen: usize) -> Self::ReturnType;
    fn set_return(ret: &mut Self::ReturnType, i: usize, j: usize, value: Precision);
    fn build_padded(input: &Self::InputType<'_>, pad_stride: usize) -> Vec<Precision>;
    fn get_sample_length(input: &Self::InputType<'_>) -> usize;
    fn get_padded_len(sample_length: usize, pad_stride: usize) -> usize {
        next_multiple_of_n(sample_length, pad_stride)
    }
    fn get_subslice<'a>(
        input: &Self::InputType<'a>,
        start: usize,
        len: usize,
    ) -> Self::InputType<'a>;
    fn apply_fn(ret: Self::ReturnType, func: impl Fn(Precision) -> Precision) -> Self::ReturnType;
    fn join_results(results: Vec<Self::ReturnType>) -> Self::ReturnType;
}

pub struct SingleBatchMode;
impl GpuBatchMode for SingleBatchMode {
    const IS_BATCH: bool = false;

    type ReturnType = Precision;
    type InputType<'a> = &'a [Precision];

    fn get_samples_count(_input: &Self::InputType<'_>) -> usize {
        1
    }

    fn new_return(_: usize, _: usize) -> Self::ReturnType {
        0.0
    }

    fn set_return(ret: &mut Self::ReturnType, _: usize, _: usize, value: Precision) {
        *ret = value as Precision;
    }

    fn build_padded(input: &Self::InputType<'_>, pad_stride: usize) -> Vec<Precision> {
        let padded_len = Self::get_padded_len(Self::get_sample_length(input), pad_stride);
        let mut padded = vec![0.0; padded_len];
        for (padded, input) in padded.iter_mut().zip(input.iter()) {
            *padded = *input as Precision;
        }

        padded
    }

    fn get_sample_length(input: &Self::InputType<'_>) -> usize {
        input.len()
    }

    fn apply_fn(ret: Self::ReturnType, func: impl Fn(Precision) -> Precision) -> Self::ReturnType {
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

    type ReturnType = Vec<Vec<Precision>>;

    type InputType<'a> = &'a [Vec<Precision>];

    fn get_samples_count(input: &Self::InputType<'_>) -> usize {
        input.len()
    }

    fn new_return(alen: usize, blen: usize) -> Self::ReturnType {
        vec![vec![0.0; blen]; alen]
    }

    fn set_return(ret: &mut Self::ReturnType, i: usize, j: usize, value: Precision) {
        ret[i][j] = value as Precision;
    }

    fn build_padded(input: &Self::InputType<'_>, pad_stride: usize) -> Vec<Precision> {
        let single_padded_len = Self::get_padded_len(Self::get_sample_length(input), pad_stride);
        let mut padded = vec![0.0; input.len() * single_padded_len];
        for i in 0..input.len() {
            for j in 0..input[i].len() {
                padded[i * single_padded_len + j] = input[i][j] as Precision;
            }
        }
        padded
    }

    fn get_sample_length(input: &Self::InputType<'_>) -> usize {
        input.first().map_or(0, |x| x.len())
    }

    fn apply_fn(
        mut ret: Self::ReturnType,
        func: impl Fn(Precision) -> Precision,
    ) -> Self::ReturnType {
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
    memory_allocator: Arc<StandardMemoryAllocator>,
    params: G,
    a: M::InputType<'a>,
    b: M::InputType<'a>,
    init_val: Precision,
) -> M::ReturnType {

    let (a, b) = if M::get_sample_length(&a) > M::get_sample_length(&b) {
        (b, a)
    } else {
        (a, b)
    };

    let properties = device
        .physical_device()
        .properties();
    
    let max_threads_x = properties.max_compute_work_group_size[0];
    let max_storage_buffer_range = properties.max_storage_buffer_range;

    let a_sample_length = M::get_sample_length(&a);

    let diag_len = compute_diag_len::<M>(a_sample_length, max_threads_x as usize);

    let max_buffer_size = max_storage_buffer_range as usize;

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
        let a = M::get_subslice(&a, start, len);
        let a_padded = M::build_padded(&a, max_threads_x as usize);
        let b_padded = M::build_padded(&b, max_threads_x as usize);

        distances.push(diamond_partitioning_gpu_::<G, M>(
            device.clone(),
            queue.clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
            memory_allocator.clone(),
            &params,
            max_threads_x as usize,
            M::get_sample_length(&a),
            M::get_sample_length(&b),
            a_padded,
            b_padded,
            init_val,
            M::IS_BATCH,
        ));
        start += len;
    }
    let x = M::join_results(distances);
    x
}

#[inline(always)]
fn diamond_partitioning_gpu_<G: GpuKernelImpl, M: GpuBatchMode>(
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    params: &G,
    max_subgroup_threads: usize,
    a_sample_len: usize,
    b_sample_len: usize,
    a: Vec<Precision>,
    b: Vec<Precision>,
    init_val: Precision,
    is_batch: bool,
) -> M::ReturnType {
    let start_time = std::time::Instant::now();
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

    println!("ELAPSED PART 1: {:?}", start_time.elapsed());
    let a_gpu = move_gpu(&a, &mut builder, &memory_allocator);
    println!("ELAPSED PART 2: {:?}", start_time.elapsed());
    let b_gpu = move_gpu(&b, &mut builder, &memory_allocator);
    println!("ELAPSED PART 3: {:?}", start_time.elapsed());
    let mut diagonal = move_gpu(&diagonal, &mut builder, &memory_allocator);
    println!("ELAPSED PART 4: {:?}", start_time.elapsed());
    let kernel_params = params.build_kernel_params(memory_allocator.clone(), &mut builder);
    println!("ELAPSED PART 5: {:?}", start_time.elapsed());

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
            &kernel_params,
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

    let diagonal = move_cpu(diagonal, &mut builder, &memory_allocator);

    let command_buffer = builder.build().unwrap();
    let future = vulkano::sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();
    println!("FINISHED GPU EXEC: {:?}", start_time.elapsed());
    let mut res = M::new_return(a_count, b_count);
    println!("PREPARE RESULT: {:?}", start_time.elapsed());
    let diagonal = diagonal.read().unwrap().to_vec();
    println!("GET DIAGONAL: {:?}", start_time.elapsed());

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
