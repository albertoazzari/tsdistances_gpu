use std::sync::Arc;

use crate::{
    kernels::kernel_trait::GpuKernelImpl,
    utils::{SubBuffersAllocator, move_diag, move_ts},
};
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage
    }, descriptor_set::allocator::StandardDescriptorSetAllocator, device::{Device, Queue}, memory::MemoryHeapFlags, sync::GpuFuture
};
use std::cmp::max;

fn compute_sample_len(a: &Vec<Vec<f32>>) -> usize {
    a.iter().map(|x| x.len()).sum()
}

fn flatten_and_pad(a: &Vec<Vec<f32>>, pad: usize) -> Vec<f32> {
    let new_len = next_multiple_of_n(a.first().unwrap().len(), pad);
    let mut padded = vec![0.0; new_len * a.len()];
    for i in 0..a.len() {
        for j in 0..a[i].len() {
            padded[i * new_len + j] = a[i][j];
        }
    }
    padded
}

pub fn diamond_partitioning_gpu<G: GpuKernelImpl>(
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    subbuffer_allocator: SubBuffersAllocator,
    params: G,
    a: &Vec<Vec<f32>>,
    b: &Vec<Vec<f32>>,
    init_val: f32,
) -> Vec<Vec<f32>> {
    let (a, b) = if compute_sample_len(a) > compute_sample_len(b) {
        (b, a)
    } else {
        (a, b)
    };

    let pdevice = device.physical_device();
    let properties = pdevice.properties();
    let max_subgroup_size = properties.max_subgroup_size.unwrap() as usize;
    let max_workgroup_size = properties.max_compute_work_group_size[0] as usize;
    let max_storage_buffer_size = properties.max_storage_buffer_range as usize;
    let device_mem = pdevice.memory_properties().memory_heaps
        .iter()
        .filter(|heap| heap.flags.contains(MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .sum::<u64>() as usize;
    
    let a_count = a.len();
    let a_len = next_multiple_of_n(a.first().unwrap().len(), max_subgroup_size);
    let b_count = b.len();
    let b_len = next_multiple_of_n(b.first().unwrap().len(), max_subgroup_size);
    let len = max(a_len, b_len);

    let a_padded = flatten_and_pad(&a, max_subgroup_size);
    let b_padded = flatten_and_pad(&b, max_subgroup_size);

    let diag_len = 2 * (next_multiple_of_n(len, max_subgroup_size) + 1).next_power_of_two();

    // let chunk_size_mem = total_device_memory / (2*ts_len + diag_len);
    let chunk_size_buf = max_storage_buffer_size / (diag_len * std::mem::size_of::<f32>());
    // let chunk_size = chunk_size_buf.min(chunk_size_mem);

    // panic!("chunk_size_buf = {chunk_size_buf}, max_storage_buffer_size = {max_storage_buffer_size}, diag_len = {diag_len}, device_mem = {device_mem}, sizeof(f32) = {}", std::mem::size_of::<f32>());

    diamond_partitioning_gpu_::<G>(
        device.clone(),
        queue.clone(),
        command_buffer_allocator.clone(),
        descriptor_set_allocator.clone(),
        subbuffer_allocator.clone(),
        &params,
        max_subgroup_size,
        max_workgroup_size,
        a_len,
        b_len,
        a_padded,
        b_padded,
        a_count,
        b_count,
        init_val,
    )
}

#[inline(always)]
fn diamond_partitioning_gpu_<G: GpuKernelImpl>(
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    buffer_allocator: SubBuffersAllocator,
    params: &G,
    max_subgroup_threads: usize,
    max_workgroup_size: usize,
    a_len: usize,
    b_len: usize,
    a_padded: Vec<f32>,
    b_padded: Vec<f32>,
    a_count: usize,
    b_count: usize,
    init_val: f32,
) -> Vec<Vec<f32>> {
    let a_padded_len = a_padded.len() / a_count;
    let b_padded_len = b_padded.len() / b_count;

    let diag_len = 2 * (max(a_padded_len, b_padded_len) + 1).next_power_of_two();

    let mut diagonal = vec![init_val; a_count * b_count * diag_len];

    for i in 0..(a_count * b_count) {
        diagonal[i * diag_len] = 0.0;
    }

    let n_tiles_in_a = a_padded_len.div_ceil(max_subgroup_threads);
    let n_tiles_in_b = b_padded_len.div_ceil(max_subgroup_threads);

    let rows_count = (a_padded_len + b_padded_len).div_ceil(max_subgroup_threads) - 1;

    let mut diamonds_count = 1;
    let mut first_coord = -(max_subgroup_threads as isize);
    let mut a_start = 0;
    let mut b_start = 0;

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).unwrap();

    let a_gpu = move_ts(&a_padded, &buffer_allocator, max_workgroup_size);
    let b_gpu = move_ts(&b_padded, &buffer_allocator, max_workgroup_size);
    let mut diagonal = move_diag(
        &diagonal,
        &buffer_allocator,
        max_workgroup_size,
    );
    let kernel_params = params.build_kernel_params(buffer_allocator.clone(), max_workgroup_size);

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
            a_len as u64,
            b_len as u64,
            max_subgroup_threads as u64,
            &a_gpu,
            &b_gpu,
            &mut diagonal,
            &kernel_params,
        );

        if i < (n_tiles_in_a - 1) {
            diamonds_count += 1;
            first_coord -= max_subgroup_threads as isize;
            a_start += max_subgroup_threads;
        } else if i < (n_tiles_in_b - 1) {
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

    let (_, cx) = index_mat_to_diag(a_len, b_len);

    // let diagonal = move_cpu(&buffer_allocator, &diagonal, &mut builder);

    // let start_time = std::time::Instant::now();
    let command_buffer = builder.build().unwrap();
    let future = vulkano::sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();
    // println!(
    //     "GPU - Command Buffer executed in {} ms",
    //     start_time.elapsed().as_millis()
    // );
    let mut res = vec![vec![0.0; b_count]; a_count];
    let diagonal = diagonal.read().unwrap();
    for i in 0..a_count {
        for j in 0..b_count {
            let diag_offset = (i * b_count + j) * diag_len;
            res[i][j] = diagonal[diag_offset + ((cx as usize) & (diag_len - 1))];
        }
    }
    res
}

fn next_multiple_of_n(x: usize, n: usize) -> usize {
    (x + n - 1) / n * n
}
