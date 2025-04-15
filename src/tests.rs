use std::sync::Arc;

use vulkano::{command_buffer::allocator::StandardCommandBufferAllocator, descriptor_set::allocator::StandardDescriptorSetAllocator, device::{self, Device, DeviceCreateInfo, QueueCreateInfo}};

use crate::warps::{MultiBatchMode, SingleBatchMode};

#[test]
fn test_device() {
    let (device, _, _, _) = crate::utils::get_device();
    println!(
        "Physical device: {:?} type: {:?}",
        device.physical_device().properties().device_name,
        device.physical_device().properties().device_name
    );
}

#[test]
fn test_shader() {
    use crate::{cpu::compute_shader, utils};
    let shader = include_bytes!(env!("tsdistances.spv"));
    let (physical_device, device_extensions, queue_family_index) = utils::get_device();
    let result = compute_shader(
        shader,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        &mut [0.0; 10],
        physical_device.clone(),
        device_extensions,
        queue_family_index,
    );
    assert!(result.is_ok(), "Failed to run compute shader: {:?}", result);
}

#[test]
pub fn test_erp() {

    let (device, queue, sba, sda) = crate::utils::get_device();
    

    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let gap_penalty = 1.0;
    let result = crate::cpu::erp::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        command_buffer_allocator.clone(),
        descriptor_set_allocator.clone(),
        &a,
        &b,
        gap_penalty,
    );

    println!("{:?}", result)
    // assert_eq!(result, 0.0, "Failed to run ERP: {:?}", result);

}
