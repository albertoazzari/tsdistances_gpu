use crate::{cpu::compute_shader, utils};


#[test]
fn test_device() {
    let (physical_device, _, queue_family_index) = utils::get_device();
    println!(
        "Physical device: {:?}",
        physical_device.properties().device_name
    );
    println!("Queue family index: {:?}", queue_family_index);
    println!(
        "Device type: {:?}",
        physical_device.properties().device_type
    );
}

#[test]
fn test_shader() {
    let shader = include_bytes!(env!("tsdistances.spv"));;
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
