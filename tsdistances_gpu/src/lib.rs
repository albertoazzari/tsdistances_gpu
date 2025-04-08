#![cfg_attr(target_arch = "spirv", no_std)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]

use glam::UVec3;
use spirv_std::{glam, spirv};
use spirv_std::num_traits::Float;

#[spirv(compute(threads(64)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input_a: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] input_b: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] diagonal: &mut [f32],
) {
    let index = id.x as usize;
    diagonal[index] = (input_a[index]*input_b[index]).log2();
}