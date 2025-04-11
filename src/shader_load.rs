use std::sync::{Arc, OnceLock};

use dashmap::DashMap;
use vulkano::{
    device::Device,
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
    Validated, VulkanError,
};

static SHADER_MODULE: OnceLock<Arc<ShaderModule>> = OnceLock::new();
static SHADE_PIPELINES: OnceLock<DashMap<&'static str, Arc<ComputePipeline>>> = OnceLock::new();

const SHADER_CODE: &[u8] = include_bytes!(env!("tsdistances.spv"));

fn load(device: Arc<Device>, shader: &[u8]) -> Result<Arc<ShaderModule>, Validated<VulkanError>> {
    // convert from &[u8] to &[u32]
    unsafe {
        let shader = std::slice::from_raw_parts(
            shader.as_ptr() as *const u32,
            shader.len() / std::mem::size_of::<u32>(),
        );
        ShaderModule::new(device, ShaderModuleCreateInfo::new(shader)).map_err(|e| {
            eprintln!("Failed to load shader module: {:?}", e);
            e
        })
    }
}

pub fn get_shader_entry_pipeline(device: Arc<Device>, name: &'static str) -> Arc<ComputePipeline> {
    let shader_module = SHADER_MODULE.get_or_init(|| load(device.clone(), SHADER_CODE).unwrap());
    let pipelines = SHADE_PIPELINES.get_or_init(Default::default);

    match pipelines.entry(name) {
        dashmap::Entry::Occupied(entry) => entry.get().clone(),
        dashmap::Entry::Vacant(vacant_entry) => {
            let Some(entry_point) = shader_module.entry_point(name) else {
                panic!("Entry point {} not found in shader module", name);
            };
            let stage = PipelineShaderStageCreateInfo::new(entry_point);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            let pipeline = ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap();
            vacant_entry.insert(pipeline.clone());
            pipeline
        }
    }
}
