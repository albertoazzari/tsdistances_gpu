use rspirv::binary::Assemble;
use spirv_builder::{Capability, MetadataPrintout, SpirvBuilder, SpirvMetadata};
use spirv_tools::{opt::Optimizer, val::Validator, TargetEnv};
use std::{env, error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    // Skip nested compile when compiling for spirv
    if env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "spirv" {
        return Ok(());
    }

    // Compile kernel to SPIR-V using spirv-builder
    let output = SpirvBuilder::new(".", "spirv-unknown-vulkan1.2")
        .print_metadata(MetadataPrintout::None)
        .spirv_metadata(SpirvMetadata::None)
        .capability(Capability::Int8)
        .capability(Capability::Int64)
        .build()?;

    let spirv_path = output.module.unwrap_single();
    let spirv_data = fs::read(&spirv_path)?;
    let spirv_words: Vec<u32> = bytemuck::cast_slice(&spirv_data).to_vec();

    // Validate before optimization (optional but recommended)
    let validator = spirv_tools::val::create(Some(TargetEnv::Vulkan_1_2));
    validator.validate(&spirv_words, None)?;

    // Optimize SPIR-V using performance passes (like krnlc)
    let mut optimizer = spirv_tools::opt::create(Some(TargetEnv::Vulkan_1_2));
    optimizer.register_performance_passes(); // equivalent to `spirv-opt -O`
    
    let optimized = optimizer.optimize(&spirv_words, &mut |_| (), None)?;
    let mut module = rspirv::dr::load_words(&optimized.as_words())
        .map_err(|e| format!("Failed to parse optimized SPIR-V: {e}"))?;

    // Remove VulkanMemoryModel capability
    use rspirv::spirv::Capability;
    module.capabilities.retain(|inst| {
        inst.operands.first().map_or(true, |op| {
            op.unwrap_capability() != Capability::VulkanMemoryModel
        })
    });

    // Reassemble and write back
    let cleaned = module.assemble();
    fs::write(&spirv_path, bytemuck::cast_slice(&cleaned))?;

    println!("cargo:rustc-env=tsdistances.spv={}", spirv_path.display());

    Ok(())
}
