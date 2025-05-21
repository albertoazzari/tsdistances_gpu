use fxhash::FxHashSet;
use rspirv::{binary::Assemble, dr::Module};
use spirv_builder::{Capability, MetadataPrintout, SpirvBuilder, SpirvMetadata};
use spirv_tools::{
    binary::Binary,
    opt::{Optimizer, Passes},
    val::Validator,
    Error, TargetEnv,
};
use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "spirv" {
        return Ok(());
    }

    let target_env = TargetEnv::Vulkan_1_2;
    let validator = spirv_tools::val::create(Some(target_env));
    let mut optimizer = spirv_tools::opt::create(Some(target_env));
    optimizer.register_performance_passes();

    let spirv = SpirvBuilder::new(".", "spirv-unknown-vulkan1.2")
        .print_metadata(MetadataPrintout::Full)
        .spirv_metadata(SpirvMetadata::None)
        .capability(Capability::Int64)
        .capability(Capability::Int8)
        .build()?;

    let spirv_path = spirv.module.unwrap_single();

    let spirv_module =
        rspirv::dr::load_bytes(std::fs::read(spirv_path)?).map_err(|e| e.to_string())?;

    let spirv = spirv_module.assemble();
    let optimized = optimizer
        .optimize(spirv, &mut |_| (), None)
        .expect("Failed to optimize SPIR-V");
    validator
        .validate(&optimized, None)
        .expect("Failed to validate SPIR-V");
    fs::write(spirv_path, &optimized).expect("Failed to write optimized SPIR-V");
    Ok(())
}
