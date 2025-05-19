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
    let passes = {
        use Passes::*;
        [
            EliminateDeadFunctions,
            DeadVariableElimination,
            EliminateDeadConstant,
            CombineAccessChains,
            CompactIds,
        ]
    };
    for pass in passes {
        optimizer.register_pass(pass);
    }
    optimizer.register_performance_passes();

    let spirv = SpirvBuilder::new(".", "spirv-unknown-vulkan1.2")
        .print_metadata(MetadataPrintout::Full)
        .spirv_metadata(SpirvMetadata::None)
        .capability(Capability::Int64)
        .capability(Capability::Int8)
        .build()?;

    let spirv_module = rspirv::dr::load_bytes(std::fs::read(spirv.module.unwrap_single())?)
        .map_err(|e| e.to_string())?;

    let entry_fns: FxHashSet<u32> = spirv_module
        .entry_points
        .iter()
        .map(|inst| inst.operands[1].unwrap_id_ref())
        .collect();

    for entry_point in &spirv_module.entry_points {
        let entry_id = entry_point.operands[1].unwrap_id_ref();
        let execution_mode = spirv_module
            .execution_modes
            .iter()
            .find(|inst| inst.operands.first().unwrap().unwrap_id_ref() == entry_id)
            .unwrap();
        let functions = spirv_module
            .functions
            .iter()
            .filter(|f| {
                let id = f.def.as_ref().unwrap().result_id.unwrap();
                id == entry_id || !entry_fns.contains(&id)
            })
            .cloned()
            .collect();

        let spirv_module = Module {
            entry_points: vec![entry_point.clone()],
            execution_modes: vec![execution_mode.clone()],
            functions,
            ..spirv_module.clone()
        };
        let spirv = spirv_module.assemble();
        let optimized = optimizer
            .optimize(spirv, &mut |_| (), None)
            .expect("Failed to optimize SPIR-V");
        validator
            .validate(optimized, None)
            .expect("Failed to validate SPIR-V");
    }
    Ok(())
}
