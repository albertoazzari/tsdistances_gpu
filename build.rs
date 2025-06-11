use rspirv::binary::Assemble;
use spirv_builder::{Capability, MetadataPrintout, SpirvBuilder, SpirvMetadata};
use std::env;
use spirv_tools::{
        opt::{Optimizer, Passes},
        val::Validator,
        TargetEnv,
    };

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "spirv" {
        return Ok(());
    }

    let compiled = SpirvBuilder::new(".", "spirv-unknown-spv1.5")
        .print_metadata(MetadataPrintout::Full)
        .spirv_metadata(SpirvMetadata::NameVariables)
        .capability(Capability::Int8)
        .capability(Capability::Int64)
        .build()?;

    let spirv_module = rspirv::dr::load_bytes(std::fs::read(compiled.module.unwrap_single())?)?;
    let target_env = TargetEnv::Universal_1_5;
    let validator = spirv_tools::val::create(Some(target_env));
    validator.validate(spirv_module.assemble(), None)?;
    let mut optimizer = spirv_tools::opt::create(Some(target_env));
    let passes =
                [
                    Passes::EliminateDeadFunctions,
                    Passes::DeadVariableElimination,
                    Passes::EliminateDeadConstant,
                    Passes::CombineAccessChains,
                    Passes::CompactIds,
                ];
            for pass in passes {
                optimizer.register_pass(pass);
            }
            optimizer.register_performance_passes();
    let optimizer = optimizer.optimize(spirv_module.assemble(), &mut |_| (), None)?;
    std::fs::write(
        compiled.module.unwrap_single(),
        optimizer.as_bytes(),
    )?;
    Ok(())
}
