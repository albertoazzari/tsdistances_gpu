use rspirv::binary::Assemble;
use spirv_builder::{Capability, MetadataPrintout, SpirvBuilder, SpirvMetadata};
use spirv_tools::{opt::Optimizer, val::Validator, TargetEnv};
use std::{env, error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    if env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "spirv" {
        return Ok(());
    }

    let mut builder = SpirvBuilder::new(".", "spirv-unknown-vulkan1.2")
        .print_metadata(MetadataPrintout::Full)
        .spirv_metadata(SpirvMetadata::None);

    let capabilities = {
        use spirv_builder::Capability::*;
        [
            Int8,
            Int64,
            VulkanMemoryModel,
        ]
    };
    for c in capabilities {
        builder = builder.capability(c);
    }

    let compiled = builder.build()?;
    let spirv_path = compiled.module.unwrap_single();
    let spirv_module =
        rspirv::dr::load_bytes(std::fs::read(spirv_path)?).map_err(|e| e.to_string())?;

    for entry_point in &spirv_module.entry_points {
        let original_spirv = spirv_module.assemble();

        // Run DCE + Performance + Optional Final DCE
        let optimized_1 = spirv_opt(&original_spirv, SpirvOptKind::DeadCodeElimination)?;
        let optimized_2 = spirv_opt(&optimized_1.as_words(), SpirvOptKind::Performance)?;
        let final_optimized =
            spirv_opt(&optimized_2.as_words(), SpirvOptKind::DeadCodeElimination)?;

        let mut spirv_module =
            rspirv::dr::load_words(&final_optimized.as_words()).map_err(|e| e.to_string())?;

        spirv_module.capabilities.retain(|inst| {
            use rspirv::spirv::Capability::*;
            matches!(
                inst.operands.first().unwrap().unwrap_capability(),
                Shader
                    | VulkanMemoryModel
                    | Int64
                    | Int8
            )
        });

        let final_spirv = spirv_module.assemble();
        spirv_val(&final_spirv)?;
        fs::write(
            spirv_path.with_extension("spv"),
            bytemuck::cast_slice(final_spirv.as_slice()),
        )?;
    }

    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum SpirvOptKind {
    DeadCodeElimination,
    Performance,
}

fn spirv_opt(
    spirv: &[u32],
    kind: SpirvOptKind,
) -> Result<spirv_tools::binary::Binary, Box<dyn Error>> {
    use spirv_tools::opt::Passes::*;
    let target_env = TargetEnv::Vulkan_1_2;

    let validator = spirv_tools::val::create(Some(target_env));
    validator.validate(spirv, None)?;

    let mut optimizer = spirv_tools::opt::create(Some(target_env));

    match kind {
        SpirvOptKind::DeadCodeElimination => {
            optimizer.register_pass(StripDebugInfo);
            optimizer.register_pass(EliminateDeadFunctions);
            optimizer.register_pass(DeadVariableElimination);
            optimizer.register_pass(EliminateDeadConstant);
            optimizer.register_pass(EliminateDeadMembers);
            optimizer.register_pass(LocalAccessChainConvert);
            optimizer.register_pass(LocalSingleStoreElim);
            optimizer.register_pass(LocalMultiStoreElim);
            optimizer.register_pass(CopyPropagateArrays);
            optimizer.register_pass(CombineAccessChains);
            optimizer.register_pass(CompactIds);
            optimizer.register_pass(UnifyConstant);
        }
        SpirvOptKind::Performance => {
            optimizer.register_pass(StripDebugInfo);
            optimizer.register_performance_passes();
            optimizer.register_pass(LocalAccessChainConvert);
            optimizer.register_pass(CopyPropagateArrays);
            optimizer.register_pass(CompactIds);
        }
    }

    Ok(optimizer.optimize(spirv, &mut |_| (), None)?)
}

fn spirv_val(spirv: &[u32]) -> Result<(), Box<dyn Error>> {
    let validator = spirv_tools::val::create(Some(TargetEnv::Vulkan_1_2));
    validator.validate(spirv, None)?;
    Ok(())
}
