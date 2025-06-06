use spirv_builder::{Capability, MetadataPrintout, SpirvBuilder, SpirvMetadata};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "spirv" {
        return Ok(());
    }

    SpirvBuilder::new(".", "spirv-unknown-spv1.5")
        .print_metadata(MetadataPrintout::Full)
        .spirv_metadata(SpirvMetadata::NameVariables)
        .capability(Capability::Int8)
        .capability(Capability::Int64)
        .capability(Capability::Float64)
        .build()?;

    Ok(())
}
