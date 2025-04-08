use spirv_builder::{MetadataPrintout, SpirvBuilder};

#[cfg(feature = "build_cpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Specify the target architecture
    let target = "spirv-unknown-spv1.5".to_string();
    // Specify the shader crate to build
    let shader_crate = ".";
    SpirvBuilder::new(shader_crate, target)
        .print_metadata(MetadataPrintout::Full)
        .build()?;
    Ok(())
}

#[cfg(not(feature = "build_cpu"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
