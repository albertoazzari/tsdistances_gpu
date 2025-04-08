use spirv_builder::{MetadataPrintout, SpirvBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Specify the target architecture
    let target = "spirv-unknown-spv1.5".to_string();
    // Specify the shader crate to build
    let shader_crate = "tsdistances_gpu/";
    SpirvBuilder::new(shader_crate, target)
        .print_metadata(MetadataPrintout::Full)
        .build()?;
    Ok(())
}
