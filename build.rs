use spirv_builder::{MetadataPrintout, SpirvBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "spirv" {
        // Avoid nested compile of the shader code
        return Ok(());
    }
    // Specify the target architecture
    let target = "spirv-unknown-spv1.5".to_string();
    // Specify the shader crate to build
    let shader_crate = ".";
    SpirvBuilder::new(shader_crate, target)
        .print_metadata(MetadataPrintout::Full)
        .build()?;
    Ok(())
}
