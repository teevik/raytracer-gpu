use spirv_builder::{MetadataPrintout, SpirvBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=X11");
        println!("cargo:rustc-link-lib=Xcursor");
        println!("cargo:rustc-link-lib=Xrandr");
        println!("cargo:rustc-link-lib=Xi");
        println!("cargo:rustc-link-lib=vulkan");
    }

    SpirvBuilder::new("../shader", "spirv-unknown-spv1.3")
        .print_metadata(MetadataPrintout::Full)
        .build()?;

    Ok(())
}
