use spirv_builder::{Capability, MetadataPrintout, SpirvBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=X11");
        println!("cargo:rustc-link-lib=Xcursor");
        println!("cargo:rustc-link-lib=Xrandr");
        println!("cargo:rustc-link-lib=Xi");
        println!("cargo:rustc-link-lib=vulkan");
    }

    SpirvBuilder::new("../shader", "spirv-unknown-vulkan1.2")
        .print_metadata(MetadataPrintout::Full)
        .capability(Capability::Float64)
        .capability(Capability::Int64)
        .build()?;

    Ok(())
}
