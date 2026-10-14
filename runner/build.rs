use cargo_gpu_install::install::Install;
use cargo_gpu_install::spirv_builder::{ShaderPanicStrategy, SpirvMetadata};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let shader_crate = [env!("CARGO_MANIFEST_DIR"), "..", "shader"]
        .iter()
        .collect::<PathBuf>();

    // Installs rust-gpu's nightly toolchain and builds the codegen backend into ~/.cache/rust-gpu
    let mut install = Install::from_shader_crate(shader_crate.clone()).within_build_script();
    install.auto_install_rust_toolchain = true;
    let backend = install.run()?;

    let mut builder = backend.to_spirv_builder(shader_crate, "spirv-unknown-vulkan1.2");
    builder.build_script.defaults = true;
    builder.shader_panic_strategy = ShaderPanicStrategy::SilentExit;
    builder.spirv_metadata = SpirvMetadata::NameVariables;

    let result = builder.build()?;
    let spv_path = result.module.unwrap_single();
    println!("cargo::rustc-env=SHADER_SPV_PATH={}", spv_path.display());

    Ok(())
}
