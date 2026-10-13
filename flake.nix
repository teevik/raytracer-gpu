{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      perSystem =
        { pkgs, lib, ... }:
        {
          devShells.default = pkgs.mkShell {
            packages = with pkgs; [
              rustup
              vulkan-loader
            ];

            # wgpu/ash dlopen libvulkan at runtime, so it has to be on the library path
            env.LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.vulkan-loader ];
          };
        };
    };
}
