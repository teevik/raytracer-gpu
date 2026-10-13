use glam::Vec3;
use spirv_std::num_traits::Float;

use crate::{material::Material, ray::Ray};

#[derive(Clone, Copy)]
pub struct Range {
    pub min: f32,
    pub max: f32,
}

impl Range {
    pub fn new(start: f32, end: f32) -> Self {
        Self {
            min: start,
            max: end,
        }
    }

    pub fn combine(a: Self, b: Self) -> Self {
        Self::new(Float::min(a.min, b.min), Float::max(a.max, b.max))
    }

    pub fn contains(self, value: f32) -> bool {
        value >= self.min && value < self.max
    }

    pub fn expand(self, delta: f32) -> Range {
        let padding = delta / 2.;

        Range {
            min: self.min - padding,
            max: self.max + padding,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub enum Face {
    #[default]
    Front,
    Back,
}

#[derive(Clone, Copy, Default)]
pub struct RayHit {
    /// Whether or not this ray hit something
    pub did_hit: bool,

    /// Distance to hit
    pub distance: f32,

    /// The point where the ray hit
    pub point: Vec3,

    /// Which face
    pub face: Face,

    /// Normal, unit length
    pub normal: Vec3,

    /// The material of the hit shape
    pub material: Material,
}

impl RayHit {
    pub fn none() -> Self {
        Default::default()
    }
}

/// Result of a closest-hit search: just the sphere index and distance, so the search loops don't
/// copy a full `RayHit` per candidate. The `RayHit` is computed once afterwards.
#[derive(Clone, Copy)]
pub struct Closest {
    /// Index into the sphere buffer, `Closest::NO_HIT` if nothing was hit.
    pub index: u32,
    pub distance: f32,
}

impl Closest {
    pub const NO_HIT: u32 = u32::MAX;

    pub fn none(max_distance: f32) -> Self {
        Self {
            index: Self::NO_HIT,
            distance: max_distance,
        }
    }
}

// rust-gpu decorates even function-local structs with their Rust member offsets, and naga (in
// wgpu) treats a struct containing a `vec3` as 16-byte aligned with its size rounded up to 16.
// `Ray` is 24 bytes in Rust but 32 to naga, so any member placed right after it "overlaps" and the
// module is rejected. `repr(C)` with `scattered` last keeps rustc from reordering it into that spot.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct ScatterResult {
    /// Whether or not did scatter
    pub did_scatter: bool,

    /// The color produced from scattering
    pub attenuation: Vec3,

    /// The new ray
    pub scattered: Ray,
}

impl ScatterResult {
    pub fn none() -> Self {
        Default::default()
    }
}
