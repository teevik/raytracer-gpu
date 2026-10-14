use glam::Vec3;

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

    pub fn contains(self, value: f32) -> bool {
        value >= self.min && value < self.max
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

#[derive(Clone, Copy)]
pub struct Closest {
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

// naga sees `Ray` as 32 bytes (Rust: 24), so it has to stay the last member or naga
// rejects the overlapping offsets
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
