use spirv_std::num_traits::Float;
use vek::Vec3;

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
    pub point: Vec3<f32>,

    /// Which face
    pub face: Face,

    /// Normal, unit length
    pub normal: Vec3<f32>,

    /// The material of the hit shape
    pub material: Material,
}

impl RayHit {
    pub fn none() -> Self {
        Default::default()
    }
}

#[derive(Clone, Copy, Default)]
pub struct ScatterResult {
    /// Whether or not did scatter
    pub did_scatter: bool,

    /// The new ray
    pub scattered: Ray,

    /// The color produced from scattering
    pub attenuation: Vec3<f32>,
}

impl ScatterResult {
    pub fn none() -> Self {
        Default::default()
    }
}
