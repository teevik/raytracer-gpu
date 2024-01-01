use vek::Vec3;

use crate::{material::Material, ray::Ray, F};

#[derive(Clone, Copy)]
pub struct Range<T: Copy> {
    pub start: T,
    pub end: T,
}

impl<T: Copy + PartialOrd> Range<T> {
    pub fn new(start: T, end: T) -> Self {
        Self { start, end }
    }

    pub fn contains(self, value: T) -> bool {
        value >= self.start && value < self.end
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
    pub distance: F,

    /// The point where the ray hit
    pub point: Vec3<F>,

    /// Which face
    pub face: Face,

    /// Normal, unit length
    pub normal: Vec3<F>,

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
    pub attenuation: Vec3<F>,
}

impl ScatterResult {
    pub fn none() -> Self {
        Default::default()
    }
}
