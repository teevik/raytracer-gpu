use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use spirv_std::num_traits::Float;

use crate::{
    RayHit,
    bvh::Aabb,
    data::{Face, Range},
    material::Material,
    ray::Ray,
    traits::Raycastable,
};

/// `radius` fills the padding after `center`, keeping `material` 16-byte aligned for std430
#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub material: Material,
}

const _: () = {
    assert!(size_of::<Sphere>() == 48);
    assert!(core::mem::offset_of!(Sphere, material) == 16);
};

impl Sphere {
    pub fn get_aabb(self) -> Aabb {
        let radius = Vec3::splat(self.radius);

        Aabb::from_extremes(self.center - radius, self.center + radius)
    }

    /// `f32::MAX` on a miss, since naga rejects infinite float literals
    pub fn intersect_distance(self, ray: Ray, range: Range) -> f32 {
        let center_to_origin = ray.origin - self.center;
        let a = ray.direction.length_squared();
        let half_b = Vec3::dot(center_to_origin, ray.direction);
        let c = center_to_origin.length_squared() - (self.radius * self.radius);

        let discriminant = (half_b * half_b) - (a * c);
        if discriminant < 0. {
            return f32::MAX;
        }

        let discriminant_sqrt = Float::sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range
        let mut root = (-half_b - discriminant_sqrt) / a;
        if !range.contains(root) {
            root = (-half_b + discriminant_sqrt) / a;

            if !range.contains(root) {
                return f32::MAX;
            }
        }

        root
    }
}

impl Raycastable for Sphere {
    fn raycast(self, ray: Ray, range: Range) -> RayHit {
        let distance = self.intersect_distance(ray, range);
        if !range.contains(distance) {
            return RayHit::none();
        }

        let point = ray.at(distance);

        let outward_normal = (point - self.center) / self.radius;
        let face = ray.get_face(outward_normal);

        let normal = match face {
            Face::Front => outward_normal,
            Face::Back => -outward_normal,
        };

        let material = self.material;

        RayHit {
            did_hit: true,
            distance,
            point,
            face,
            normal,
            material,
        }
    }
}
