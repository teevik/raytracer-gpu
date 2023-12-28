use spirv_std::{glam::Vec3, num_traits::Float};

use crate::{
    data::{Face, Range},
    ray::Ray,
    RayHit,
};

#[derive(Clone, Copy)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl Sphere {
    pub fn raytrace(self, ray: Ray, range: Range<f32>) -> RayHit {
        let center_to_origin = ray.origin - self.center;
        let a = ray.direction.length_squared();
        let half_b = Vec3::dot(center_to_origin, ray.direction);
        let c = center_to_origin.length_squared() - (self.radius * self.radius);

        let discriminant = (half_b * half_b) - (a * c);
        if discriminant < 0. {
            return RayHit::none();
        }

        let discriminant_sqrt = Float::sqrt(discriminant.sqrt());

        // Find the nearest root that lies in the acceptable range
        let mut root = (-half_b - discriminant_sqrt) / a;
        if !range.contains(root) {
            root = (-half_b + discriminant_sqrt) / a;

            if !range.contains(root) {
                return RayHit::none();
            }
        }

        let distance = root;
        let point = ray.at(distance);

        let outward_normal = (point - self.center) / self.radius;
        let face = ray.get_face(outward_normal);

        let normal = match face {
            Face::Front => outward_normal,
            Face::Back => -outward_normal,
        };

        RayHit {
            did_hit: true,
            distance,
            point,
            face,
            normal,
        }
    }
}
