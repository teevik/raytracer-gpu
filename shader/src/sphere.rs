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

/// `radius` fills the padding after `center`, keeping `motion` and `material` 16-byte aligned for
/// std430
#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct Sphere {
    /// Center at time 0
    pub center: Vec3,
    pub radius: f32,
    /// How far the center moves between time 0 and time 1
    pub motion: Vec3,
    _padding: f32,
    pub material: Material,
}

const _: () = {
    assert!(size_of::<Sphere>() == 64);
    assert!(core::mem::offset_of!(Sphere, motion) == 16);
    assert!(core::mem::offset_of!(Sphere, material) == 32);
};

impl Sphere {
    pub fn stationary(center: Vec3, radius: f32, material: Material) -> Self {
        Self::moving(center, center, radius, material)
    }

    /// Moves linearly from `start` at time 0 to `end` at time 1
    pub fn moving(start: Vec3, end: Vec3, radius: f32, material: Material) -> Self {
        Self {
            center: start,
            radius,
            motion: end - start,
            _padding: 0.,
            material,
        }
    }

    pub fn center_at(self, time: f32) -> Vec3 {
        self.center + time * self.motion
    }

    /// Bounds the sphere over its whole motion
    pub fn get_aabb(self) -> Aabb {
        let radius = Vec3::splat(self.radius);
        let start = self.center;
        let end = self.center_at(1.);

        Aabb {
            min: start.min(end) - radius,
            max: start.max(end) + radius,
        }
    }

    /// `f32::MAX` on a miss, since naga rejects infinite float literals
    pub fn intersect_distance(self, ray: Ray, range: Range) -> f32 {
        let center_to_origin = ray.origin - self.center_at(ray.time);
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

        let outward_normal = (point - self.center_at(ray.time)) / self.radius;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moving_sphere_is_hit_where_it_is_at_the_ray_time() {
        let sphere = Sphere::moving(
            Vec3::new(0., 0., -5.),
            Vec3::new(0., 2., -5.),
            0.5,
            Material::default(),
        );
        let range = Range::new(0.001, f32::MAX);

        for (time, y, hits) in [(0., 0., true), (1., 2., true), (0.5, 0., false)] {
            let ray = Ray {
                origin: Vec3::new(0., y, 0.),
                direction: Vec3::NEG_Z,
                time,
            };
            let distance = sphere.intersect_distance(ray, range);
            assert_eq!(distance < f32::MAX, hits, "time {time}, y {y}");

            if hits {
                assert!((distance - 4.5).abs() < 1e-5);
                assert!((sphere.raycast(ray, range).normal - Vec3::Z).length() < 1e-5);
            }
        }
    }

    #[test]
    fn aabb_covers_the_whole_motion() {
        let sphere = Sphere::moving(Vec3::ZERO, Vec3::new(1., -2., 0.), 0.5, Material::default());
        let aabb = sphere.get_aabb();

        assert_eq!(aabb.min, Vec3::new(-0.5, -2.5, -0.5));
        assert_eq!(aabb.max, Vec3::new(1.5, 0.5, 0.5));
    }
}
