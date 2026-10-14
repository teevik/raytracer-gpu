#![no_std]

mod bvh;
mod data;
mod material;
mod rand;
mod ray;
mod sphere;
mod traits;

use bytemuck::{Pod, Zeroable};
use data::{Closest, Range, RayHit};
use glam::{UVec2, Vec2, Vec3, Vec4};
use rand::{Rand, hash_combine2};
use ray::Ray;
use spirv_std::{num_traits::Float, spirv};
use traits::Raycastable;

pub use bvh::{Aabb, BvhNode};
pub use glam::UVec3;
pub use material::{Material, Reflection};
pub use sphere::Sphere;

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct RaytraceSettings {
    pub viewport: Viewport,
    pub screen_size: UVec2,
    pub amount_of_samples: u32,
    pub max_depth: u32,
    pub use_bvh: u32,
    pub _padding: [u32; 3],
}

const _: () = {
    assert!(size_of::<RaytraceSettings>() == 128);
    assert!(core::mem::offset_of!(RaytraceSettings, screen_size) == 96);
    assert!(core::mem::offset_of!(RaytraceSettings, use_bvh) == 112);
};

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct DispatchParams {
    pub seed: u32,
    pub sample_start: u32,
    pub sample_count: u32,
    pub _padding: u32,
}

const _: () = assert!(size_of::<DispatchParams>() == 16);

/// std430 aligns `vec3` to 16 bytes, hence the padding
#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct Viewport {
    pub origin: Vec3,
    pub _padding0: f32,
    pub upper_left_pixel_position: Vec3,
    pub _padding1: f32,

    pub horizontal_pixel_delta: Vec3,
    pub _padding2: f32,
    pub vertical_pixel_delta: Vec3,
    pub _padding3: f32,

    pub horizontal_defocus_disk: Vec3,
    pub _padding4: f32,
    pub vertical_defocus_disk: Vec3,
    pub _padding5: f32,
}

fn closest_sphere(spheres: &[Sphere], ray: Ray, range: Range) -> Closest {
    let mut closest = Closest::none(range.max);

    // rust-gpu can't compile slice iterators
    #[allow(clippy::needless_range_loop)]
    for i in 0..spheres.len() {
        let distance = spheres[i].intersect_distance(ray, Range::new(range.min, closest.distance));

        if distance < closest.distance {
            closest = Closest {
                index: i as u32,
                distance,
            };
        }
    }

    closest
}

fn raytrace_spheres(
    spheres: &[Sphere],
    bvh_nodes: &[BvhNode],
    use_bvh: bool,
    ray: Ray,
    range: Range,
) -> RayHit {
    let closest = if use_bvh {
        bvh::closest_sphere(bvh_nodes, spheres, ray, range)
    } else {
        closest_sphere(spheres, ray, range)
    };

    if closest.index == Closest::NO_HIT {
        RayHit::none()
    } else {
        spheres[closest.index as usize].raycast(ray, range)
    }
}

fn ray_color(
    ray: Ray,
    spheres: &[Sphere],
    bvh_nodes: &[BvhNode],
    use_bvh: bool,
    max_depth: u32,
    rand: &mut Rand,
) -> Vec3 {
    let mut accumulated_color = Vec3::ONE;
    let mut next_ray = ray;

    for _ in 0..max_depth {
        let ray_hit = raytrace_spheres(
            spheres,
            bvh_nodes,
            use_bvh,
            next_ray,
            Range::new(0.001, Float::max_value()),
        );

        if ray_hit.did_hit {
            let scatter_result = ray_hit.material.scatter(next_ray, ray_hit, rand);

            if scatter_result.did_scatter {
                accumulated_color *= scatter_result.attenuation;
                next_ray = scatter_result.scattered;
            } else {
                // Didn't scatter
                return Vec3::ZERO;
            }
        } else {
            // Didn't hit anything
            let unit_direction = next_ray.direction.normalize();
            let a = (unit_direction.y + 1.) / 2.;
            let background_color = Vec3::splat(1. - a) + a * Vec3::new(0.5, 0.7, 1.);

            accumulated_color *= background_color;

            return accumulated_color;
        }
    }

    // Reached max depth
    Vec3::ZERO
}

fn pixel_sample_offset(rand: &mut Rand) -> Vec2 {
    rand.gen_vec2() - (Vec2::ONE / 2.) // From -0.5 to 0.5
}

fn defocus_sample_offset(rand: &mut Rand) -> Vec2 {
    rand.gen_in_unit_disk()
}

fn trace_sample(
    pixel_position: UVec2,
    raytrace_settings: &RaytraceSettings,
    spheres: &[Sphere],
    bvh_nodes: &[BvhNode],
    rand: &mut Rand,
) -> Vec3 {
    let viewport = raytrace_settings.viewport;

    let sample_position = pixel_position.as_vec2() + pixel_sample_offset(rand);

    let pixel_center = viewport.upper_left_pixel_position
        + sample_position.x * viewport.horizontal_pixel_delta
        + sample_position.y * viewport.vertical_pixel_delta;

    let defocus_offset = defocus_sample_offset(rand);
    let ray_origin = viewport.origin
        + defocus_offset.x * viewport.horizontal_defocus_disk
        + defocus_offset.y * viewport.vertical_defocus_disk;

    let ray_direction = pixel_center - ray_origin;

    let ray = Ray {
        origin: ray_origin,
        direction: ray_direction,
    };

    ray_color(
        ray,
        spheres,
        bvh_nodes,
        raytrace_settings.use_bvh != 0,
        raytrace_settings.max_depth,
        rand,
    )
}

#[spirv(compute(threads(8, 8)))]
pub fn main(
    #[spirv(global_invocation_id)] pixel_position: UVec3,
    #[spirv(push_constant)] dispatch_params: &DispatchParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] raytrace_settings: &RaytraceSettings,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] spheres: &[Sphere],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] bvh_nodes: &[BvhNode],
) {
    let pixel_position = pixel_position.truncate();
    let screen_size = raytrace_settings.screen_size;

    if pixel_position.x >= screen_size.x || pixel_position.y >= screen_size.y {
        return;
    }

    let DispatchParams {
        seed,
        sample_start,
        sample_count,
        ..
    } = *dispatch_params;

    let mut color_sum = Vec3::ZERO;
    for sample in sample_start..sample_start + sample_count {
        let mut rand = Rand::from(pixel_position.extend(hash_combine2(seed, sample)));
        color_sum += trace_sample(
            pixel_position,
            raytrace_settings,
            spheres,
            bvh_nodes,
            &mut rand,
        );
    }

    output[(pixel_position.y * screen_size.x + pixel_position.x) as usize] +=
        (color_sum / (raytrace_settings.amount_of_samples as f32)).extend(0.);
}
