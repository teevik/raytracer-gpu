#![no_std]

mod data;
mod material;
mod rand;
mod ray;
mod sphere;

use bytemuck::{Pod, Zeroable};
use data::{Range, RayHit};
use rand::Rand;
use ray::Ray;
use spirv_std::{glam, num_traits::Float, spirv};
use vek::{Vec2, Vec3};

pub use glam::UVec3;
pub use material::{Material, Reflection};
pub use sphere::Sphere;

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct RaytraceSettings {
    pub viewport: Viewport,
    pub screen_size: Vec2<u32>,
    pub amount_of_samples: u32,
    pub max_depth: u32,
}

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct Viewport {
    pub origin: Vec3<f32>,
    pub upper_left_pixel_position: Vec3<f32>,

    pub horizontal_pixel_delta: Vec3<f32>,
    pub vertical_pixel_delta: Vec3<f32>,

    pub horizontal_defocus_disk: Vec3<f32>,
    pub vertical_defocus_disk: Vec3<f32>,
}

fn raytrace_spheres(spheres: &[Sphere], ray: Ray, range: Range<f32>) -> RayHit {
    let mut closest_hit = RayHit::none();
    let mut closest_distance = range.end;

    for i in 0..spheres.len() {
        let ray_hit = spheres[i].raytrace(
            ray,
            Range {
                start: range.start,
                end: closest_distance,
            },
        );

        if ray_hit.did_hit {
            closest_distance = ray_hit.distance;
            closest_hit = ray_hit;
        }
    }

    closest_hit
}

fn ray_color(ray: Ray, spheres: &[Sphere], max_depth: u32, rand: &mut Rand) -> Vec3<f32> {
    let mut accumulated_color = Vec3::one();
    let mut next_ray = ray;

    for _ in 0..max_depth {
        let ray_hit = raytrace_spheres(spheres, next_ray, Range::new(0.001, Float::max_value()));

        if ray_hit.did_hit {
            let scatter_result = ray_hit.material.scatter(next_ray, ray_hit, rand);

            if scatter_result.did_scatter {
                accumulated_color *= scatter_result.attenuation;
                next_ray = scatter_result.scattered;
            } else {
                // Didn't scatter
                return Vec3::zero();
            }
        } else {
            // Didn't hit anything
            let unit_direction = next_ray.direction.normalized();
            let a = (unit_direction.y + 1.) / 2.;
            let background_color = Vec3::broadcast(1. - a) + a * Vec3::new(0.5, 0.7, 1.);

            accumulated_color *= background_color;

            return accumulated_color;
        }
    }

    // Reached max depth
    Vec3::zero()
}

fn pixel_sample_offset(rand: &mut Rand) -> Vec2<f32> {
    rand.gen_vec2() - (Vec2::one() / 2.) // From -0.5 to 0.5
}

fn defocus_sample_offset(rand: &mut Rand) -> Vec2<f32> {
    rand.gen_in_unit_disk()
}

#[spirv(compute(threads(1)))]
pub fn main(
    #[spirv(global_invocation_id)] pixel_position: glam::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] &seed: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] &raytrace_settings: &RaytraceSettings,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] spheres: &[Sphere],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] output: &mut [Vec3<f32>],
) {
    let pixel_position = Vec2::new(pixel_position.x, pixel_position.y);

    let RaytraceSettings {
        viewport,
        screen_size,
        amount_of_samples,
        max_depth,
    } = raytrace_settings;

    let mut rand = Rand::from(pixel_position.with_z(seed));
    let sample_position = pixel_position.as_::<f32>() + pixel_sample_offset(&mut rand);

    let pixel_center = viewport.upper_left_pixel_position
        + sample_position.x * viewport.horizontal_pixel_delta
        + sample_position.y * viewport.vertical_pixel_delta;

    let defocus_offset = defocus_sample_offset(&mut rand);
    let ray_origin = viewport.origin
        + defocus_offset.x * viewport.horizontal_defocus_disk
        + defocus_offset.y * viewport.vertical_defocus_disk;

    let ray_direction = pixel_center - ray_origin;

    let ray = Ray {
        origin: ray_origin,
        direction: ray_direction,
    };

    let color = ray_color(ray, spheres, max_depth, &mut rand);

    output[(pixel_position.y * screen_size.x + pixel_position.x) as usize] +=
        color / (amount_of_samples as f32);
}
