#![no_std]

mod data;
mod material;
mod rand;
mod ray;
mod sphere;

use data::{Range, RayHit};
use rand::Rand;
use ray::Ray;
use spirv_std::{
    glam::{UVec2, UVec3, Vec2, Vec3, Vec3Swizzles},
    num_traits::Float,
    spirv,
};

pub use material::{Material, Reflection};
pub use sphere::Sphere;

#[derive(Clone, Copy)]
struct Camera {
    position: Vec3,
    target: Vec3,
    up: Vec3,

    vertical_fov: f32,
    defocus_angle: f32,
    focus_distance: f32,
}

struct Viewport {
    upper_left_pixel_position: Vec3,

    horizontal_pixel_delta: Vec3,
    vertical_pixel_delta: Vec3,

    horizontal_defocus_disk: Vec3,
    vertical_defocus_disk: Vec3,
}

fn calculate_viewport(camera: Camera, screen_size: UVec2) -> Viewport {
    let aspect_ratio = (screen_size.x as f32) / (screen_size.y as f32);

    let h = Float::tan(camera.vertical_fov / 2.);
    let height = 2.0 * h * camera.focus_distance;
    let width = height * aspect_ratio;

    let w = (camera.position - camera.target).normalize();
    let u = Vec3::cross(camera.up, w).normalize();
    let v = Vec3::cross(w, u);

    let horizontal = width * u;
    let vertical = -height * v;

    let horizontal_pixel_delta = horizontal / (screen_size.x as f32);
    let vertical_pixel_delta = vertical / (screen_size.y as f32);

    let upper_left_corner =
        camera.position - (camera.focus_distance * w) - horizontal / 2. - vertical / 2.;

    let upper_left_pixel_position =
        upper_left_corner + horizontal_pixel_delta / 2. + vertical_pixel_delta / 2.;

    let defocus_radius = camera.focus_distance * Float::tan(camera.defocus_angle / 2.);
    let horizontal_defocus_disk = u * defocus_radius;
    let vertical_defocus_disk = v * defocus_radius;

    Viewport {
        upper_left_pixel_position,

        horizontal_pixel_delta,
        vertical_pixel_delta,

        horizontal_defocus_disk,
        vertical_defocus_disk,
    }
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

fn ray_color(ray: Ray, spheres: &[Sphere], max_depth: u32, rand: &mut Rand) -> Vec3 {
    let mut accumulated_color = Vec3::ONE;
    let mut next_ray = ray;

    for _ in 0..max_depth {
        let ray_hit = raytrace_spheres(spheres, next_ray, Range::new(0.001, Float::max_value()));

        if ray_hit.did_hit {
            let scatter_result = ray_hit.material.scatter(ray, ray_hit, rand);

            if scatter_result.did_scatter {
                accumulated_color *= scatter_result.attenuation;
                next_ray = scatter_result.scattered;
            } else {
                // Didn't scatter
                return accumulated_color;
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

#[spirv(compute(threads(1)))]
pub fn main(
    #[spirv(global_invocation_id)] pixel_position: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] &screen_size: &UVec2,
    #[spirv(uniform, descriptor_set = 0, binding = 1)] &max_depth: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] spheres: &[Sphere],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] output: &mut [Vec3],
) {
    let camera = Camera {
        position: Vec3::new(13., 2., 3.),
        target: Vec3::new(0., 0., 0.),
        up: Vec3::new(0., 1., 0.),

        vertical_fov: (20.).to_radians(),
        defocus_angle: (0.6).to_radians(),
        focus_distance: 10.,
    };

    let viewport = calculate_viewport(camera, screen_size);

    let mut rand = Rand::from(pixel_position);
    let sample_position = pixel_position.xy().as_vec2() + pixel_sample_offset(&mut rand);

    let pixel_center = viewport.upper_left_pixel_position
        + sample_position.x * viewport.horizontal_pixel_delta
        + sample_position.y * viewport.vertical_pixel_delta;

    let defocus_offset = defocus_sample_offset(&mut rand);
    let ray_origin = camera.position
        + defocus_offset.x * viewport.horizontal_defocus_disk
        + defocus_offset.y * viewport.vertical_defocus_disk;

    let ray_direction = pixel_center - ray_origin;

    let ray = Ray {
        origin: ray_origin,
        direction: ray_direction,
    };

    let color = ray_color(ray, spheres, max_depth, &mut rand);

    output[(pixel_position.y * screen_size.x + pixel_position.x) as usize] += color;
}
