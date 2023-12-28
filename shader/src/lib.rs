#![no_std]

mod data;
mod rand;
mod ray;
mod sphere;

use data::{Face, Range};
use rand::Rand;
use ray::Ray;
use sphere::Sphere;
use spirv_std::{
    glam::{UVec2, UVec3, Vec2, Vec3, Vec3Swizzles},
    num_traits::Float,
    spirv,
};

#[derive(Clone, Default)]
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
    // /// The material of the hit shape
    // pub material: Material,
}

impl RayHit {
    pub fn none() -> Self {
        Default::default()
    }
}

#[derive(Clone, Copy)]
struct Camera {
    position: Vec3,
    vertical_fov: f32,
    focal_length: f32,
}

struct Viewport {
    horizontal_pixel_delta: Vec3,
    vertical_pixel_delta: Vec3,

    upper_left_pixel_position: Vec3,
}

fn calculate_viewport(camera: Camera, screen_size: UVec2) -> Viewport {
    let aspect_ratio = (screen_size.x as f32) / (screen_size.y as f32);

    let height = 2.0;
    let width = height * aspect_ratio;

    let horizontal = Vec3::new(width, 0., 0.);
    let vertical = Vec3::new(0., -height, 0.);

    let horizontal_pixel_delta = horizontal / (screen_size.x as f32);
    let vertical_pixel_delta = vertical / (screen_size.y as f32);

    let upper_left_corner =
        camera.position - Vec3::new(0., 0., camera.focal_length) - horizontal / 2. - vertical / 2.;

    let upper_left_pixel_position =
        upper_left_corner + horizontal_pixel_delta / 2. + vertical_pixel_delta / 2.;

    Viewport {
        horizontal_pixel_delta,
        vertical_pixel_delta,
        upper_left_pixel_position,
    }
}

fn raytrace_spheres<const N: usize>(spheres: [Sphere; N], ray: Ray, range: Range<f32>) -> RayHit {
    let mut closest_hit = RayHit::none();
    let mut closest_distance = range.end;

    for i in 0..N {
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

fn ray_color<const N: usize>(ray: Ray, spheres: [Sphere; N]) -> Vec3 {
    let ray_hit = raytrace_spheres(spheres, ray, Range::new(0., Float::max_value()));

    if ray_hit.did_hit {
        return (ray_hit.normal + Vec3::ONE) / 2.;
        // let normal = ray.at(ray_hit.distance) - Vec3::new(0., 0., -1.);
        // let normal = normal.normalize();
        //
        // return (normal + Vec3::ONE) / 2.;
    }

    let unit_direction = ray.direction.normalize();
    let a = (unit_direction.y + 1.) / 2.;
    return Vec3::splat(1. - a) + a * Vec3::new(0.5, 0.7, 1.);
}

fn pixel_sample_offset(rand: &mut Rand) -> Vec2 {
    rand.gen_vec2() - (Vec2::ONE / 2.) // From -0.5 to 0.5
}

#[spirv(compute(threads(1)))]
pub fn main(
    #[spirv(global_invocation_id)] pixel_position: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] &screen_size: &UVec2,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [Vec3],
) {
    let spheres = [
        Sphere {
            center: Vec3::new(0., 0., -1.),
            radius: 0.5,
        },
        Sphere {
            center: Vec3::new(0., -100.5, -1.),
            radius: 100.,
        },
    ];

    let camera = Camera {
        position: Vec3::new(0., 0., 0.),
        vertical_fov: 30_f32.to_radians(),
        focal_length: 1.,
    };

    let viewport = calculate_viewport(camera, screen_size);

    let mut rand = Rand::from(pixel_position);
    let sample_position = pixel_position.xy().as_vec2() + pixel_sample_offset(&mut rand);

    let pixel_center = viewport.upper_left_pixel_position
        + sample_position.x * viewport.horizontal_pixel_delta
        + sample_position.y * viewport.vertical_pixel_delta;

    let ray_direction = pixel_center - camera.position;

    let ray = Ray {
        origin: camera.position,
        direction: ray_direction,
    };

    let color = ray_color(ray, spheres);

    output[(pixel_position.y * screen_size.x + pixel_position.x) as usize] += color;
}
