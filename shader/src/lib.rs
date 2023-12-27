#![no_std]

use spirv_std::{
    glam::{UVec2, UVec3, Vec3},
    spirv,
};

#[derive(Clone, Copy)]
struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn at(self, distance: f32) -> Vec3 {
        self.origin + (self.direction * distance)
    }
}

struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

fn hit_sphere(sphere: Sphere, ray: Ray) -> bool {
    let oc = ray.origin - sphere.center;
    let a = Vec3::dot(ray.direction, ray.direction);
    let b = 2. * Vec3::dot(oc, ray.direction);
    let c = Vec3::dot(oc, oc) - sphere.radius * sphere.radius;

    let discriminant = b * b - 4. * a * c;

    discriminant >= 0.
}

fn ray_color(ray: Ray) -> Vec3 {
    let sphere = Sphere {
        center: Vec3::new(0., 0., -1.),
        radius: 0.5,
    };

    if hit_sphere(sphere, ray) {
        return Vec3::new(1., 0., 0.);
    }

    let unit_direction = ray.direction.normalize();
    let a = (unit_direction.y + 1.) / 2.;
    return Vec3::splat(1. - a) + a * Vec3::new(0.5, 0.7, 1.);
}

#[spirv(compute(threads(1)))]
pub fn main(
    #[spirv(global_invocation_id)] pixel_position: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] _input: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 1)] screen_size: &UVec2,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [Vec3],
) {
    let vertical_fov: f32 = 30.;
    let camera_position = Vec3::new(0., 0., 0.);

    let aspect_ratio = (screen_size.x as f32) / (screen_size.y as f32);
    let theta = vertical_fov.to_radians();

    let h = theta * 2.;
    let viewport_height = 2. * h;
    let viewport_width = viewport_height * aspect_ratio;
    let focal_length = 1.;

    let horizontal_viewport = Vec3::new(viewport_width, 0., 0.);
    let vertical_viewport = Vec3::new(0., -viewport_height, 0.);
    let horizontal_pixel_delta = horizontal_viewport / (screen_size.x as f32);
    let vertical_pixel_delta = vertical_viewport / (screen_size.y as f32);

    let upper_left_corner = camera_position
        - Vec3::new(0., 0., focal_length)
        - horizontal_viewport / 2.
        - vertical_viewport / 2.;

    let first_pixel = upper_left_corner + horizontal_pixel_delta / 2. + vertical_pixel_delta / 2.;

    let pixel_center = first_pixel
        + (pixel_position.x as f32) * horizontal_pixel_delta
        + (pixel_position.y as f32) * vertical_pixel_delta;

    let ray_direction = pixel_center - camera_position;

    let ray = Ray {
        origin: camera_position,
        direction: ray_direction,
    };

    let color = ray_color(ray);

    output[(pixel_position.y * screen_size.x + pixel_position.x) as usize] = color;
}
