@group(0)
@binding(0)
var<storage, read_write> data: array<f32>;

@group(0)
@binding(1)
var<storage, read> screen_size: vec2<u32>;

@group(0)
@binding(2)
var<storage, read_write> output: array<vec3<f32>>;

struct Ray {
  origin: vec3<f32>,
  direction: vec3<f32>
}

struct Sphere {
  center: vec3<f32>,
  radius: f32
}

fn hit_sphere(sphere: Sphere, ray: Ray) -> bool {
  let oc = ray.origin - sphere.center;
  let a = dot(ray.direction, ray.direction);
  let b = 2. * dot(oc, ray.direction);
  let c = dot(oc, oc) - sphere.radius * sphere.radius;

  let discriminant = b*b - 4. * a * c;

  return discriminant >= 0.;
}

fn ray_color(ray: Ray) -> vec3<f32> {
  if (hit_sphere(Sphere(vec3(0., 0., -1.), 0.5), ray)) {
    return vec3(1., 0., 0.);
  }

  let unit_direction = normalize(ray.direction);
  let a = (unit_direction.y + 1.) / 2.;
  return vec3(1. - a) + a * vec3(0.5, 0.7, 1.);
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) pixel_position: vec3<u32>) {
  let vertical_fov = 30.;
  let camera_position = vec3(0., 0., 0.);

  let aspect_ratio = f32(screen_size.x) / f32(screen_size.y);
  let theta = radians(vertical_fov);
  
  let h = theta * 2.;
  let viewport_height = 2. * h;
  let viewport_width = viewport_height * aspect_ratio;
  let focal_length = 1.;

  let horizontal_viewport = vec3(viewport_width, 0., 0.);
  let vertical_viewport = vec3(0., -viewport_height, 0.);
  let horizontal_pixel_delta = horizontal_viewport / f32(screen_size.x);
  let vertical_pixel_delta = vertical_viewport / f32(screen_size.y);

  let upper_left_corner = camera_position
    - vec3(0., 0., focal_length)
    - horizontal_viewport/2.
    - vertical_viewport/2.;

  let first_pixel = upper_left_corner + horizontal_pixel_delta / 2. + vertical_pixel_delta / 2.;

  let pixel_center = first_pixel 
    + f32(pixel_position.x) * horizontal_pixel_delta
    + f32(pixel_position.y) * vertical_pixel_delta;
  
  let ray_direction = pixel_center - camera_position;

  let ray = Ray(camera_position, ray_direction);
  
  output[pixel_position.y * screen_size.x + pixel_position.x] = ray_color(ray);

   // output[pixel_position.y * screen_size.x + pixel_position.x] = vec3(
   //  f32(pixel_position.x) / f32(screen_size.x),
   //  f32(pixel_position.y) / f32(screen_size.y),
   //  0.
  // );
}
