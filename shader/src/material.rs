use bytemuck::{Pod, Zeroable};
use spirv_std::num_traits::Float;
use vek::Vec3;

use crate::{
    data::{Face, RayHit, ScatterResult},
    rand::Rand,
    ray::Ray,
    F,
};

#[derive(Clone, Copy, Default)]
#[repr(u32)]
pub enum Reflection {
    #[default]
    Diffuse,
    Metal,
    Glass,
}

unsafe impl Zeroable for Reflection {}
unsafe impl Pod for Reflection {}

#[derive(Clone, Copy, Default, Zeroable, Pod)]
#[repr(C)]
pub struct Material {
    pub reflection: Reflection,
    pub albedo: Vec3<F>,
    pub fuzz: F,
    pub refraction_index: F,
}

impl Material {
    pub fn diffuse(albedo: Vec3<F>) -> Self {
        Self {
            reflection: Reflection::Diffuse,
            albedo,
            ..Default::default()
        }
    }

    pub fn metal(albedo: Vec3<F>, fuzz: F) -> Self {
        Self {
            reflection: Reflection::Metal,
            albedo,
            fuzz,
            ..Default::default()
        }
    }

    pub fn glass(refraction_index: F) -> Self {
        Self {
            reflection: Reflection::Glass,
            refraction_index,
            ..Default::default()
        }
    }

    pub fn scatter(self, ray: Ray, ray_hit: RayHit, rand: &mut Rand) -> ScatterResult {
        match self.reflection {
            Reflection::Diffuse => scatter_diffuse(self.albedo.into(), ray_hit, rand),
            Reflection::Metal => scatter_metal(self.albedo.into(), self.fuzz, ray, ray_hit, rand),
            Reflection::Glass => scatter_glass(self.refraction_index, ray, ray_hit, rand),
        }
    }
}

fn is_near_zero(value: Vec3<F>) -> bool {
    const S: F = 1e-8;

    value.x < S && value.y < S && value.z < S
}

fn reflect(value: Vec3<F>, normal: Vec3<F>) -> Vec3<F> {
    value - (2. * Vec3::dot(value, normal) * normal)
}

fn refract(value: Vec3<F>, normal: Vec3<F>, etai_over_etat: F) -> Vec3<F> {
    let cos_theta = Float::min(Vec3::dot(-value, normal), 1.);

    let r_out_perp = etai_over_etat * (value + (cos_theta * normal));
    let r_out_parallel = -Float::sqrt(Float::abs(1. - r_out_perp.magnitude_squared())) * normal;

    r_out_perp + r_out_parallel
}

fn reflectance(cosine: F, refraction_ratio: F) -> F {
    let r0 = (1. - refraction_ratio) / (1. + refraction_ratio);
    let r0 = r0 * r0;

    r0 + (1. - r0) * Float::powi(1. - cosine, 5)
}

fn scatter_diffuse(albedo: Vec3<F>, ray_hit: RayHit, rand: &mut Rand) -> ScatterResult {
    let mut scatter_direction = ray_hit.normal + rand.gen_unit_vector();

    // Catch degenerate scatter direction
    if is_near_zero(scatter_direction) {
        scatter_direction = ray_hit.normal;
    }

    let scattered = Ray {
        origin: ray_hit.point,
        direction: scatter_direction,
    };
    let attenuation = albedo;

    ScatterResult {
        did_scatter: true,
        scattered,
        attenuation,
    }
}

pub fn scatter_metal(
    albedo: Vec3<F>,
    fuzz: F,
    ray: Ray,
    ray_hit: RayHit,
    rand: &mut Rand,
) -> ScatterResult {
    let reflected = reflect(ray.direction.normalized(), ray_hit.normal);

    let scattered = Ray {
        origin: ray_hit.point,
        direction: reflected + rand.gen_unit_vector() * fuzz,
    };
    let attenuation = albedo;

    if scattered.direction.dot(ray_hit.normal) > 0. {
        ScatterResult {
            did_scatter: true,
            scattered,
            attenuation,
        }
    } else {
        ScatterResult::none()
    }
}

pub fn scatter_glass(
    refraction_index: F,
    ray: Ray,
    ray_hit: RayHit,
    rand: &mut Rand,
) -> ScatterResult {
    let refraction_index = 1.5;
    let refraction_ratio = match ray_hit.face {
        Face::Front => 1. / refraction_index,
        Face::Back => refraction_index,
    };

    let unit_direction = ray.direction.normalized();
    let cos_theta = Float::min(Vec3::dot(-unit_direction, ray_hit.normal), 1.);
    let sin_theta = Float::sqrt(1. - cos_theta * cos_theta);

    let cannot_refract = (refraction_ratio * sin_theta > 1.)
        || (reflectance(cos_theta, refraction_ratio) > rand.gen_float());

    let direction = if cannot_refract {
        reflect(unit_direction, ray_hit.normal)
    } else {
        refract(unit_direction, ray_hit.normal, refraction_ratio)
    };

    let scattered = Ray {
        origin: ray_hit.point,
        direction,
    };
    let attenuation = Vec3::one();

    ScatterResult {
        did_scatter: true,
        scattered,
        attenuation,
    }
}
