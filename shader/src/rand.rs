use core::f32::consts::TAU;

use glam::{UVec2, UVec3, Vec2, Vec3};
use spirv_std::num_traits::Float;

use crate::data::Range;

pub fn hash1(mut x: u32) -> u32 {
    x += x << 10;
    x ^= x >> 6;
    x += x << 3;
    x ^= x >> 11;
    x += x << 15;
    x
}

pub fn hash_combine2(x: u32, y: u32) -> u32 {
    const M: u32 = 1664525;
    const C: u32 = 1013904223;
    let mut seed = (x * M + y + C) * M;
    // Tempering (from Matsumoto)
    seed ^= seed >> 11;
    seed ^= (seed << 7) & 0x9d2c5680;
    seed ^= (seed << 15) & 0xefc60000;
    seed ^= seed >> 18;
    seed
}

pub fn hash2(v: UVec2) -> u32 {
    hash_combine2(v.x, hash1(v.y))
}

pub fn hash3(v: UVec3) -> u32 {
    hash_combine2(v.x, hash2(UVec2::new(v.y, v.z)))
}

pub fn uint_to_u01_float(h: u32) -> f32 {
    const MANTISSA_MASK: u32 = 0x007FFFFF;
    const ONE: u32 = 0x3F800000;
    f32::from_bits((h & MANTISSA_MASK) | ONE) - 1.0
}

pub struct Rand {
    current: u32,
}

impl Rand {
    /// `hash1` is a bijection with exactly two fixed points, 0 and 4016952291. A generator seeded
    /// with either would return the same value forever. The samplers are loop-free, so that can no
    /// longer hang the GPU, but the pixel-sample would get a constant "random" stream (every
    /// bounce scattering the same way). Seeds are hashes, so this happens ~2^-31 per pixel-sample.
    /// Nudge those seeds off the fixed point.
    pub fn new(seed: u32) -> Rand {
        const FIXED_POINTS: [u32; 2] = [0, 4016952291];
        let current = if seed == FIXED_POINTS[0] || seed == FIXED_POINTS[1] {
            seed ^ 0x9E37_79B9
        } else {
            seed
        };
        Self { current }
    }

    pub fn next_u32(&mut self) -> u32 {
        let next = hash1(self.current);
        self.current = next;

        next
    }

    pub fn gen_float(&mut self) -> f32 {
        let rand = self.next_u32();

        uint_to_u01_float(rand)
    }

    pub fn gen_range(&mut self, range: Range) -> f32 {
        let rand = self.gen_float();

        range.min + (rand * (range.max - range.min))
    }

    pub fn gen_vec2(&mut self) -> Vec2 {
        Vec2::new(self.gen_float(), self.gen_float())
    }

    /// Uniform on the unit sphere. Archimedes: `z` is uniform on [-1, 1] and the azimuth is
    /// independent and uniform, so no rejection loop is needed.
    pub fn gen_unit_vector(&mut self) -> Vec3 {
        let z = 1. - 2. * self.gen_float();
        let r = Float::sqrt(Float::max(0., 1. - z * z));
        let phi = TAU * self.gen_float();

        Vec3::new(r * Float::cos(phi), r * Float::sin(phi), z)
    }

    /// Uniform in the unit disk: `r = sqrt(u)` makes the area density constant.
    pub fn gen_in_unit_disk(&mut self) -> Vec2 {
        let r = Float::sqrt(self.gen_float());
        let theta = TAU * self.gen_float();

        Vec2::new(r * Float::cos(theta), r * Float::sin(theta))
    }
}

impl From<UVec2> for Rand {
    fn from(value: UVec2) -> Self {
        Self::new(hash2(value))
    }
}

impl From<UVec3> for Rand {
    fn from(value: UVec3) -> Self {
        Self::new(hash3(value))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use core::f32::consts::PI;
    use std::{println, vec::Vec};

    use super::*;

    const SEEDS: [u32; 3] = [1, 2, 0xDEAD_BEEF];
    const PER_SEED: u32 = 100_000;
    const BINS: usize = 10;

    /// One fresh generator per simulated pixel-sample, seeded exactly like the shader does, with
    /// two draws from each: this is the stream the samplers actually see on the GPU.
    fn samples<T>(mut draw: impl FnMut(&mut Rand) -> T) -> Vec<T> {
        let mut out = Vec::new();
        for seed in SEEDS {
            for i in 0..PER_SEED {
                let pixel = UVec2::new(i % 800, i / 800);
                let mut rand = Rand::from(pixel.extend(hash_combine2(seed, i)));
                out.push(draw(&mut rand));
                out.push(draw(&mut rand));
            }
        }
        out
    }

    fn mean(values: impl Iterator<Item = f32>) -> f64 {
        let (sum, n) = values.fold((0f64, 0usize), |(s, n), v| (s + v as f64, n + 1));
        sum / n as f64
    }

    /// Asserts `values` (all in [min, max]) fill `BINS` equal bins with 1/BINS each, ±1.5%.
    fn assert_uniform(name: &str, values: impl Iterator<Item = f32>, min: f32, max: f32) {
        let mut counts = [0usize; BINS];
        let mut n = 0;
        for v in values {
            let t = (v - min) / (max - min);
            counts[((t * BINS as f32) as usize).min(BINS - 1)] += 1;
            n += 1;
        }
        let fractions = counts.map(|c| c as f64 / n as f64);
        println!("{name} bins: {fractions:.4?}");
        for f in fractions {
            assert!((f - 0.1).abs() < 0.015, "{name} not uniform: {fractions:?}");
        }
    }

    #[test]
    fn unit_vector_is_uniform_on_sphere() {
        let v = samples(Rand::gen_unit_vector);
        assert!(v.len() >= 200_000);

        let max_len_err = v.iter().map(|p| (p.length() - 1.).abs()).fold(0., f32::max);
        println!("n = {}, max |len - 1| = {max_len_err:e}", v.len());
        assert!(max_len_err < 1e-4);

        for (axis, name) in ["x", "y", "z"].into_iter().enumerate() {
            let m1 = mean(v.iter().map(|p| p[axis]));
            let m2 = mean(v.iter().map(|p| p[axis] * p[axis]));
            println!("{name}: mean = {m1:+.5}, E[{name}^2] = {m2:.5} (1/3 = 0.33333)");
            assert!(m1.abs() < 0.01);
            assert!((m2 - 1. / 3.).abs() < 0.01);
            // Archimedes: every coordinate of a uniform sphere point is uniform on [-1, 1].
            assert_uniform(name, v.iter().map(|p| p[axis]), -1., 1.);
        }
        assert_uniform("azimuth", v.iter().map(|p| p.y.atan2(p.x)), -PI, PI);
    }

    #[test]
    fn in_unit_disk_is_uniform() {
        let v = samples(Rand::gen_in_unit_disk);
        assert!(v.len() >= 200_000);

        let max_r = v.iter().map(|p| p.length()).fold(0., f32::max);
        println!("n = {}, max r = {max_r}", v.len());
        assert!(max_r < 1.);

        let mean_x = mean(v.iter().map(|p| p.x));
        let mean_y = mean(v.iter().map(|p| p.y));
        let mean_r2 = mean(v.iter().map(|p| p.length_squared()));
        println!("mean x = {mean_x:+.5}, mean y = {mean_y:+.5}, E[r^2] = {mean_r2:.5} (0.5)");
        assert!(mean_x.abs() < 0.01 && mean_y.abs() < 0.01);
        assert!((mean_r2 - 0.5).abs() < 0.01);

        // Uniform area density means r^2 is uniform on [0, 1].
        assert_uniform("r^2", v.iter().map(|p| p.length_squared()), 0., 1.);
        assert_uniform("angle", v.iter().map(|p| p.y.atan2(p.x)), -PI, PI);
    }

    #[test]
    fn samplers_are_finite_from_fixed_point_seeds() {
        for seed in [0, 4016952291] {
            let mut rand = Rand::new(seed);
            let first = rand.next_u32();
            assert_ne!(first, rand.next_u32(), "seed {seed} still stuck");
            assert!(rand.gen_unit_vector().is_finite());
            assert!(rand.gen_in_unit_disk().is_finite());
        }
    }
}
