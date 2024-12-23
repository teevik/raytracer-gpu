use vek::{Vec2, Vec3};

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

pub fn hash2(v: Vec2<u32>) -> u32 {
    hash_combine2(v.x, hash1(v.y))
}

pub fn hash3(v: Vec3<u32>) -> u32 {
    hash_combine2(v.x, hash2(Vec2::new(v.y, v.z)))
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
    pub fn new(seed: u32) -> Rand {
        Self { current: seed }
    }

    pub fn gen(&mut self) -> u32 {
        let next = hash1(self.current);
        self.current = next;

        next
    }

    pub fn gen_float(&mut self) -> f32 {
        let rand = self.gen();

        uint_to_u01_float(rand)
    }

    pub fn gen_range(&mut self, range: Range) -> f32 {
        let rand = self.gen_float();

        range.min + (rand * (range.max - range.min))
    }

    pub fn gen_vec2(&mut self) -> Vec2<f32> {
        Vec2::new(self.gen_float(), self.gen_float())
    }

    pub fn gen_in_unit_sphere(&mut self) -> Vec3<f32> {
        let mut random = || self.gen_range(Range { min: -1., max: 1. });

        loop {
            let sample = Vec3::new(random(), random(), random());

            if sample.magnitude_squared() < 1. {
                break sample;
            }
        }
    }

    pub fn gen_unit_vector(&mut self) -> Vec3<f32> {
        self.gen_in_unit_sphere().normalized()
    }

    pub fn gen_on_hemisphere(&mut self, normal: Vec3<f32>) -> Vec3<f32> {
        let on_unit_sphere = self.gen_unit_vector();

        if on_unit_sphere.dot(normal) > 0. {
            on_unit_sphere
        } else {
            -on_unit_sphere
        }
    }

    pub fn gen_in_unit_disk(&mut self) -> Vec2<f32> {
        let mut random = || self.gen_range(Range::new(-1., 1.));

        loop {
            let sample = Vec2::new(random(), random());

            if sample.magnitude_squared() < 1. {
                break sample;
            }
        }
    }
}

impl From<Vec2<u32>> for Rand {
    fn from(value: Vec2<u32>) -> Self {
        Self::new(hash2(value))
    }
}

impl From<Vec3<u32>> for Rand {
    fn from(value: Vec3<u32>) -> Self {
        Self::new(hash3(value))
    }
}
