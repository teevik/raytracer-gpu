use spirv_std::glam::{UVec2, UVec3, Vec2, Vec3, Vec3Swizzles};

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
    hash_combine2(v.x, hash2(v.yz()))
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

        // (rand as f32) / (u32::MAX as f32)
        uint_to_u01_float(rand)
    }

    pub fn gen_range(&mut self, range: Range<f32>) -> f32 {
        let rand = self.gen_float();

        range.start + (rand * (range.end - range.start))
    }

    pub fn gen_vec2(&mut self) -> Vec2 {
        Vec2::new(self.gen_float(), self.gen_float())
    }

    pub fn gen_in_unit_sphere(&mut self) -> Vec3 {
        let mut random = || {
            self.gen_range(Range {
                start: -1.,
                end: 1.,
            })
        };

        loop {
            let sample = Vec3::new(random(), random(), random());

            if sample.length_squared() < 1. {
                break sample;
            }
        }
    }

    pub fn gen_unit_vector(&mut self) -> Vec3 {
        self.gen_in_unit_sphere().normalize()
    }

    pub fn gen_on_hemisphere(&mut self, normal: Vec3) -> Vec3 {
        let on_unit_sphere = self.gen_unit_vector();

        if on_unit_sphere.dot(normal) > 0. {
            on_unit_sphere
        } else {
            -on_unit_sphere
        }
    }

    pub fn gen_in_unit_disk(&mut self) -> Vec2 {
        let mut random = || self.gen_range(Range::new(-1., 1.));

        loop {
            let sample = Vec2::new(random(), random());

            if sample.length_squared() < 1. {
                break sample;
            }
        }
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
