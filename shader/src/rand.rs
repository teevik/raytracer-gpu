use spirv_std::glam::{UVec2, UVec3, Vec2, Vec3Swizzles};

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

        uint_to_u01_float(rand)
    }

    pub fn gen_range(&mut self, range: Range<f32>) -> f32 {
        let rand = self.gen_float();

        range.start + (rand * (range.end - range.start))
    }

    pub fn gen_vec2(&mut self) -> Vec2 {
        Vec2::new(self.gen_float(), self.gen_float())
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
