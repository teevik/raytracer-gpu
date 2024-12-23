use core::mem::swap;

use spirv_std::num_traits::Float;
use vek::Vec3;

use crate::{
    data::{Range, RayHit},
    ray::Ray,
    traits::Raycastable,
};

#[derive(Clone, Copy)]
pub struct Aabb {
    pub axes: Vec3<Range>,
}

impl Aabb {
    pub fn from_extremes(a: Vec3<f32>, b: Vec3<f32>) -> Self {
        let x = Range::new(Float::min(a.x, b.x), Float::max(a.x, b.x));
        let y = Range::new(Float::min(a.y, b.y), Float::max(a.y, b.y));
        let z = Range::new(Float::min(a.z, b.z), Float::max(a.z, b.z));

        Self {
            axes: Vec3::new(x, y, z),
        }
    }

    pub fn combine(a: Self, b: Self) -> Self {
        let x = Range::combine(a.axes.x, b.axes.x);
        let y = Range::combine(a.axes.y, b.axes.y);
        let z = Range::combine(a.axes.z, b.axes.z);

        Self {
            axes: Vec3::new(x, y, z),
        }
    }

    pub fn raycast(self, ray: Ray, range: Range) -> bool {
        let mut range = range;

        for axis in 0..3 {
            let inverse_direction = 1. / ray.direction[axis];
            let origin = ray.origin[axis];

            let mut t0 = (self.axes[axis].min - origin) * inverse_direction;
            let mut t1 = (self.axes[axis].max - origin) * inverse_direction;

            if inverse_direction < 0. {
                swap(&mut t0, &mut t1);
            }

            range.min = Float::max(t0, range.min);
            range.max = Float::min(t1, range.max);

            if range.max <= range.min {
                return false;
            }
        }

        true
    }
}

pub struct BvhIndex(usize);

impl BvhIndex {
    pub fn left(self) -> Self {
        Self((2 * self.0) + 1)
    }

    pub fn right(self) -> Self {
        Self((2 * self.0) + 2)
    }
}

#[derive(Clone, Copy)]
pub struct BvhNode<T: Copy> {
    bounding_box: Aabb,
    value: T,
}

#[derive(Clone, Copy)]
pub struct Bvh<'a, T: Copy> {
    nodes: &'a [BvhNode<T>],
}

impl<'a, T: Copy> Bvh<'a, T> {
    pub fn root(self) -> BvhIndex {
        BvhIndex(0)
    }

    pub fn contains(self, index: BvhIndex) -> bool {
        index.0 < self.nodes.len()
    }

    pub fn at(self, index: BvhIndex) -> BvhNode<T> {
        self.nodes[index.0]
    }
}

impl<'a, T: Raycastable + Copy> Raycastable for Bvh<'a, T> {
    fn raycast(self, ray: Ray, range: Range) -> RayHit {
        if self.nodes.is_empty() {
            return RayHit::none();
        }

        let mut current = self.root();

        if self.contains(current.left()) {
            // let ray_hit = self.at(current.left()).raycast(ray, )
        }

        RayHit::none()
    }
}
