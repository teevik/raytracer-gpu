use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use spirv_std::num_traits::Float;

use crate::{
    data::{Closest, Range},
    ray::Ray,
    sphere::Sphere,
};

/// Axis-aligned bounding box. Each axis spans `min[axis]..max[axis]`.
#[derive(Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn from_extremes(a: Vec3, b: Vec3) -> Self {
        Self {
            min: a.min(b),
            max: a.max(b),
        }
    }
}

/// A ray prepared for repeated slab tests: `inverse_direction` is computed once per ray.
#[derive(Clone, Copy)]
pub struct SlabRay {
    pub origin: Vec3,
    pub inverse_direction: Vec3,
}

impl SlabRay {
    pub fn new(ray: Ray) -> Self {
        // A zero direction component would give `inf` here and `0 * inf = NaN` in the slab test
        // when the origin lies exactly on a slab plane. Nudging it to a tiny same-signed value
        // keeps every product finite (|slab| * 1e30 stays far below `f32::MAX`) while still
        // behaving like an axis-parallel ray.
        fn safe(direction: f32) -> f32 {
            const EPSILON: f32 = 1e-30;

            if Float::abs(direction) < EPSILON {
                if direction < 0. { -EPSILON } else { EPSILON }
            } else {
                direction
            }
        }

        let direction = ray.direction;

        Self {
            origin: ray.origin,
            inverse_direction: Vec3::ONE
                / Vec3::new(safe(direction.x), safe(direction.y), safe(direction.z)),
        }
    }

    /// Slab test: does the ray overlap `min..max` anywhere inside `range`?
    pub fn hits(self, min: Vec3, max: Vec3, range: Range) -> bool {
        let t0 = (min - self.origin) * self.inverse_direction;
        let t1 = (max - self.origin) * self.inverse_direction;

        let entry = t0.min(t1).max_element();
        let exit = t0.max(t1).min_element();

        Float::max(entry, range.min) <= Float::min(exit, range.max)
    }
}

/// Element of the BVH storage buffer, shared with the host.
///
/// Nodes are stored depth first, so an interior node's first child is always the next node.
/// Traversal is stackless: visit `index + 1` when the box is hit (or after a leaf), and jump to
/// `exit_index` (the node after this node's whole subtree) when it is missed. Each `Vec3` is
/// followed by a `u32` so the Rust layout matches std430 (32 bytes, 16-byte aligned fields).
#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct BvhNode {
    pub min: Vec3,
    /// Index of the first node after this node's subtree.
    pub exit_index: u32,
    pub max: Vec3,
    /// `sphere_index + 1` for a leaf (one sphere per leaf), `0` for an interior node.
    pub sphere: u32,
}

const _: () = {
    assert!(size_of::<BvhNode>() == 32);
    assert!(core::mem::offset_of!(BvhNode, exit_index) == 12);
    assert!(core::mem::offset_of!(BvhNode, max) == 16);
    assert!(core::mem::offset_of!(BvhNode, sphere) == 28);
};

impl BvhNode {
    pub fn interior(bounds: Aabb, exit_index: u32) -> Self {
        Self {
            min: bounds.min,
            exit_index,
            max: bounds.max,
            sphere: 0,
        }
    }

    pub fn leaf(bounds: Aabb, exit_index: u32, sphere_index: u32) -> Self {
        Self {
            min: bounds.min,
            exit_index,
            max: bounds.max,
            sphere: sphere_index + 1,
        }
    }
}

/// Finds the closest sphere along `ray` within `range` by walking the BVH.
pub fn closest_sphere(nodes: &[BvhNode], spheres: &[Sphere], ray: Ray, range: Range) -> Closest {
    let slab_ray = SlabRay::new(ray);
    let mut closest = Closest::none(range.max);

    let node_count = nodes.len() as u32;
    let mut index = 0;

    while index < node_count {
        let node = nodes[index as usize];

        if node.sphere != 0 {
            // Leaf: the sphere test is about as cheap as its box test, so skip the box. A leaf's
            // subtree is just itself, so the next node is always `index + 1`.
            let sphere_index = node.sphere - 1;
            let distance = spheres[sphere_index as usize]
                .intersect_distance(ray, Range::new(range.min, closest.distance));

            if distance < closest.distance {
                closest = Closest {
                    index: sphere_index,
                    distance,
                };
            }

            index += 1;
        } else if slab_ray.hits(node.min, node.max, Range::new(range.min, closest.distance)) {
            index += 1;
        } else {
            index = node.exit_index;
        }
    }

    closest
}
