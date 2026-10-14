use std::time::Duration;

use glam::Vec3;
use obvhs::{
    BvhBuildParams,
    aabb::Aabb as ObvhsAabb,
    bvh2::{Bvh2, builder::build_bvh2},
};
use shader::{Aabb, BvhNode, Sphere};

/// Builds a BVH over `spheres` with one sphere per leaf, flattened into the layout the shader walks.
pub fn build_bvh(spheres: &[Sphere]) -> Vec<BvhNode> {
    if spheres.is_empty() {
        return Vec::new();
    }

    let aabbs: Vec<ObvhsAabb> = spheres
        .iter()
        .map(|sphere| {
            let Aabb { min, max } = sphere.get_aabb();
            ObvhsAabb::new(min.into(), max.into())
        })
        .collect();

    // Slowest preset still builds in under a millisecond and traces faster on the GPU
    let params = BvhBuildParams {
        max_prims_per_leaf: 1,
        ..BvhBuildParams::very_slow_build()
    };
    let bvh = build_bvh2(&aabbs, params, &mut Duration::default());

    let mut nodes = Vec::with_capacity(bvh.nodes.len());
    flatten(&bvh, 0, &mut nodes);

    nodes
}

/// Appends the subtree rooted at `bvh.nodes[index]` to `nodes` in depth-first order.
fn flatten(bvh: &Bvh2, index: u32, nodes: &mut Vec<BvhNode>) {
    let node = bvh.nodes[index as usize];
    let aabb = node.aabb();
    let bounds = Aabb {
        min: Vec3::from(aabb.min),
        max: Vec3::from(aabb.max),
    };
    let flat_index = nodes.len();

    if node.is_leaf() {
        assert_eq!(node.prim_count, 1, "BVH leaf with more than one sphere");
        let sphere_index = bvh.primitive_indices[node.first_index as usize];
        nodes.push(BvhNode::leaf(bounds, flat_index as u32 + 1, sphere_index));
        return;
    }

    // Exit index is patched in once the subtree is flattened
    nodes.push(BvhNode::interior(bounds, 0));

    // Order children along the longest axis so neighbouring rays take similar paths
    let (mut first, mut second) = (node.first_index, node.first_index + 1);
    let axis = aabb.largest_axis();
    if bvh.nodes[first as usize].aabb().center_axis(axis)
        > bvh.nodes[second as usize].aabb().center_axis(axis)
    {
        std::mem::swap(&mut first, &mut second);
    }

    flatten(bvh, first, nodes);
    flatten(bvh, second, nodes);

    nodes[flat_index].exit_index = nodes.len() as u32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::scene;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn structure_is_valid() {
        let spheres = scene(&mut StdRng::seed_from_u64(1));
        let nodes = build_bvh(&spheres);

        assert_eq!(nodes.len(), 2 * spheres.len() - 1);

        let mut seen = vec![false; spheres.len()];
        check_subtree(&nodes, 0, &mut seen);
        assert!(seen.iter().all(|&s| s));
        assert_eq!(nodes[0].exit_index as usize, nodes.len());
    }

    /// Returns the index after the subtree rooted at `index`.
    fn check_subtree(nodes: &[BvhNode], index: usize, seen: &mut [bool]) -> usize {
        let node = nodes[index];

        if node.sphere != 0 {
            let sphere = (node.sphere - 1) as usize;
            assert!(!seen[sphere], "sphere {sphere} in two leaves");
            seen[sphere] = true;
            assert_eq!(node.exit_index as usize, index + 1);
            return index + 1;
        }

        let left = index + 1;
        let right = check_subtree(nodes, left, seen);
        let end = check_subtree(nodes, right, seen);
        assert_eq!(node.exit_index as usize, end);

        for child in [nodes[left], nodes[right]] {
            assert!(child.min.cmpge(node.min).all() && child.max.cmple(node.max).all());
        }

        end
    }
}
