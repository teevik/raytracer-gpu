use shader::{Aabb, BvhNode, Sphere};

/// Builds a BVH over `spheres` with one sphere per leaf
pub fn build_bvh(spheres: &[Sphere]) -> Vec<BvhNode> {
    let mut nodes = Vec::with_capacity(2 * spheres.len());

    if !spheres.is_empty() {
        let mut indices: Vec<u32> = (0..spheres.len() as u32).collect();
        build_node(spheres, &mut indices, &mut nodes);
    }

    nodes
}

fn bounds_of(spheres: &[Sphere], indices: &[u32]) -> Aabb {
    indices
        .iter()
        .map(|&i| spheres[i as usize].get_aabb())
        .reduce(Aabb::combine)
        .expect("BVH node without spheres")
}

fn build_node(spheres: &[Sphere], indices: &mut [u32], nodes: &mut Vec<BvhNode>) {
    let node_index = nodes.len();
    let bounds = bounds_of(spheres, indices);

    if let [sphere_index] = *indices {
        nodes.push(BvhNode::leaf(bounds, node_index as u32 + 1, sphere_index));
        return;
    }

    // Placeholder, the exit index is only known once the subtree is built
    nodes.push(BvhNode::interior(bounds, 0));

    let split = sah_split(spheres, indices);
    let (left, right) = indices.split_at_mut(split);
    build_node(spheres, left, nodes);
    build_node(spheres, right, nodes);

    nodes[node_index].exit_index = nodes.len() as u32;
}

/// Sorts `indices` along the best axis and returns the split position minimizing
/// `count_left * area_left + count_right * area_right`.
fn sah_split(spheres: &[Sphere], indices: &mut [u32]) -> usize {
    let count = indices.len();
    let mut best: Option<(f32, usize, usize)> = None; // (cost, axis, split)

    for axis in 0..3 {
        sort_by_centroid(spheres, indices, axis);

        // right_areas[i] = area of the bounds of indices[i..]
        let mut right_areas = vec![0.; count];
        let mut right_bounds = spheres[indices[count - 1] as usize].get_aabb();
        for i in (1..count).rev() {
            right_bounds = Aabb::combine(right_bounds, spheres[indices[i] as usize].get_aabb());
            right_areas[i] = right_bounds.surface_area();
        }

        let mut left_bounds = spheres[indices[0] as usize].get_aabb();
        for split in 1..count {
            left_bounds =
                Aabb::combine(left_bounds, spheres[indices[split - 1] as usize].get_aabb());

            let cost = split as f32 * left_bounds.surface_area()
                + (count - split) as f32 * right_areas[split];

            if best.is_none_or(|(best_cost, _, _)| cost < best_cost) {
                best = Some((cost, axis, split));
            }
        }
    }

    let (_, axis, split) = best.expect("SAH split needs at least two spheres");
    sort_by_centroid(spheres, indices, axis);

    split
}

fn sort_by_centroid(spheres: &[Sphere], indices: &mut [u32], axis: usize) {
    indices.sort_by(|&a, &b| {
        let ca = spheres[a as usize].get_aabb().center()[axis];
        let cb = spheres[b as usize].get_aabb().center()[axis];
        ca.total_cmp(&cb).then(a.cmp(&b))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::scene;
    use rand::{rngs::StdRng, SeedableRng};

    /// Every sphere appears in exactly one leaf, children lie inside their parent, and the exit
    /// links describe a valid depth-first layout.
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
