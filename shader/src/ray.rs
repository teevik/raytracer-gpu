use vek::Vec3;

use crate::{data::Face, F};

#[derive(Clone, Copy, Default)]
pub struct Ray {
    pub origin: Vec3<F>,
    pub direction: Vec3<F>,
}

impl Ray {
    pub fn at(self, distance: F) -> Vec3<F> {
        self.origin + (self.direction * distance)
    }

    pub fn get_face(self, outward_normal: Vec3<F>) -> Face {
        let direction = Vec3::dot(self.direction, outward_normal);

        if direction < 0. {
            Face::Front
        } else {
            Face::Back
        }
    }
}
