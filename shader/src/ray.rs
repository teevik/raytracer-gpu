use vek::Vec3;

use crate::data::Face;

#[derive(Clone, Copy, Default)]
pub struct Ray {
    pub origin: Vec3<f32>,
    pub direction: Vec3<f32>,
}

impl Ray {
    pub fn at(self, distance: f32) -> Vec3<f32> {
        self.origin + (self.direction * distance)
    }

    pub fn get_face(self, outward_normal: Vec3<f32>) -> Face {
        let direction = Vec3::dot(self.direction, outward_normal);

        if direction < 0. {
            Face::Front
        } else {
            Face::Back
        }
    }
}
