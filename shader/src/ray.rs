use spirv_std::glam::Vec3;

use crate::data::Face;

#[derive(Clone, Copy, Default)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn at(self, distance: f32) -> Vec3 {
        self.origin + (self.direction * distance)
    }

    pub fn get_face(self, outward_normal: Vec3) -> Face {
        let direction = Vec3::dot(self.direction, outward_normal);

        if direction < 0. {
            Face::Front
        } else {
            Face::Back
        }
    }
}
