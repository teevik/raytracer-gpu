use crate::{
    data::{Range, RayHit},
    ray::Ray,
};

pub trait Raycastable {
    fn raycast(self, ray: Ray, range: Range) -> RayHit;
}
