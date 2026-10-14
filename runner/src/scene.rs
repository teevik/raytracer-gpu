use glam::Vec3;
use rand::{Rng, RngExt};
use shader::{Material, Sphere};

pub fn scene(rng: &mut impl Rng) -> Vec<Sphere> {
    let mut spheres = vec![
        // Ground
        Sphere::stationary(
            Vec3::new(0., -1000., 0.),
            1000.,
            Material::diffuse(Vec3::new(0.5, 0.5, 0.5)),
        ),
        // Center sphere
        Sphere::stationary(Vec3::new(0., 1., 0.), 1., Material::glass(1.5)),
        // Left sphere
        Sphere::stationary(
            Vec3::new(-4., 1., 0.),
            1.,
            Material::diffuse(Vec3::new(0.4, 0.2, 0.1)),
        ),
        // Right sphere
        Sphere::stationary(
            Vec3::new(4., 1., 0.),
            1.,
            Material::metal(Vec3::new(0.7, 0.6, 0.5), 0.),
        ),
    ];

    for a in -11..11 {
        for b in -11..11 {
            let choose_material = rng.random::<f32>();

            let center = Vec3::new(
                a as f32 + 0.9 * rng.random::<f32>(),
                0.2,
                b as f32 + 0.9 * rng.random::<f32>(),
            );

            if center.distance(Vec3::new(4., 0.2, 0.)) > 0.9 {
                if choose_material < 0.8 {
                    // Diffuse

                    let mut random = || rng.random::<f32>();
                    let albedo = Vec3::new(
                        random() * random(),
                        random() * random(),
                        random() * random(),
                    );

                    // Bounces upwards while the shutter is open
                    let end = center + Vec3::new(0., rng.random_range(0. ..0.5), 0.);

                    spheres.push(Sphere::moving(center, end, 0.2, Material::diffuse(albedo)));
                } else if choose_material < 0.95 {
                    // Metal
                    let mut random = || rng.random_range(0.5..1.);
                    let albedo = Vec3::new(random(), random(), random());
                    let fuzz = rng.random_range(0. ..0.5);

                    spheres.push(Sphere::stationary(
                        center,
                        0.2,
                        Material::metal(albedo, fuzz),
                    ));
                } else {
                    spheres.push(Sphere::stationary(center, 0.2, Material::glass(1.5)));
                }
            }
        }
    }

    spheres
}
