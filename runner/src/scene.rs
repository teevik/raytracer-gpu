use rand::{thread_rng, Rng};
use shader::{Material, Sphere};
use vek::Vec3;

pub fn scene() -> Vec<Sphere> {
    let mut spheres = Vec::new();

    // Ground
    spheres.push(Sphere {
        center: [0., -1000., 0.],
        radius: 1000.,
        material: Material::diffuse([0.5, 0.5, 0.5]),
    });

    // Center sphere
    spheres.push(Sphere {
        center: [0., 1., 0.],
        radius: 1.,
        material: Material::glass(1.5),
    });

    // Left sphere
    spheres.push(Sphere {
        center: [-4., 1., 0.],
        radius: 1.,
        material: Material::diffuse([0.4, 0.2, 0.1]),
    });

    // Right sphere
    spheres.push(Sphere {
        center: [4., 1., 0.],
        radius: 1.,
        material: Material::metal([0.7, 0.6, 0.5], 0.),
    });

    let rng = &mut thread_rng();

    for a in -11..11 {
        for b in -11..11 {
            let choose_material = rng.gen::<f32>();

            let center = Vec3::new(
                a as f32 + 0.9 * rng.gen::<f32>(),
                0.2,
                b as f32 + 0.9 * rng.gen::<f32>(),
            );

            if center.distance(Vec3::new(4., 0.2, 0.)) > 0.9 {
                let center = center.into_array();

                if choose_material < 0.8 {
                    // Diffuse

                    let mut random = || rng.gen::<f32>();
                    let albedo = [
                        random() * random(),
                        random() * random(),
                        random() * random(),
                    ];

                    spheres.push(Sphere {
                        center,
                        radius: 0.2,
                        material: Material::diffuse(albedo),
                    });
                } else if choose_material < 0.95 {
                    // Metal
                    let mut random = || rng.gen_range(0.5..1.);
                    let albedo = [random(), random(), random()];
                    let fuzz = rng.gen_range(0. ..0.5);

                    spheres.push(Sphere {
                        center,
                        radius: 0.2,
                        material: Material::metal(albedo, fuzz),
                    });
                } else {
                    spheres.push(Sphere {
                        center,
                        radius: 0.2,
                        material: Material::glass(1.5),
                    });
                }
            }
        }
    }

    spheres
}
