mod bvh;
mod scene;

use bvh::build_bvh;
use bytemuck::Zeroable;
use glam::{UVec2, Vec3, Vec4};
use rand::{rngs::StdRng, Rng, RngExt, SeedableRng};
use scene::scene;
use shader::{BvhNode, RaytraceSettings, Sphere, UVec3, Viewport};
use std::{fs, mem::size_of, sync::mpsc, time::Instant};
use wgpu::{
    include_spirv,
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferDescriptor, BufferUsages, ComputePipelineDescriptor,
    DeviceDescriptor, ExperimentalFeatures, Features, Instance, InstanceDescriptor, MapMode,
    PipelineLayoutDescriptor, PollType, ShaderStages,
};

pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,

    pub vertical_fov: f32,
    pub defocus_angle: f32,
    pub focus_distance: f32,
}

fn calculate_viewport(camera: Camera, screen_size: UVec2) -> Viewport {
    let aspect_ratio = (screen_size.x as f32) / (screen_size.y as f32);

    let h = f32::tan(camera.vertical_fov / 2.);
    let height = 2.0 * h * camera.focus_distance;
    let width = height * aspect_ratio;

    let w = (camera.position - camera.target).normalize();
    let u = Vec3::cross(camera.up, w).normalize();
    let v = Vec3::cross(w, u);

    let horizontal = width * u;
    let vertical = -height * v;

    let horizontal_pixel_delta = horizontal / (screen_size.x as f32);
    let vertical_pixel_delta = vertical / (screen_size.y as f32);

    let upper_left_corner =
        camera.position - (camera.focus_distance * w) - horizontal / 2. - vertical / 2.;

    let upper_left_pixel_position =
        upper_left_corner + horizontal_pixel_delta / 2. + vertical_pixel_delta / 2.;

    let defocus_radius = camera.focus_distance * f32::tan(camera.defocus_angle / 2.);
    let horizontal_defocus_disk = u * defocus_radius;
    let vertical_defocus_disk = v * defocus_radius;

    Viewport {
        origin: camera.position,
        upper_left_pixel_position,

        horizontal_pixel_delta,
        vertical_pixel_delta,

        horizontal_defocus_disk,
        vertical_defocus_disk,

        ..Zeroable::zeroed()
    }
}

fn make_rng() -> Box<dyn Rng> {
    let seed = std::env::var("SEED").unwrap_or_else(|_| "1".to_string());

    if seed == "random" {
        return Box::new(rand::rng());
    }

    let seed: u64 = seed
        .parse()
        .unwrap_or_else(|_| panic!("SEED must be a u64 or \"random\", got {seed:?}"));
    Box::new(StdRng::seed_from_u64(seed))
}

/// Reads an optional env var, falling back to `default` when unset
fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{name} has an invalid value: {value:?}")),
        Err(_) => default,
    }
}

#[allow(dead_code)]
fn render_cpu(
    raytrace_settings: &RaytraceSettings,
    spheres: &[Sphere],
    bvh_nodes: &[BvhNode],
) -> Vec<Vec4> {
    let screen_size = raytrace_settings.screen_size;

    let mut output = vec![Vec4::ZERO; (screen_size.x * screen_size.y) as usize];

    for _ in 0..raytrace_settings.amount_of_samples {
        for y in 0..screen_size.y {
            for x in 0..screen_size.x {
                shader::main(
                    UVec3 { x, y, z: 0 },
                    &0,
                    raytrace_settings,
                    spheres,
                    &mut output,
                    bvh_nodes,
                );
            }
        }
    }

    output
}

#[pollster::main]
async fn main() {
    env_logger::init();

    let shader = include_spirv!(env!("SHADER_SPV_PATH"));

    let camera = Camera {
        position: Vec3::new(13., 2., 3.),
        target: Vec3::new(0., 0., 0.),
        up: Vec3::new(0., 1., 0.),

        vertical_fov: 20f32.to_radians(),
        defocus_angle: 0.6f32.to_radians(),
        focus_distance: 10.,
    };

    let screen_size = UVec2::new(800, 400);
    let viewport = calculate_viewport(camera, screen_size);

    let amount_of_samples: u32 = env_or("SAMPLES", 1000);
    let max_depth = 50;
    let use_bvh = env_or::<u32>("BVH", 1) != 0;

    let raytrace_settings = RaytraceSettings {
        viewport,
        screen_size,
        amount_of_samples,
        max_depth,
        use_bvh: use_bvh as u32,
        ..Zeroable::zeroed()
    };

    let mut rng = make_rng();
    let spheres = scene(&mut rng);
    let bvh_nodes = build_bvh(&spheres);

    // Setup
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::PRIMARY,
        ..InstanceDescriptor::new_without_display_handle_from_env()
    });

    let adapter = instance
        .request_adapter(&Default::default())
        .await
        .expect("No adapter");
    eprintln!("Using adapter: {}", adapter.get_info().name);
    if use_bvh {
        eprintln!(
            "Intersection: BVH ({} nodes over {} spheres)",
            bvh_nodes.len(),
            spheres.len()
        );
    } else {
        eprintln!("Intersection: brute force ({} spheres)", spheres.len());
    }

    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            label: Some("Device"),
            required_features: Features::empty(),
            required_limits: Default::default(),
            experimental_features: ExperimentalFeatures::disabled(),
            memory_hints: Default::default(),
            trace: Default::default(),
        })
        .await
        .unwrap();

    let compute_shader_module = device.create_shader_module(shader);

    // Data
    let seed_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Seed buffer"),
        size: size_of::<u32>() as u64,
        mapped_at_creation: false,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let raytrace_settings_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Raytrace settings buffer"),
        contents: bytemuck::bytes_of(&raytrace_settings),
        usage: BufferUsages::STORAGE,
    });

    let sphere_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Sphere buffer"),
        contents: bytemuck::cast_slice(spheres.as_slice()),
        usage: BufferUsages::STORAGE,
    });

    let bvh_node_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("BVH node buffer"),
        contents: bytemuck::cast_slice(bvh_nodes.as_slice()),
        usage: BufferUsages::STORAGE,
    });

    let output_size = (screen_size.x * screen_size.y) as u64 * (size_of::<Vec4>() as u64);
    let output_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Output buffer"),
        size: output_size,
        mapped_at_creation: false,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let readback_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Readback buffer"),
        size: output_size,
        mapped_at_creation: false,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Compute bind group layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let compute_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Compute pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Compute pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Compute bind group"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: seed_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: raytrace_settings_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: sphere_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: bvh_node_buffer.as_entire_binding(),
            },
        ],
    });

    let time_started = Instant::now();
    for i in 0..amount_of_samples {
        eprintln!("Sample {i}");

        let seed = rng.random::<u32>();
        queue.write_buffer(&seed_buffer, 0, bytemuck::bytes_of(&seed));

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                screen_size.x.div_ceil(8),
                screen_size.y.div_ceil(8),
                1,
            );
        }

        queue.submit([encoder.finish()]);
        device.poll(PollType::wait_indefinitely()).unwrap();
    }

    let elapsed_time = time_started.elapsed().as_secs_f32();
    eprintln!("Elapsed time: {elapsed_time:.3}");

    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size);
    queue.submit([encoder.finish()]);

    let buffer_slice = readback_buffer.slice(..);

    let (sender, receiver) = mpsc::channel();
    buffer_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll device so the map request completes
    device.poll(PollType::wait_indefinitely()).unwrap();

    receiver
        .recv()
        .expect("Map callback dropped")
        .expect("Buffer map error");

    let buffer_view = buffer_slice.get_mapped_range().unwrap();
    let output_data: &[[f32; 4]] = bytemuck::cast_slice(&buffer_view);

    // Export to ppm
    let mut output_ppm = String::new();
    output_ppm += &format!("P3\n{} {}\n255\n", screen_size.x, screen_size.y);

    for pixel in output_data {
        // map from linear to gamma 2, then to 0..=255
        let [r, g, b, _] = pixel.map(|c| f32::round(c.sqrt() * 255.) as u8);

        output_ppm += &format!("{r} {g} {b}\n");
    }

    fs::write("image.ppm", output_ppm).unwrap();
}
