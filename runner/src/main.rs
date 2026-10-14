mod bvh;
mod scene;

use bvh::build_bvh;
use bytemuck::Zeroable;
use clap::{ArgAction, Parser, builder::BoolishValueParser};
use glam::{UVec2, Vec3, Vec4};
use image::{ImageFormat, RgbImage};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use scene::scene;
use shader::{BvhNode, DispatchParams, RaytraceSettings, Sphere, UVec3, Viewport};
use std::{mem::size_of, num::NonZero, path::PathBuf, str::FromStr, sync::mpsc, time::Instant};
use wgpu::{
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBinding, BufferBindingType, BufferDescriptor, BufferUsages,
    ComputePipelineDescriptor, DeviceDescriptor, ExperimentalFeatures, Features, Instance,
    InstanceDescriptor, MapMode, PipelineLayoutDescriptor, PollType, ShaderStages, include_spirv,
    util::{BufferInitDescriptor, DeviceExt},
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

#[derive(Parser)]
struct Args {
    /// Samples per pixel
    #[arg(long, env = "SAMPLES", default_value_t = 1000)]
    samples: u32,

    /// Seed for the scene and the shader's RNG: a u64, or "random"
    #[arg(long, env = "SEED", default_value = "1")]
    seed: Seed,

    /// Intersect through the BVH instead of testing every sphere
    #[arg(
        long,
        env = "BVH",
        default_value_t = true,
        action = ArgAction::Set,
        value_parser = BoolishValueParser::new(),
    )]
    bvh: bool,

    /// Where to write the rendered image
    #[arg(short, long, env = "OUTPUT", default_value = "render.png")]
    output: PathBuf,
}

#[derive(Clone, Copy)]
enum Seed {
    Fixed(u64),
    Random,
}

impl FromStr for Seed {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "random" {
            return Ok(Seed::Random);
        }

        s.parse()
            .map(Seed::Fixed)
            .map_err(|_| format!("expected a u64 or \"random\", got {s:?}"))
    }
}

fn make_rng(seed: Seed) -> Box<dyn Rng> {
    match seed {
        Seed::Fixed(seed) => Box::new(StdRng::seed_from_u64(seed)),
        Seed::Random => Box::new(rand::rng()),
    }
}

/// Samples traced per pixel by one dispatch. 8-16 was fastest on an RTX 4090, and it keeps each
/// dispatch short enough to stay clear of driver watchdogs.
const SAMPLES_PER_DISPATCH: u32 = 16;

fn dispatch_params(seed: u32, amount_of_samples: u32) -> impl Iterator<Item = DispatchParams> {
    (0..amount_of_samples)
        .step_by(SAMPLES_PER_DISPATCH as usize)
        .map(move |sample_start| DispatchParams {
            seed,
            sample_start,
            sample_count: SAMPLES_PER_DISPATCH.min(amount_of_samples - sample_start),
            ..Zeroable::zeroed()
        })
}

#[allow(dead_code)]
fn render_cpu(
    seed: u32,
    raytrace_settings: &RaytraceSettings,
    spheres: &[Sphere],
    bvh_nodes: &[BvhNode],
) -> Vec<Vec4> {
    let screen_size = raytrace_settings.screen_size;

    let mut output = vec![Vec4::ZERO; (screen_size.x * screen_size.y) as usize];

    for params in dispatch_params(seed, raytrace_settings.amount_of_samples) {
        for y in 0..screen_size.y {
            for x in 0..screen_size.x {
                shader::main(
                    UVec3 { x, y, z: 0 },
                    &params,
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
    let args = Args::parse();

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

    let amount_of_samples = args.samples;
    let max_depth = 50;
    let use_bvh = args.bvh;

    let raytrace_settings = RaytraceSettings {
        viewport,
        screen_size,
        amount_of_samples,
        max_depth,
        use_bvh: use_bvh as u32,
        ..Zeroable::zeroed()
    };

    let mut rng = make_rng(args.seed);
    let spheres = scene(&mut rng);
    let bvh_nodes = build_bvh(&spheres);
    let seed = rng.random::<u32>();

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
    // One aligned `DispatchParams` per dispatch, selected with a dynamic offset
    let dispatches: Vec<DispatchParams> = dispatch_params(seed, amount_of_samples).collect();
    let params_stride = (size_of::<DispatchParams>() as u32)
        .next_multiple_of(device.limits().min_uniform_buffer_offset_alignment);
    let mut params_bytes = vec![0u8; dispatches.len().max(1) * params_stride as usize];
    for (params, chunk) in dispatches
        .iter()
        .zip(params_bytes.chunks_exact_mut(params_stride as usize))
    {
        chunk[..size_of::<DispatchParams>()].copy_from_slice(bytemuck::bytes_of(params));
    }
    let params_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Dispatch params buffer"),
        contents: &params_bytes,
        usage: BufferUsages::UNIFORM,
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
                    has_dynamic_offset: true,
                    min_binding_size: NonZero::new(size_of::<DispatchParams>() as u64),
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
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &params_buffer,
                    offset: 0,
                    size: NonZero::new(size_of::<DispatchParams>() as u64),
                }),
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

    eprintln!(
        "Samples: {amount_of_samples} in {} dispatches of up to {SAMPLES_PER_DISPATCH}",
        dispatches.len()
    );

    let progress = ProgressBar::new(amount_of_samples as u64).with_style(
        ProgressStyle::with_template("{bar:40} {pos}/{len} samples [{elapsed_precise}, eta {eta}]")
            .unwrap(),
    );

    // Queue every dispatch up front, then wait on each in turn to drive the progress bar
    let time_started = Instant::now();
    let submissions: Vec<_> = (0..)
        .zip(&dispatches)
        .map(|(i, params): (u32, _)| {
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut compute_pass = encoder.begin_compute_pass(&Default::default());
                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[i * params_stride]);
                compute_pass.dispatch_workgroups(
                    screen_size.x.div_ceil(8),
                    screen_size.y.div_ceil(8),
                    1,
                );
            }
            (queue.submit([encoder.finish()]), params.sample_count)
        })
        .collect();

    for (submission, sample_count) in submissions {
        device
            .poll(PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .unwrap();
        progress.inc(sample_count as u64);
    }
    progress.finish_and_clear();

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

    // Export to png
    // map from linear to gamma 2, then to 0..=255
    let rgb: Vec<u8> = output_data
        .iter()
        .flat_map(|&[r, g, b, _]| [r, g, b])
        .map(|c| f32::round(c.clamp(0., 1.).sqrt() * 255.) as u8)
        .collect();

    RgbImage::from_raw(screen_size.x, screen_size.y, rgb)
        .expect("Image buffer size mismatch")
        .save_with_format(&args.output, ImageFormat::Png)
        .unwrap();
}
