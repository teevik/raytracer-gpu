mod scene;

use bevy_utils::default;
use rand::{thread_rng, Rng};
use scene::scene;
use shader::{Camera, RaytraceSettings};
use std::{fs, mem::size_of, time::Instant};
use vek::{num_traits::Float, Vec2, Vec3};
use wgpu::{
    include_spirv,
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferDescriptor, BufferUsages, ComputePipelineDescriptor,
    DeviceDescriptor, Features, Instance, InstanceDescriptor, Maintain, PipelineLayoutDescriptor,
    ShaderStages,
};

#[pollster::main]
async fn main() {
    env_logger::init();

    let shader = include_spirv!(env!("shader.spv"));

    let camera = Camera {
        position: Vec3::new(13., 2., 3.),
        target: Vec3::new(0., 0., 0.),
        up: Vec3::new(0., 1., 0.),

        vertical_fov: (20.).to_radians(),
        defocus_angle: (0.6).to_radians(),
        focus_distance: 10.,
    };

    let screen_size = Vec2::new(800, 400);
    let amount_of_samples = 10;
    let max_depth = 50;

    let raytrace_settings = RaytraceSettings {
        camera,
        screen_size,
        amount_of_samples,
        max_depth,
    };

    let spheres = scene();

    // Setup
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::PRIMARY,
        ..default()
    });

    let adapter = instance
        .request_adapter(&default())
        .await
        .expect("No adapter");

    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: Some("Device"),
                features: Features::MAPPABLE_PRIMARY_BUFFERS,
                limits: default(),
            },
            None,
        )
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
        usage: BufferUsages::STORAGE | BufferUsages::MAP_READ,
    });

    let output_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Output buffer"),
        size: (screen_size[0] * screen_size[1]) as u64 * (size_of::<Vec3<f32>>() as u64),
        mapped_at_creation: false,
        usage: BufferUsages::STORAGE | BufferUsages::MAP_READ,
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
        ],
    });

    let compute_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Compute pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Compute pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader_module,
        entry_point: "main",
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
        ],
    });

    let mut rng = thread_rng();

    let time_started = Instant::now();
    for i in 0..amount_of_samples {
        eprintln!("Sample {i}");

        let seed = rng.gen::<u32>();
        queue.write_buffer(&seed_buffer, 0, bytemuck::bytes_of(&seed));

        let mut encoder = device.create_command_encoder(&default());
        {
            let mut compute_pass = encoder.begin_compute_pass(&default());
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(screen_size[0], screen_size[1], 1);
        }

        queue.submit([encoder.finish()]);
        device.poll(wgpu::MaintainBase::Wait);
    }

    let elapsed_time = time_started.elapsed().as_secs_f32();
    eprintln!("Elapsed time: {elapsed_time:.2}");

    let buffer_slice = output_buffer.slice(..);

    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll device to let it run the compute shader
    device.poll(Maintain::Wait);

    receiver
        .recv_async()
        .await
        .expect("Flume")
        .expect("Buffer map error");

    let buffer_view = buffer_slice.get_mapped_range();
    let output_data: &[Vec3<f32>] = bytemuck::cast_slice(&*buffer_view);

    // Export to ppm
    let mut output_ppm = String::new();
    output_ppm += &format!("P3\n{} {}\n255\n", screen_size[0], screen_size[1]);

    for pixel in output_data {
        let pixel = pixel.map(|c| c.sqrt()); // map from linear to gamma 2
        let pixel = pixel.map(|c| f32::round(c * 255.) as u8);

        output_ppm += &format!("{} {} {}\n", pixel[0], pixel[1], pixel[2]);
    }

    fs::write("image.ppm", output_ppm).unwrap();
}
