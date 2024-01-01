mod scene;

use std::fs;

use bevy_utils::default;
use scene::scene;
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
    let screen_size: [u32; 2] = [800, 400];
    let amount_of_samples = 5;
    let max_depth: u32 = 400;
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
    let screen_size_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Screen size buffer"),
        contents: bytemuck::bytes_of(&screen_size),
        usage: BufferUsages::UNIFORM,
    });

    let max_depth_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Screen size buffer"),
        contents: bytemuck::bytes_of(&max_depth),
        usage: BufferUsages::UNIFORM,
    });

    let sphere_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Sphere buffer"),
        contents: bytemuck::cast_slice(spheres.as_slice()),
        usage: BufferUsages::STORAGE | BufferUsages::MAP_READ,
    });

    let output_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Output buffer"),
        size: ((screen_size[0] * screen_size[1]) as u64) * 4 * 4,
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
                    ty: BufferBindingType::Uniform,
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
                resource: screen_size_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: max_depth_buffer.as_entire_binding(),
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

    let mut encoder = device.create_command_encoder(&default());

    {
        let mut compute_pass = encoder.begin_compute_pass(&default());
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(screen_size[0], screen_size[1], amount_of_samples);
    }

    queue.submit([encoder.finish()]);

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
    let output_data: &[[f32; 4]] = bytemuck::cast_slice(&*buffer_view);

    // Export to ppm
    let mut output_ppm = String::new();
    output_ppm += &format!("P3\n{} {}\n255\n", screen_size[0], screen_size[1]);

    for pixel in output_data {
        let pixel = pixel.map(|c| c / (amount_of_samples as f32));
        let pixel = pixel.map(|c| c.sqrt()); // map from linear to gamma 2
        let pixel = pixel.map(|c| f32::round(c * 255.) as u8);

        output_ppm += &format!("{} {} {}\n", pixel[0], pixel[1], pixel[2]);
    }

    fs::write("image.ppm", output_ppm).unwrap();
}
