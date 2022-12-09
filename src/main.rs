#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

use std::time::Instant;

use cgmath::Vector3;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::EventLoop,
    window::{WindowBuilder, Window, Fullscreen},
};

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
struct InstanceRaw {
    pos: [f32; 4],
    velocity: [f32; 4],
    scale: f32,
}

impl InstanceRaw {
    const ATTRIBS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![2 => Float32x4, 3 => Float32x4, 4 => Float32];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

struct Instance {
    position: Vector3<f32>,
    scale: f32,
    velocity: Vector3<f32>,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            pos: [self.position.x, self.position.y, self.position.z, 0.0],
            velocity: [self.velocity.x, self.velocity.y, self.velocity.z, 0.0],
            scale: self.scale,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
struct ShaderData {
    time: f32,
    dt: f32,
    aspect: f32, // height / width  (should be multiplied by width)
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

const VERTECIES: &[Vertex] = &[
    Vertex { position: [-0.0868241, 0.49240386, 0.0], color: [1.0, 1.0, 1.0] }, // A
    Vertex { position: [-0.49513406, 0.06958647, 0.0], color: [1.0, 1.0, 1.0] }, // B
    Vertex { position: [0.44147372, 0.2347359, 0.0], color: [1.0, 1.0, 1.0] }, // E

    Vertex { position: [-0.49513406, 0.06958647, 0.0], color: [1.0, 1.0, 1.0] }, // B
    Vertex { position: [-0.21918549, -0.44939706, 0.0], color: [1.0, 1.0, 1.0] }, // C
    Vertex { position: [0.44147372, 0.2347359, 0.0], color: [1.0, 1.0, 1.0] }, // E

    Vertex { position: [-0.21918549, -0.44939706, 0.0], color: [1.0, 1.0, 1.0] }, // C
    Vertex { position: [0.35966998, -0.3473291, 0.0], color: [1.0, 1.0, 1.0] }, // D
    Vertex { position: [0.44147372, 0.2347359, 0.0], color: [1.0, 1.0, 1.0] }, // E
];

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertecies: u32,

    instance_buffer: wgpu::Buffer,
    num_instances: u32,

    compute_pipeline: wgpu::ComputePipeline,
    instance_bind_group: wgpu::BindGroup,

    data_buffer: wgpu::Buffer,
    data_bind_group: wgpu::BindGroup,

    start_time: Instant,
    max_work_groups: u32,
}

impl State {
    async fn new(win: &Window, args: &Args) -> Self {
        let size = win.inner_size();

        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(win) };
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.unwrap();

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
            label: Some("dev"),
        }, None).await.unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
        };
        surface.configure(&device, &config);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buffer"),
            contents: bytemuck::cast_slice(VERTECIES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let num_vertecies = VERTECIES.len() as _;

        let num_instances = args.count;
        let instances: Vec<_> = (0..num_instances).map(|_| {
            let (x, y): (f32, f32) = rand::random();
            Instance {
                position: Vector3::new(x, y, 0.0) * 2.0 - Vector3::from([1.0, 1.0, 0.0]),
                scale: rand::random::<f32>() * 2.0 + 1.0,
                velocity: Vector3::new(0.0, 0.0, 0.0),
            }.to_raw()
        }).collect();

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instance_buffer"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
        });

        let compute_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/compute.wgsl"));

        let instance_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("instance_bind_group"),
            layout: &instance_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_buffer.as_entire_binding(),
            }],
        });

        let data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("data_buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            contents: bytemuck::cast_slice(&[ShaderData {
                time: 0.0,
                dt: 0.0,
                aspect: 1.0,
            }]),
        });
        let data_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("data_bind_group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let data_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("data_bind_group"),
            layout: &data_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: data_buffer.as_entire_binding(),
            }],
        });

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute_pipeline_layout"),
            bind_group_layouts: &[&instance_bind_group_layout, &data_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });


        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/main.wgsl"));
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render_pipeline_layout"),
            bind_group_layouts: &[&data_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), InstanceRaw::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let max_work_groups = device.limits().max_compute_workgroups_per_dimension;

        let start_time = Instant::now();

        Self {
            surface,
            device,
            queue,
            config,
            size,

            render_pipeline,
            vertex_buffer,
            num_vertecies,

            instance_buffer,
            num_instances,

            compute_pipeline,
            instance_bind_group,

            data_buffer,
            data_bind_group,

            start_time,
            max_work_groups,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, _ev: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self, dt: f32) {
        self.queue.write_buffer(&self.data_buffer, 0, bytemuck::cast_slice(&[ShaderData {
            time: self.start_time.elapsed().as_secs_f32(),
            dt,
            aspect: (self.config.height as f32) / (self.config.width as f32),
        }]));
    }
    
    // <render>
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass")
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.instance_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.data_bind_group, &[]);
            compute_pass.insert_debug_marker("compute_pass_update_fn");

            let compute_size = 256;
            let n_instances = (self.num_instances as f32 / compute_size as f32).ceil() as u32;
            compute_pass.dispatch_workgroups(n_instances, 1, 1);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: true,
                    }
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.data_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.draw(0..self.num_vertecies, 0..self.num_instances);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

use clap::Parser;
#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value_t = 10_000)]
    count: u32,

    #[arg(short, long, default_value_t = false)]
    background: bool,
}

async fn run() {
    env_logger::init();
    let args = Args::parse();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_transparent(true)
        .with_fullscreen(Some(Fullscreen::Borderless(None)))
        .with_always_on_top(!args.background)
        .with_decorations(false)
        .with_title("snow")
        .build(&event_loop).unwrap();

    #[cfg(windows)] {
        let handle = winit::platform::windows::WindowExtWindows::hwnd(&window);
        let handle = handle as *mut winapi::shared::windef::HWND__;

        use winapi::um::winuser::{SetWindowLongA, GWL_EXSTYLE, WS_EX_TRANSPARENT, WS_EX_LAYERED};
        unsafe { SetWindowLongA(handle, GWL_EXSTYLE as _, (WS_EX_TRANSPARENT | WS_EX_LAYERED) as _) };
    }

    let mut state = State::new(&window, &args).await;

    let mut last_t = Instant::now();

    event_loop.run(move |event, _, ctrl| match event {
        Event::WindowEvent { window_id, ref event }
        if window.id() == window_id => if !state.input(event) { match event {
            WindowEvent::CloseRequested => {
                ctrl.set_exit();
            },
            WindowEvent::Resized(size) => state.resize(*size),
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => state.resize(**new_inner_size),

            _ => {},
        }},
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            let dt = last_t.elapsed().as_secs_f32();
            let _fps = 1.0 / dt;
            // println!("fps: {fps}");
            last_t = Instant::now();
            state.update(dt);

            match state.render() {
                Ok(_) => {},
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                Err(wgpu::SurfaceError::OutOfMemory) => ctrl.set_exit(),
                Err(e) => eprintln!("render_err: {e:?}"),
            }
        },
        Event::MainEventsCleared => window.request_redraw(),
        _ => {},
    });
}

fn main() {
    pollster::block_on(run());
}


