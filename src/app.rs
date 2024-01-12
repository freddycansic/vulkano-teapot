use crate::{buffers, context, debug, model, scene, shaders, texture, vertex};
use cgmath::{
    EuclideanSpace, Euler, Matrix, Matrix4, Point3, Quaternion, Rad, SquareMatrix, Vector3, Vector4,
};
use color_eyre::Result;
use itertools::Itertools;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;
use vulkano::command_buffer::sys::CommandBufferBeginInfo;
use vulkano::command_buffer::{CommandBufferLevel, CommandBufferUsage};
use vulkano::command_buffer::{
    RecordingCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};

use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};

use crate::model::ModelInstance;
use crate::scene::Scene;
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::format::{ClearValue, Format};
use vulkano::image::sampler::{
    BorderColor, Filter, Sampler, SamplerAddressMode, SamplerCreateInfo,
};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageLayout, ImageType, ImageUsage, SampleCount};
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::padded::Padded;
use vulkano::pipeline::graphics::color_blend::{
    ColorBlendAttachmentState, ColorBlendState, ColorComponents,
};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{
    CullMode, DepthBiasState, FrontFace, PolygonMode, RasterizationState,
};
use vulkano::pipeline::graphics::vertex_input::{VertexDefinition, VertexInputState};
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::render_pass::{
    AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, Framebuffer,
    FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, Subpass, SubpassDependency,
    SubpassDescription, SubpassDescriptionFlags,
};
use vulkano::shader::ShaderStages;
use vulkano::swapchain::{acquire_next_image, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::{AccessFlags, DependencyFlags, GpuFuture, PipelineStages};
use vulkano::{sync, Validated, VulkanError};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

// TODO temp
struct Shadowmap {
    dimensions: u32,
    image_view: Arc<ImageView>,
    sampler: Arc<Sampler>,
    render_pass: Arc<RenderPass>,
    framebuffer: Arc<Framebuffer>,
    pipeline: Arc<GraphicsPipeline>,
}

pub struct App {
    scene: scene::Scene,
    texture: texture::Texture,
    shadowmap: Shadowmap,
    debug_depth_pipeline: Arc<GraphicsPipeline>,
    vulkan_context: context::VulkanContext,
    rendering_context: context::RenderingContext,
    window_context: context::WindowContext,
    allocators: context::Allocators,
    // TODO decide what to do with this
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl App {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        color_eyre::install().unwrap();
        debug::set_up_logging();

        // TODO deferred rendering https://learnopengl.com/Advanced-Lighting/Deferred-Shading
        let mut window_context = context::WindowContext::new(event_loop);
        let vulkan_context = context::VulkanContext::new(&window_context, event_loop);
        let allocators = context::Allocators::new(vulkan_context.device.clone());
        let rendering_context =
            context::RenderingContext::new(&vulkan_context, &mut window_context, &allocators);

        // TODO --------------------------------------------------------------------------------------------------------
        let shadowmap = {
            let dimensions = 1024;
            let image_view = ImageView::new_default(
                Image::new(
                    allocators.memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::D16_UNORM,
                        extent: [dimensions, dimensions, 1],
                        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap(),
            )
            .unwrap();

            let sampler = Sampler::new(
                vulkan_context.device.clone(),
                SamplerCreateInfo {
                    mag_filter: Filter::Linear,
                    min_filter: Filter::Linear,
                    address_mode: [SamplerAddressMode::ClampToEdge; 3],
                    border_color: BorderColor::FloatOpaqueWhite,
                    lod: 0.0..=1.0,
                    // anisotropy: Some(1.0),
                    ..Default::default()
                },
            )
            .unwrap();

            let render_pass = {
                let shadowmap_attachment_description = AttachmentDescription {
                    format: Format::D16_UNORM,
                    samples: SampleCount::Sample1,
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    stencil_load_op: Some(AttachmentLoadOp::DontCare),
                    stencil_store_op: Some(AttachmentStoreOp::DontCare),
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::DepthStencilReadOnlyOptimal,
                    ..Default::default()
                };

                let shadowmap_attachment_reference = AttachmentReference {
                    attachment: 0,
                    layout: ImageLayout::DepthStencilAttachmentOptimal,
                    ..Default::default()
                };

                let shadowmap_subpass = SubpassDescription {
                    depth_stencil_attachment: Some(shadowmap_attachment_reference),
                    // TODO there is no pipeline bind point here...
                    ..Default::default()
                };

                let subpass_dependencies = vec![
                    SubpassDependency {
                        src_subpass: None,
                        dst_subpass: Some(0),
                        src_stages: PipelineStages::FRAGMENT_SHADER,
                        // IDK
                        dst_stages: PipelineStages::EARLY_FRAGMENT_TESTS,
                        src_access: AccessFlags::SHADER_READ,
                        dst_access: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dependency_flags: DependencyFlags::BY_REGION,
                        ..Default::default()
                    },
                    SubpassDependency {
                        src_subpass: Some(0),
                        dst_subpass: None,
                        // IDK
                        src_stages: PipelineStages::LATE_FRAGMENT_TESTS,
                        dst_stages: PipelineStages::FRAGMENT_SHADER,
                        src_access: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dst_access: AccessFlags::SHADER_READ,
                        dependency_flags: DependencyFlags::BY_REGION,
                        ..Default::default()
                    },
                ];

                RenderPass::new(
                    vulkan_context.device.clone(),
                    RenderPassCreateInfo {
                        attachments: vec![shadowmap_attachment_description],
                        subpasses: vec![shadowmap_subpass],
                        dependencies: subpass_dependencies,
                        ..Default::default()
                    },
                )
                .unwrap()
            };

            let framebuffer = Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![image_view.clone()],
                    extent: [dimensions, dimensions],
                    layers: 1,
                    ..Default::default()
                },
            )
            .unwrap();

            let vertex_shader =
                shaders::offscreen_depth_vertex_shader::load(vulkan_context.device.clone())
                    .unwrap()
                    .entry_point("main")
                    .unwrap();

            let vertex_input_state =
                <vertex::Vertex as vulkano::pipeline::graphics::vertex_input::Vertex>::per_vertex()
                    .definition(&vertex_shader.info().input_interface)
                    .unwrap();

            let stage = PipelineShaderStageCreateInfo::new(vertex_shader);

            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            let layout = PipelineLayout::new(
                vulkan_context.device.clone(),
                PipelineLayoutCreateInfo {
                    set_layouts: vec![
                        DescriptorSetLayout::new(
                            vulkan_context.device.clone(),
                            DescriptorSetLayoutCreateInfo {
                                bindings: BTreeMap::from([(
                                    // View projection matrices
                                    0,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::VERTEX,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::UniformBuffer,
                                        )
                                    },
                                )]),
                                ..Default::default()
                            },
                        )
                        .unwrap(),
                        DescriptorSetLayout::new(
                            vulkan_context.device.clone(),
                            DescriptorSetLayoutCreateInfo {
                                bindings: BTreeMap::from([(
                                    // Model matrix
                                    0,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::VERTEX,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::UniformBuffer,
                                        )
                                    },
                                )]),
                                ..Default::default()
                            },
                        )
                        .unwrap(),
                    ],
                    ..Default::default()
                },
            )
            .unwrap();

            let pipeline = GraphicsPipeline::new(
                vulkan_context.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: [stage].into_iter().collect(),
                    input_assembly_state: Some(InputAssemblyState {
                        topology: PrimitiveTopology::TriangleList,
                        ..Default::default()
                    }),
                    vertex_input_state: Some(vertex_input_state),
                    rasterization_state: Some(RasterizationState {
                        cull_mode: CullMode::None,
                        depth_bias: Some(DepthBiasState::default()),
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
                    viewport_state: Some(ViewportState {
                        scissors: [Scissor::default()].into_iter().collect(),
                        viewports: [Viewport::default()].into_iter().collect(),
                        ..Default::default()
                    }),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    dynamic_state: [
                        DynamicState::Viewport,
                        DynamicState::Scissor,
                        DynamicState::DepthBias,
                    ]
                    .into_iter()
                    .collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap();

            Shadowmap {
                render_pass,
                image_view,
                dimensions,
                framebuffer,
                sampler,
                pipeline,
            }
        };

        let debug_depth_pipeline = {
            let vertex_shader = shaders::depth_texture_vertex::load(vulkan_context.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let fragment_shader =
                shaders::depth_texture_fragment::load(vulkan_context.device.clone())
                    .unwrap()
                    .entry_point("main")
                    .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vertex_shader),
                PipelineShaderStageCreateInfo::new(fragment_shader),
            ];

            let subpass = Subpass::from(rendering_context.render_pass.clone(), 0).unwrap();

            let layout = PipelineLayout::new(
                vulkan_context.device.clone(),
                PipelineLayoutCreateInfo {
                    set_layouts: vec![DescriptorSetLayout::new(
                        vulkan_context.device.clone(),
                        DescriptorSetLayoutCreateInfo {
                            bindings: BTreeMap::from([(
                                // Depth texture
                                0,
                                DescriptorSetLayoutBinding {
                                    stages: ShaderStages::FRAGMENT,
                                    ..DescriptorSetLayoutBinding::descriptor_type(
                                        DescriptorType::CombinedImageSampler,
                                    )
                                },
                            )]),
                            ..Default::default()
                        },
                    )
                    .unwrap()],
                    ..Default::default()
                },
            )
            .unwrap();

            let pipeline = GraphicsPipeline::new(
                vulkan_context.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    input_assembly_state: Some(InputAssemblyState {
                        topology: PrimitiveTopology::TriangleList,
                        ..Default::default()
                    }),
                    rasterization_state: Some(RasterizationState {
                        cull_mode: CullMode::None,
                        polygon_mode: PolygonMode::Fill,
                        front_face: FrontFace::CounterClockwise,
                        ..Default::default()
                    }),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        1,
                        ColorBlendAttachmentState {
                            blend: None,
                            color_write_mask: ColorComponents::all(),
                            ..Default::default()
                        },
                    )),
                    vertex_input_state: Some(VertexInputState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    viewport_state: Some(ViewportState {
                        scissors: [Scissor::default()].into_iter().collect(),
                        viewports: [Viewport::default()].into_iter().collect(),
                        ..Default::default()
                    }),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                        .into_iter()
                        .collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap();

            pipeline
        };

        // TODO --------------------------------------------------------------------------------------------------------

        // TODO move models, textures to resource manager / scene
        let load_model = |path| {
            model::Model::load(path, allocators.memory_allocator.clone())
                .expect(format!("Could not load \"{}\".", path).as_str())
        };

        let teapot = load_model("assets/models/teapot.glb");
        let backdrop = load_model("assets/models/backdrop.glb");
        let cube = load_model("assets/models/cube.glb");

        let scene = Scene {
            // models: vec![teapot.into(), backdrop.into(), cube.into()],
            models: vec![teapot.into(), backdrop.into()],
        };

        let mut texture_uploads = RecordingCommandBuffer::new(
            allocators.command_buffer_allocator.clone(),
            vulkan_context.queue.queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
        )
        .unwrap();

        let mut load_texture = |path| {
            texture::Texture::load(
                path,
                allocators.memory_allocator.clone(),
                vulkan_context.device.clone(),
                &mut texture_uploads,
            )
            .unwrap()
        };

        let _ferris = load_texture("assets/textures/ferris.png");
        let _wojak = load_texture("assets/textures/wojak.jpg");
        let _gmod = load_texture("assets/textures/gmod.jpg");
        let white = load_texture("assets/textures/white.jpg");

        let texture = white;

        // Submit uploading textures
        let texture_uploads_end = Some(
            texture_uploads
                .end()
                .unwrap()
                .execute(vulkan_context.queue.clone())
                .unwrap()
                .boxed(),
        );

        Self {
            debug_depth_pipeline,
            window_context,
            rendering_context,
            vulkan_context,
            allocators,
            previous_frame_end: texture_uploads_end,
            scene,
            texture,
            shadowmap,
        }
    }

    pub fn run(mut self, event_loop: EventLoop<()>) {
        let mut frame_state = FrameState {
            start: Instant::now(),
            recreate_swapchain: false,
        };

        event_loop
            .run(move |event, event_loop_window_target| {
                event_loop_window_target.set_control_flow(ControlFlow::Poll);

                match event {
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        ..
                    } => {
                        event_loop_window_target.exit();
                    }
                    Event::WindowEvent {
                        // Purposefully ignore new window size to retain 16:9 aspect ratio = no stretching
                        event: WindowEvent::Resized(_new_size),
                        ..
                    } => {
                        frame_state.recreate_swapchain = true;
                    }
                    Event::WindowEvent {
                        event: WindowEvent::RedrawRequested,
                        ..
                    } => {
                        self.render(&mut frame_state);
                    }
                    Event::AboutToWait => self.window_context.window.request_redraw(),
                    _ => (),
                };
            })
            .unwrap();
    }

    fn render(&mut self, frame_state: &mut FrameState) {
        let current_window_extent: [u32; 2] = self.window_context.window.inner_size().into();
        if current_window_extent.contains(&0) {
            return;
        }

        // Clean up last frame's resources
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if frame_state.recreate_swapchain {
            self.resize_swapchain_and_framebuffers(current_window_extent)
                .unwrap();

            frame_state.recreate_swapchain = false;
        }

        let camera_uniform_subbuffer = {
            let aspect_ratio = self.rendering_context.swapchain.image_extent()[0] as f32
                / self.rendering_context.swapchain.image_extent()[1] as f32;

            let proj =
                cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0);

            let camera_position = Point3::new(5.0, 2.0, 5.0);

            let view = Matrix4::look_at_rh(
                camera_position,
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, -1.0, 0.0),
            );

            let uniform_data = shaders::vs::CameraUniform {
                view: view.into(),
                projection: proj.into(),
                camera_position: camera_position.into(),
            };

            let subbuffer = self
                .allocators
                .subbuffer_allocator
                .allocate_sized()
                .unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };

        let elapsed = frame_state.start.elapsed().as_millis() as f32 / 1000.0;
        let radius = 4.0;
        let light_z = radius * elapsed.sin();
        let light_x = radius * elapsed.cos();
        let light_position = [light_x, 0.5, light_z, 1.0];

        let shadowmap_view_projection_matrix = {
            // 45 degrees
            let projection = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_4), 1.0, 1.0, 96.0);

            let view = Matrix4::look_at_rh(
                Point3::from_vec(Vector4::from(light_position).xyz()),
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, -1.0, 0.0),
            );

            view * projection
        };

        let lights_uniform_subbuffer = {
            const MAX_LIGHTS: usize = 10;
            let mut lights: [Padded<shaders::vs::Light, 12>; 10] =
                [shaders::vs::Light::default().into(); MAX_LIGHTS];

            lights[0] = shaders::vs::Light {
                position: light_position.into(),
                // position: [7.0, 0.7, 7.0].into(),
                color: [1.0, 1.0, 1.0, 1.0].into(),
                // IDK?????
                view_projection: shadowmap_view_projection_matrix.into(),
                intensity: 1.0,
            }
            .into();

            let lights_data = shaders::vs::LightsUniform { lights };

            let subbuffer = self
                .allocators
                .subbuffer_allocator
                .allocate_sized()
                .unwrap();
            *subbuffer.write().unwrap() = lights_data;

            subbuffer
        };

        let per_frame_descriptor_set = DescriptorSet::new(
            self.allocators.descriptor_set_allocator.clone(),
            self.rendering_context.pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, camera_uniform_subbuffer),
                WriteDescriptorSet::buffer(1, lights_uniform_subbuffer),
                WriteDescriptorSet::image_view_sampler(
                    2,
                    self.shadowmap.image_view.clone(),
                    self.shadowmap.sampler.clone(),
                ),
            ],
            [],
        )
        .unwrap();

        let debug_depth_pipeline_descriptor_set = DescriptorSet::new(
            self.allocators.descriptor_set_allocator.clone(),
            self.debug_depth_pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                self.shadowmap.image_view.clone(),
                self.shadowmap.sampler.clone(),
            )],
            [],
        )
        .unwrap();

        // Acquire next image to draw upon
        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.rendering_context.swapchain.clone(), None)
                .map_err(Validated::unwrap)
            {
                Ok(next) => next,
                Err(VulkanError::OutOfDate) => {
                    frame_state.recreate_swapchain = true;
                    return;
                }
                Err(error) => panic!("Failed to acquire next image: {error}"),
            };

        // Drawing on suboptimal images can produce graphical errors
        if suboptimal {
            return;
        }

        // Holds list of commands to be executed
        let mut builder = RecordingCommandBuffer::new(
            self.allocators.command_buffer_allocator.clone(),
            self.vulkan_context.queue.queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
        )
        .unwrap();

        self.scene.models[1].transform =
            Matrix4::from_translation(Vector3::new(0.0, -2.0, 0.0)) * Matrix4::from_scale(10.0);

        // self.scene.models[2].transform =
        //     Matrix4::from_translation(Vector4::from(light_position).xyz())
        //         * Matrix4::from_scale(0.2);

        {
            let shadowmap_view_projection_subbuffer = {
                let uniform_data = shaders::offscreen_depth_vertex_shader::ViewProjectionUniform {
                    view_projection: shadowmap_view_projection_matrix.into(),
                };

                let subbuffer = self
                    .allocators
                    .subbuffer_allocator
                    .allocate_sized()
                    .unwrap();
                *subbuffer.write().unwrap() = uniform_data;

                subbuffer
            };

            let shadowmap_descriptor_set = DescriptorSet::new(
                self.allocators.descriptor_set_allocator.clone(),
                self.shadowmap.pipeline.layout().set_layouts()[0].clone(),
                [WriteDescriptorSet::buffer(
                    0,
                    shadowmap_view_projection_subbuffer,
                )],
                [],
            )
            .unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        render_pass: self.shadowmap.render_pass.clone(),
                        // TODO should be depth stencil?
                        clear_values: vec![Some(ClearValue::Depth(1.0))],
                        // clear_values: vec![Some(ClearValue::DepthStencil((1.0, 0)))],
                        ..RenderPassBeginInfo::framebuffer(self.shadowmap.framebuffer.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap()
                .set_viewport(
                    0,
                    [Viewport {
                        extent: [
                            self.shadowmap.dimensions as f32,
                            self.shadowmap.dimensions as f32,
                        ],
                        ..Default::default()
                    }]
                    .into_iter()
                    .collect(),
                )
                .unwrap()
                .set_scissor(
                    0,
                    [Scissor {
                        extent: [self.shadowmap.dimensions, self.shadowmap.dimensions],
                        ..Default::default()
                    }]
                    .into_iter()
                    .collect(),
                )
                .unwrap()
                // "Avoid shadowing artefacts"
                .set_depth_bias(1.25, 0.0, 1.75)
                .unwrap()
                .bind_pipeline_graphics(self.shadowmap.pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.shadowmap.pipeline.layout().clone(),
                    0,
                    shadowmap_descriptor_set,
                )
                .unwrap();

            self.render_shadowmap(&mut builder);

            builder.end_render_pass(Default::default()).unwrap();
        }

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    render_pass: self.rendering_context.render_pass.clone(),
                    // Clear values for each attachment
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into()), Some(1_f32.into())],
                    ..RenderPassBeginInfo::framebuffer(
                        self.rendering_context.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .set_viewport(
                0,
                [self.window_context.viewport.clone()].into_iter().collect(),
            )
            .unwrap();

        if (false) {
            builder
                .set_scissor(0, [Scissor::default()].into_iter().collect())
                .unwrap()
                .bind_pipeline_graphics(self.debug_depth_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.debug_depth_pipeline.layout().clone(),
                    0,
                    debug_depth_pipeline_descriptor_set,
                )
                .unwrap();

            unsafe {
                builder.draw(3, 1, 0, 0).unwrap();
            }
        } else {
            builder
                .bind_pipeline_graphics(self.rendering_context.pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.rendering_context.pipeline.layout().clone(),
                    0,
                    per_frame_descriptor_set,
                )
                .unwrap();

            self.scene.models[0].render(
                &mut builder,
                &self.allocators,
                self.rendering_context.pipeline.clone(),
                &self.texture,
            );

            // Backdrop
            // self.scene.models[1].render(
            //     &mut builder,
            //     &self.allocators,
            //     self.rendering_context.pipeline.clone(),
            //     &self.texture,
            // );

            // Cube
            // self.scene.models[2].render(
            //     &mut builder,
            //     &self.allocators,
            //     self.rendering_context.pipeline.clone(),
            //     &self.texture,
            // );
        }

        builder.end_render_pass(Default::default()).unwrap();

        // Finish recording commands
        let command_buffer = builder.end().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.vulkan_context.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.vulkan_context.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.rendering_context.swapchain.clone(),
                    image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => self.previous_frame_end = Some(future.boxed()),
            Err(VulkanError::OutOfDate) => {
                frame_state.recreate_swapchain = true;
                self.previous_frame_end =
                    Some(sync::now(self.vulkan_context.device.clone()).boxed());
            }
            Err(error) => {
                panic!("Failed to flush future: {error}");
            }
        };
    }

    fn render_shadowmap(&mut self, builder: &mut RecordingCommandBuffer) {
        // for model in &self.scene.models {
        // TODO temp only do teapot
        let model = &self.scene.models[0];
        let model_matrix_subbuffer = {
            let uniform_data = shaders::offscreen_depth_vertex_shader::ModelUniform {
                model: model.transform.into(),
            };

            let subbuffer = self
                .allocators
                .subbuffer_allocator
                .allocate_sized()
                .unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };

        let model_descriptor_set = DescriptorSet::new(
            self.allocators.descriptor_set_allocator.clone(),
            self.shadowmap.pipeline.layout().set_layouts()[1].clone(),
            [WriteDescriptorSet::buffer(0, model_matrix_subbuffer)],
            [],
        )
        .unwrap();

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.shadowmap.pipeline.layout().clone(),
                1,
                model_descriptor_set,
            )
            .unwrap()
            .bind_vertex_buffers(0, model.model.meshes[0].primitives[0].vertex_buffer.clone())
            .unwrap()
            .bind_index_buffer(model.model.meshes[0].primitives[0].index_buffer.clone())
            .unwrap();

        unsafe {
            builder
                .draw_indexed(
                    model.model.meshes[0].primitives[0].index_buffer.len() as u32,
                    1,
                    0,
                    0,
                    0,
                )
                .unwrap();
        }
    }

    fn resize_swapchain_and_framebuffers(&mut self, new_window_extent: [u32; 2]) -> Result<()> {
        let (new_swapchain, new_images) =
            self.rendering_context
                .swapchain
                .recreate(SwapchainCreateInfo {
                    // New size of window
                    image_extent: new_window_extent,
                    ..self.rendering_context.swapchain.create_info()
                })?;

        self.rendering_context.swapchain = new_swapchain;

        self.rendering_context.framebuffers = buffers::create_framebuffers(
            &new_images,
            &mut self.window_context.viewport,
            self.allocators.memory_allocator.clone(),
            self.rendering_context.render_pass.clone(),
        )?;

        Ok(())
    }
}

struct FrameState {
    start: Instant,
    recreate_swapchain: bool,
}
