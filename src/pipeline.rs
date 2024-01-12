use crate::{shaders, vertex};
use color_eyre::Result;
use std::collections::BTreeMap;
use std::sync::Arc;
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::device::Device;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::VertexDefinition;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{RenderPass, Subpass};
use vulkano::shader::ShaderStages;

pub fn create_pipeline(
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
) -> Result<Arc<GraphicsPipeline>> {
    let vs = shaders::vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = shaders::fs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();

    let vertex_input_state =
        <vertex::Vertex as vulkano::pipeline::graphics::vertex_input::Vertex>::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap();

    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineLayoutCreateInfo {
            set_layouts: vec![
                DescriptorSetLayout::new(
                    device.clone(),
                    DescriptorSetLayoutCreateInfo {
                        bindings: BTreeMap::from([
                            (
                                0,
                                DescriptorSetLayoutBinding {
                                    stages: ShaderStages::VERTEX,
                                    ..DescriptorSetLayoutBinding::descriptor_type(
                                        DescriptorType::UniformBuffer,
                                    )
                                },
                            ),
                            (
                                1,
                                DescriptorSetLayoutBinding {
                                    stages: ShaderStages::VERTEX,
                                    ..DescriptorSetLayoutBinding::descriptor_type(
                                        DescriptorType::UniformBuffer,
                                    )
                                },
                            ),
                            (
                                2,
                                DescriptorSetLayoutBinding {
                                    stages: ShaderStages::FRAGMENT,
                                    ..DescriptorSetLayoutBinding::descriptor_type(
                                        DescriptorType::CombinedImageSampler,
                                    )
                                },
                            ),
                        ]),
                        ..Default::default()
                    },
                )
                .unwrap(),
                DescriptorSetLayout::new(
                    device.clone(),
                    DescriptorSetLayoutCreateInfo {
                        bindings: BTreeMap::from([
                            (
                                0,
                                DescriptorSetLayoutBinding {
                                    stages: ShaderStages::VERTEX,
                                    ..DescriptorSetLayoutBinding::descriptor_type(
                                        DescriptorType::UniformBuffer,
                                    )
                                },
                            ),
                            (
                                1,
                                DescriptorSetLayoutBinding {
                                    stages: ShaderStages::FRAGMENT,
                                    ..DescriptorSetLayoutBinding::descriptor_type(
                                        DescriptorType::CombinedImageSampler,
                                    )
                                },
                            ),
                        ]),
                        ..Default::default()
                    },
                )
                .unwrap(),
            ],
            ..Default::default()
        },
    )
    .unwrap();

    let subpass = Subpass::from(render_pass, 0).unwrap();

    Ok(GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            // How vertex data is read from the vertex buffers into the vertex shader
            vertex_input_state: Some(vertex_input_state),
            // How vertices are arranged into primitive shapes (triangle)
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend::alpha()),
                    ..Default::default()
                },
            )),
            // By making the viewport dynamic, we can simply recreate it when the window is resized instead of having to recreate the entire pipeline
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(subpass.into()),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )?)
}
