use std::io::Read;

use ash::{Device, vk};
use bevy_ecs::{
    resource::Resource,
    system::{Res, ResMut, SystemState},
    world::World,
};

use crate::rendering::mesh::Vertex;

use super::{
    ViewUniform,
    asset::{
        Handle, RenderAssets,
        common::{Sampler, SimpleImage, SimpleImageCreateInfo, UniformBuffers},
    },
    depth::DepthAttachment,
    descriptor_management::{DescriptorCache, DescriptorKind, DescriptorSetBinding},
    render_context::RenderContext,
    utils::transition_image_layout,
};

pub mod mesh;

pub struct GraphicsPipelineCreateDescription<'a> {
    pub stages: &'a [vk::PipelineShaderStageCreateInfo<'a>],
    pub vertex_input_state: vk::PipelineVertexInputStateCreateInfo<'a>,
    pub input_assembly_state: vk::PipelineInputAssemblyStateCreateInfo<'a>,
    pub viewport_state: vk::PipelineViewportStateCreateInfo<'a>,
    pub rasterization_state: vk::PipelineRasterizationStateCreateInfo<'a>,
    pub multisample_state: vk::PipelineMultisampleStateCreateInfo<'a>,
    pub depth_stencil_state: vk::PipelineDepthStencilStateCreateInfo<'a>,
    pub color_blend_state: vk::PipelineColorBlendStateCreateInfo<'a>,
    pub dynamic_state: vk::PipelineDynamicStateCreateInfo<'a>,
}

impl<'a> GraphicsPipelineCreateDescription<'a> {
    pub fn basic(
        stages: &'a [vk::PipelineShaderStageCreateInfo<'a>],
        vertex_input_state: vk::PipelineVertexInputStateCreateInfo<'a>,
        depth_stencil_state: vk::PipelineDepthStencilStateCreateInfo<'a>,
        color_blend_attachments: &'a [vk::PipelineColorBlendAttachmentState],
    ) -> Self {
        Self {
            stages,
            vertex_input_state,
            input_assembly_state: vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
            viewport_state: vk::PipelineViewportStateCreateInfo::default()
                .scissor_count(1)
                .viewport_count(1),
            rasterization_state: vk::PipelineRasterizationStateCreateInfo::default()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false)
                .line_width(1.0),
            multisample_state: vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                .sample_shading_enable(false),
            depth_stencil_state,
            color_blend_state: vk::PipelineColorBlendStateCreateInfo::default()
                .logic_op_enable(false)
                .logic_op(vk::LogicOp::COPY)
                .attachments(color_blend_attachments),
            dynamic_state: vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]),
        }
    }

    pub fn vk_info(
        &'a self,
        pipeline_layout: vk::PipelineLayout,
    ) -> vk::GraphicsPipelineCreateInfo<'a> {
        vk::GraphicsPipelineCreateInfo::default()
            .stages(self.stages)
            .vertex_input_state(&self.vertex_input_state)
            .input_assembly_state(&self.input_assembly_state)
            .viewport_state(&self.viewport_state)
            .rasterization_state(&self.rasterization_state)
            .multisample_state(&self.multisample_state)
            .color_blend_state(&self.color_blend_state)
            .dynamic_state(&self.dynamic_state)
            .layout(pipeline_layout)
            .depth_stencil_state(&self.depth_stencil_state)
    }
}
