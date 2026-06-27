use std::io::Read;

use ash::{Device, vk};
use bevy_ecs::{
    system::{Res, ResMut, SystemState},
    world::World,
};

use super::{
    ViewUniform,
    depth::DepthAttachment,
    descriptor_management::{DescriptorChache, DescriptorKind, DescriptorSetBinding},
    render_context::RenderContext,
};

pub trait GraphicsPipeline {
    fn create(world: &mut World);
    fn destroy(self, world: &mut World);
}

pub fn create_shader_module(device: &Device, buf: &[u8]) -> vk::ShaderModule {
    let code: &[u32] = bytemuck::cast_slice(buf);
    let create_info = vk::ShaderModuleCreateInfo::default().code(code);

    unsafe { device.create_shader_module(&create_info, None).unwrap() }
}

pub struct MeshPipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_cache: DescriptorChache,
}

impl GraphicsPipeline for MeshPipeline {
    fn create(world: &mut World) {
        let mut system_state =
            SystemState::<(ResMut<RenderContext>, Res<DepthAttachment>)>::new(world);
        let (mut rc, depth_attachment) = system_state.get_mut(world);

        let mut shader_code = std::fs::File::open("./shaders/triangle.spv").unwrap();
        let mut buf = vec![];
        shader_code.read_to_end(&mut buf).unwrap();
        let shader_module = create_shader_module(&rc.device, &buf);

        let vertex_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader_module)
            .name(c"vertMain");

        let fragment_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader_module)
            .name(c"fragMain");

        let shader_stages = [vertex_shader_stage_info, fragment_shader_stage_info];

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let pipeline_dynamic_state_create_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let vertex_binding_descriptions = &[Vertex::get_binding_description()];
        let vertex_attribute_descriptions = &Vertex::get_attribute_descriptions();
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(vertex_binding_descriptions)
            .vertex_attribute_descriptions(vertex_attribute_descriptions);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let pipeline_viewport_state_create_info = vk::PipelineViewportStateCreateInfo::default()
            .scissor_count(1)
            .viewport_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let attachments = &[color_blend_attachment];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(attachments);

        let cache = DescriptorChache::new(
            &rc.device,
            &[
                DescriptorSetBinding {
                    binding: 0,
                    descriptor_kind: DescriptorKind::UniformBuffer {
                        buffers: todo!(),
                        offset: 0,
                        range: size_of::<ViewUniform>() as u64,
                    },
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                },
                DescriptorSetBinding {
                    binding: 1,
                    descriptor_kind: DescriptorKind::CombinedSampler {
                        sampler: todo!(),
                        image_view: todo!(),
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    },
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                },
            ],
        );

        let set_layouts = &[];
        let pipeline_layout_create_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(set_layouts);
        let pipeline_layout = unsafe {
            rc.device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .unwrap()
        };

        let formats = &[rc.swapchain_surface_format.format];
        let mut pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(formats)
            .depth_attachment_format(depth_attachment.format);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&pipeline_viewport_state_create_info)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blend)
            .dynamic_state(&pipeline_dynamic_state_create_info)
            .layout(pipeline_layout)
            .depth_stencil_state(&depth_stencil)
            .push_next(&mut pipeline_rendering_create_info);

        let graphics_pipeline = unsafe {
            rc.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .unwrap()
        };

        unsafe { rc.device.destroy_shader_module(shader_module, None) };

        todo!("CREATE RESOURCE");
    }

    fn destroy(self, world: &mut World) {
        let rc = world.resource_mut::<RenderContext>();
        unsafe {
            rc.device.destroy_pipeline(self.pipeline, None);
            rc.device.destroy_pipeline_layout(self.layout, None);
            self.descriptor_cache.destroy(&rc.device);
        }
    }
}
