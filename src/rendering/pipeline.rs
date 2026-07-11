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

pub trait GraphicsPipeline {
    fn create(world: &mut World);
    fn record_command_buffer(
        world: &mut World,
        rc: &mut RenderContext,
        swapchain_image_index: usize,
    ) -> anyhow::Result<()>;
    fn destroy(self, world: &mut World);
}

pub fn create_shader_module(device: &Device, buf: &[u8]) -> vk::ShaderModule {
    let code: &[u32] = bytemuck::cast_slice(buf);
    let create_info = vk::ShaderModuleCreateInfo::default().code(code);

    unsafe { device.create_shader_module(&create_info, None).unwrap() }
}

#[derive(Resource)]
pub struct MeshPipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_cache: DescriptorCache,
    uniform: Handle<UniformBuffers>,
}

impl GraphicsPipeline for MeshPipeline {
    fn create(world: &mut World) {
        let mut system_state = SystemState::<(
            ResMut<RenderContext>,
            Res<DepthAttachment>,
            ResMut<RenderAssets<UniformBuffers>>,
        )>::new(world);
        let (mut rc, depth_attachment, mut uniform_buffers) = system_state.get_mut(world);

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

        let vertex_binding_descriptions = &[Vertex::get_binding_description()];
        let vertex_attribute_descriptions = &Vertex::get_attribute_descriptions();
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(vertex_binding_descriptions)
            .vertex_attribute_descriptions(vertex_attribute_descriptions);

        let color_blend_attachments = &[vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];

        let buffers = UniformBuffers::new::<ViewUniform>(&rc.device.clone(), &mut rc.allocator);
        // let sampler = Sampler::basic(&rc.instance, rc.physical_device, &rc.device);
        // let image = SimpleImage::new(
        //     &rc.device,
        //     &mut rc.allocator,
        //     &rc.command_pool,
        //     &rc.queue,
        //     SimpleImageCreateInfo {
        //         width: todo!(),
        //         height: todo!(),
        //         format: todo!(),
        //         tiling: todo!(),
        //         usage: todo!(),
        //         memory_location: todo!(),
        //         data: todo!(),
        //     },
        // );

        let descriptor_cache = DescriptorCache::new(
            &rc.device,
            &[
                DescriptorSetBinding {
                    binding: 0,
                    descriptor_kind: DescriptorKind::UniformBuffer {
                        buffers: buffers.raw_buffers(),
                        offset: 0,
                        range: size_of::<ViewUniform>() as u64,
                    },
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                },
                // DescriptorSetBinding {
                //     binding: 1,
                //     descriptor_kind: DescriptorKind::CombinedSampler {
                //         sampler: sampler.0,
                //         image_view: todo!(),
                //         image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                //     },
                //     descriptor_count: 1,
                //     stage_flags: vk::ShaderStageFlags::FRAGMENT,
                // },
            ],
        )
        .unwrap();

        let set_layouts = &[descriptor_cache.layout()];
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

        let desc = GraphicsPipelineCreateDescription::basic(
            &shader_stages,
            vertex_input_info,
            depth_stencil,
            color_blend_attachments,
        );

        let pipeline_create_info = desc
            .vk_info(pipeline_layout)
            .push_next(&mut pipeline_rendering_create_info);

        let graphics_pipeline = unsafe {
            rc.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .unwrap()
                .pop()
                .unwrap()
        };

        unsafe { rc.device.destroy_shader_module(shader_module, None) };

        let pipeline = MeshPipeline {
            pipeline: graphics_pipeline,
            layout: pipeline_layout,
            descriptor_cache,
            uniform: todo!(),
        };

        world.insert_resource(pipeline);
    }

    fn destroy(self, world: &mut World) {
        let rc = world.resource_mut::<RenderContext>();
        unsafe {
            rc.device.destroy_pipeline(self.pipeline, None);
            rc.device.destroy_pipeline_layout(self.layout, None);
            self.descriptor_cache.destroy(&rc.device);
        }
    }

    fn record_command_buffer(
        world: &mut World,
        rc: &mut RenderContext,
        swapchain_image_index: usize,
    ) -> anyhow::Result<()> {
        let (depth_attachment, simple_images, pipeline) = SystemState::<(
            Res<DepthAttachment>,
            Res<RenderAssets<SimpleImage>>,
            Res<MeshPipeline>,
        )>::new(world)
        .get_mut(world);

        let depth_attachment_image = simple_images.get(depth_attachment.image.clone()).unwrap();

        unsafe {
            let command_buffer = rc.command_buffers[rc.frame_index];

            rc.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;

            transition_image_layout(
                &rc.device,
                rc.command_buffers[rc.frame_index],
                rc.swapchain_images[swapchain_image_index],
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::ImageAspectFlags::COLOR,
            );

            transition_image_layout(
                &rc.device,
                rc.command_buffers[rc.frame_index],
                depth_attachment_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            );

            let clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            };
            let clear_depth = vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue::default().depth(1.0).stencil(0),
            };

            let attachment_info = vk::RenderingAttachmentInfo::default()
                .image_view(rc.swapchain_image_views[swapchain_image_index])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_value);

            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth_attachment_image.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(clear_depth);

            let color_attachments = &[attachment_info];
            let rendering_info = vk::RenderingInfo::default()
                .render_area(
                    vk::Rect2D::default()
                        .offset(vk::Offset2D::default().x(0).y(0))
                        .extent(rc.swapchain_extent),
                )
                .layer_count(1)
                .color_attachments(color_attachments)
                .depth_attachment(&depth_attachment);

            rc.device
                .cmd_begin_rendering(command_buffer, &rendering_info);

            rc.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline,
            );

            rc.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &[rc.vertex_buffer], &[0]);

            rc.device.cmd_bind_index_buffer(
                command_buffer,
                rc.index_buffer,
                0,
                vk::IndexType::UINT32,
            );

            rc.device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport::default()
                    .x(0.0)
                    .y(0.0)
                    .width(rc.swapchain_extent.width as f32)
                    .height(rc.swapchain_extent.height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0)],
            );
            rc.device.cmd_set_scissor(
                command_buffer,
                0,
                &[vk::Rect2D::default()
                    .offset(vk::Offset2D::default())
                    .extent(rc.swapchain_extent)],
            );

            rc.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                rc.pipeline_layout,
                0,
                &[rc.descriptor_sets[rc.frame_index]],
                &[],
            );

            rc.device
                .cmd_draw_indexed(command_buffer, rc.index_count as u32, 1, 0, 0, 0);

            rc.device.cmd_end_rendering(command_buffer);

            transition_image_layout(
                &rc.device,
                rc.command_buffers[rc.frame_index],
                rc.swapchain_images[swapchain_image_index],
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                vk::ImageAspectFlags::COLOR,
            );

            rc.device.end_command_buffer(command_buffer).unwrap();
        }

        Ok(())
    }
}

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
