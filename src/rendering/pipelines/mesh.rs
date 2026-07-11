use std::io::Read;

use ash::{Device, vk};
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut, SystemState},
    world::World,
};

use crate::rendering::{mesh::Vertex, utils::create_shader_module};

use crate::rendering::{
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

use super::GraphicsPipelineCreateDescription;

#[derive(Resource)]
pub struct MeshPipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_cache: DescriptorCache,
    uniform: Handle<UniformBuffers>,
}

pub fn create_mesh_pipeline(
    mut commands: Commands,
    mut rc: ResMut<RenderContext>,
    depth_attachment: Res<DepthAttachment>,
    mut uniform_buffers: ResMut<RenderAssets<UniformBuffers>>,
) {
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

    commands.insert_resource(pipeline);
}

pub fn destroy_mesh_pipeline(world: &mut World) {
    let pipeline = world.remove_resource::<MeshPipeline>().unwrap();
    let rc = world.resource::<RenderContext>();
    unsafe {
        rc.device.destroy_pipeline(pipeline.pipeline, None);
        rc.device.destroy_pipeline_layout(pipeline.layout, None);
        pipeline.descriptor_cache.destroy(&rc.device);
    }
}

pub fn record_command_buffer_mesh_pipeline(
    pipeline: Res<MeshPipeline>,
    mut rc: ResMut<RenderContext>,
    depth_attachment: Res<DepthAttachment>,
    simple_images: Res<RenderAssets<SimpleImage>>,
) -> bevy_ecs::error::Result {
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

        rc.device
            .cmd_bind_index_buffer(command_buffer, rc.index_buffer, 0, vk::IndexType::UINT32);

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
