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

pub fn record_command_buffer_render(rc: Res<RenderContext>, mesh_pipeline: Res<MeshPipeline>) {
    let command_buffer = rc.get_current_command_buffer();

    unsafe {
        rc.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            mesh_pipeline.pipeline,
        );

        // rc.device
        //     .cmd_bind_vertex_buffers(command_buffer, 0, &[rc.vertex_buffer], &[0]);

        // rc.device
        //     .cmd_bind_index_buffer(command_buffer, rc.index_buffer, 0, vk::IndexType::UINT32);

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
            mesh_pipeline.layout,
            0,
            mesh_pipeline.descriptor_cache.descriptor_sets(),
            &[],
        );

        // rc.device
        //     .cmd_draw_indexed(command_buffer, rc.index_count as u32, 1, 0, 0, 0);
    }
}
