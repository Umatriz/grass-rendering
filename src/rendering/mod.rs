use std::{
    borrow::Cow,
    f32::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_8},
    ffi::CStr,
    io::Read,
    mem,
    sync::Arc,
};

use ash::{
    Device, Entry, Instance, ext, khr,
    nv::device_diagnostics_config,
    vk::{self, Fence},
};
use bevy_app::{Plugin, PostUpdate, Startup};
use bevy_ecs::{
    message::MessageReader,
    resource::Resource,
    schedule::{IntoScheduleConfigs, ScheduleLabel, SystemSet},
    system::{Commands, Res, ResMut, Single},
    world::World,
};
use bevy_time::Time;
use bytemuck::{Pod, Zeroable};
use camera::Camera;
use glam::{Mat3, Mat4, Quat, Vec2, Vec3, vec3};
use gltf::{
    accessor::{self, DataType, Dimensions},
    json::camera::Type,
    mesh::util::{ReadIndices, ReadPositions},
};
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings, MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};
use itertools::{Itertools, multizip};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use render_context::RenderContext;
use tracing::{error, info, trace, warn};
use winit::{
    dpi::PhysicalSize, event::WindowEvent, event_loop::OwnedDisplayHandle, window::Window,
};

use crate::{
    transform::Transform,
    windowing::{AppWindows, RawWinitWindowEvent, WinitOwnedDisplayHandle},
};

pub mod asset;
pub mod camera;
pub mod depth;
pub mod descriptor_management;
pub mod mesh;
pub mod pipelines;
pub mod render_context;
pub mod utils;

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ViewUniform {
    model: Mat4,
    view: Mat4,
    projection: Mat4,
    normal_matrix: Mat3,
    eye_pos: Vec3,
    light_pos: Vec3,
}

// const VERTICES: &[Vertex] = &[
//     Vertex {
//         pos: vec2(-0.5, -0.5),
//         color: vec3(1.0, 0.0, 0.0),
//     },
//     Vertex {
//         pos: vec2(0.5, -0.5),
//         color: vec3(0.0, 1.0, 0.0),
//     },
//     Vertex {
//         pos: vec2(0.5, 0.5),
//         color: vec3(0.0, 0.0, 1.0),
//     },
//     Vertex {
//         pos: vec2(-0.5, 0.5),
//         color: vec3(1.0, 1.0, 1.0),
//     },
// ];

// const INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];

pub struct RenderingPlugin;

impl Plugin for RenderingPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.configure_sets(
            PostUpdate,
            (
                RenderSet::Resize,
                RenderSet::Prepare,
                RenderSet::BeginRender,
                RenderSet::AquireSwapchainImage,
                RenderSet::RecordCommandBuffers,
                RenderSet::SubmitQueue,
            )
                .chain(),
        );

        app.add_systems(
            PostUpdate,
            (
                aquire_swapchain_image_index.in_set(RenderSet::AquireSwapchainImage),
                record_command_buffer.in_set(RenderSet::RecordCommandBuffers),
                queue_submit_present.in_set(RenderSet::SubmitQueue),
            ),
        );
    }
}

/// This schedule runs after the window is closed but before `world.clear_all()`.
#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone)]
pub struct CleanUp;

#[derive(SystemSet, Debug, PartialEq, Eq, Hash, Clone)]
pub enum RenderSet {
    Resize,
    Prepare,
    BeginRender,
    AquireSwapchainImage,
    RecordCommandBuffers,
    SubmitQueue,
}

#[derive(Resource)]
pub struct FrameSwapchainImageIndex(pub usize);

fn aquire_swapchain_image_index(mut commands: Commands, mut rc: ResMut<RenderContext>) {
    unsafe {
        rc.device
            .wait_for_fences(&[rc.in_flight_fences[rc.frame_index]], true, u64::MAX)
            .unwrap();

        let image_index = match rc.swapchain.1.acquire_next_image(
            rc.swapchain.0,
            u64::MAX,
            rc.present_complete_semaphores[rc.frame_index],
            vk::Fence::null(),
        ) {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                rc.swapchain_ok = false;
                return;
            }
            Err(e) => {
                panic!("failed to aquire swapchain image: {e}");
            }
        };

        rc.device
            .reset_fences(&[rc.in_flight_fences[rc.frame_index]])
            .unwrap();

        commands.insert_resource(FrameSwapchainImageIndex(image_index as usize));
    }

    // rc.update_uniform_buffers(elapsed_time, camera_data);

    // rc.record_command_buffer(image_index as usize);
}

fn record_command_buffer(rc: Res<RenderContext>) {
    unsafe {
        rc.device
            .reset_command_buffer(
                rc.command_buffers[rc.frame_index],
                vk::CommandBufferResetFlags::empty(),
            )
            .unwrap();
    }
}

fn queue_submit_present(mut rc: ResMut<RenderContext>, image_index: Res<FrameSwapchainImageIndex>) {
    let wait_destination_stage_mask = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
    let wait_semaphores = &[rc.present_complete_semaphores[rc.frame_index]];
    let wait_dst_stage_mask = &[wait_destination_stage_mask];
    let command_buffers = &[rc.command_buffers[rc.frame_index]];
    let signal_semaphores = &[rc.render_finished_semaphores[image_index.0]];

    let submit_info = vk::SubmitInfo::default()
        .wait_semaphores(wait_semaphores)
        .wait_dst_stage_mask(wait_dst_stage_mask)
        .command_buffers(command_buffers)
        .signal_semaphores(signal_semaphores);

    unsafe {
        rc.device
            .queue_submit(
                rc.queue,
                &[submit_info],
                rc.in_flight_fences[rc.frame_index],
            )
            .unwrap();
    }

    let wait_semaphores = &[rc.render_finished_semaphores[image_index.0]];
    let swapchains = &[rc.swapchain.0];
    let image_indices = &[image_index.0 as u32];
    let present_info = vk::PresentInfoKHR::default()
        .wait_semaphores(wait_semaphores)
        .swapchains(swapchains)
        .image_indices(image_indices);

    match unsafe { rc.swapchain.1.queue_present(rc.queue, &present_info) } {
        Ok(false) => {}
        Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
            rc.swapchain_ok = false;
        }
        e => panic!("queue_present error: {e:?}"),
    }

    rc.frame_index = (rc.frame_index + 1) % MAX_FRAMES_IN_FLIGHT as usize
}

// fn render(
//     mut render_context: ResMut<RenderContext>,
//     _windows: Res<AppWindows>,
//     time: Res<Time>,
//     camera: Single<(&Camera, &Transform)>,
// ) {
//     // render_context.draw_frame(time.elapsed_secs_wrapped(), camera.into_inner());
// }

// fn resize(
//     mut render_context: ResMut<RenderContext>,
//     mut winit_events: MessageReader<RawWinitWindowEvent>,
//     _windows: Res<AppWindows>,
//     time: Res<Time>,
//     camera: Single<(&Camera, &Transform)>,
// ) {
//     let camera_data = camera.into_inner();
//     for event in winit_events.read() {
//         match event.event {
//             WindowEvent::Resized(size) => {
//                 render_context.recreate_window_dependent_objects(size);
//                 render_context.swapchain_ok = true;
//                 render_context.draw_frame(time.elapsed_secs_wrapped(), camera_data);
//             }
//             _ => {}
//         }
//     }
// }

// fn destroy_render_context(world: &mut World) {
//     let mut rc = world.remove_resource::<RenderContext>().unwrap();
//     unsafe {
//         rc.device.device_wait_idle().unwrap();

//         for image_view in &rc.swapchain_image_views {
//             rc.device.destroy_image_view(*image_view, None);
//         }

//         // for image in &render_context.swapchain_images {
//         //     render_context.device.destroy_image(*image, None);
//         // }

//         rc.swapchain.1.destroy_swapchain(rc.swapchain.0, None);

//         rc.surface.1.destroy_surface(rc.surface.0, None);

//         rc.debug_utils_loader
//             .destroy_debug_utils_messenger(rc.debug_callback, None);

//         // Drop allocator before destroying device because it holds memory
//         drop(rc.allocator);

//         rc.device.destroy_device(None);
//         rc.instance.destroy_instance(None);
//     }
// }

// mod foo {
//     fn create_depth_attachment(
//         instance: &Instance,
//         device: &Device,
//         allocator: &mut Allocator,
//         physical_device: vk::PhysicalDevice,
//         size: PhysicalSize<u32>,
//     ) -> (vk::Image, Allocation, vk::ImageView, vk::Format) {
//         let depth_format_list = &[
//             vk::Format::D32_SFLOAT_S8_UINT,
//             vk::Format::D24_UNORM_S8_UINT,
//         ];
//         let mut depth_format = vk::Format::UNDEFINED;
//         for format in depth_format_list {
//             let mut format_properties = vk::FormatProperties2::default();
//             unsafe {
//                 instance.get_physical_device_format_properties2(
//                     physical_device,
//                     *format,
//                     &mut format_properties,
//                 )
//             };
//             if format_properties
//                 .format_properties
//                 .optimal_tiling_features
//                 .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
//             {
//                 depth_format = *format;
//                 break;
//             }
//         }

//         let depth_image_info = vk::ImageCreateInfo::default()
//             .image_type(vk::ImageType::TYPE_2D)
//             .format(depth_format)
//             .extent(
//                 vk::Extent3D::default()
//                     .width(size.width)
//                     .height(size.height)
//                     .depth(1),
//             )
//             .mip_levels(1)
//             .array_layers(1)
//             .samples(vk::SampleCountFlags::TYPE_1)
//             .tiling(vk::ImageTiling::OPTIMAL)
//             .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
//             .initial_layout(vk::ImageLayout::UNDEFINED);

//         let depth_image = unsafe { device.create_image(&depth_image_info, None).unwrap() };
//         let mem_req = unsafe { device.get_image_memory_requirements(depth_image) };

//         let allocation = allocator
//             .allocate(&AllocationCreateDesc {
//                 name: "Depth image allocation",
//                 requirements: mem_req,
//                 location: MemoryLocation::GpuOnly,
//                 linear: false,
//                 allocation_scheme: AllocationScheme::GpuAllocatorManaged,
//             })
//             .unwrap();

//         unsafe {
//             device
//                 .bind_image_memory(depth_image, allocation.memory(), allocation.offset())
//                 .unwrap()
//         }

//         let depth_image_view_info = vk::ImageViewCreateInfo::default()
//             .image(depth_image)
//             .view_type(vk::ImageViewType::TYPE_2D)
//             .format(depth_format)
//             .subresource_range(
//                 vk::ImageSubresourceRange::default()
//                     .aspect_mask(vk::ImageAspectFlags::DEPTH)
//                     .level_count(1)
//                     .layer_count(1),
//             );

//         let depth_image_view = unsafe {
//             device
//                 .create_image_view(&depth_image_view_info, None)
//                 .unwrap()
//         };

//         (depth_image, allocation, depth_image_view, depth_format)
//     }

//     fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
//         let bindings = &[
//             vk::DescriptorSetLayoutBinding::default()
//                 .binding(0)
//                 .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
//                 .descriptor_count(1)
//                 .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
//             vk::DescriptorSetLayoutBinding::default()
//                 .binding(1)
//                 .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
//                 .descriptor_count(1)
//                 .stage_flags(vk::ShaderStageFlags::FRAGMENT),
//         ];

//         let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);
//         unsafe {
//             device
//                 .create_descriptor_set_layout(&create_info, None)
//                 .unwrap()
//         }
//     }

//     fn create_graphics_pipeline(
//         device: &Device,
//         _swapchain_extent: vk::Extent2D,
//         swapchain_format: vk::SurfaceFormatKHR,
//         depth_format: vk::Format,
//         descriptor_set_layout: vk::DescriptorSetLayout,
//     ) -> (vk::Pipeline, vk::PipelineLayout) {
//         let mut shader_code = std::fs::File::open("./shaders/triangle.spv").unwrap();
//         let mut buf = vec![];
//         shader_code.read_to_end(&mut buf).unwrap();
//         let shader_module = create_shader_module(device, &buf);

//         let vertex_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
//             .stage(vk::ShaderStageFlags::VERTEX)
//             .module(shader_module)
//             .name(c"vertMain");

//         let fragment_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
//             .stage(vk::ShaderStageFlags::FRAGMENT)
//             .module(shader_module)
//             .name(c"fragMain");

//         let shader_stages = [vertex_shader_stage_info, fragment_shader_stage_info];

//         let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

//         let pipeline_dynamic_state_create_info =
//             vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

//         let vertex_binding_descriptions = &[Vertex::get_binding_description()];
//         let vertex_attribute_descriptions = &Vertex::get_attribute_descriptions();
//         let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
//             .vertex_binding_descriptions(vertex_binding_descriptions)
//             .vertex_attribute_descriptions(vertex_attribute_descriptions);
//         let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
//             .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

//         let pipeline_viewport_state_create_info = vk::PipelineViewportStateCreateInfo::default()
//             .scissor_count(1)
//             .viewport_count(1);

//         let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
//             .depth_clamp_enable(false)
//             .rasterizer_discard_enable(false)
//             .polygon_mode(vk::PolygonMode::FILL)
//             .cull_mode(vk::CullModeFlags::BACK)
//             .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
//             .depth_bias_enable(false)
//             .line_width(1.0);

//         let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
//             .rasterization_samples(vk::SampleCountFlags::TYPE_1)
//             .sample_shading_enable(false);

//         let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
//             .blend_enable(false)
//             .color_write_mask(vk::ColorComponentFlags::RGBA);

//         let attachments = &[color_blend_attachment];
//         let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
//             .logic_op_enable(false)
//             .logic_op(vk::LogicOp::COPY)
//             .attachments(attachments);

//         let set_layouts = &[descriptor_set_layout];
//         let pipeline_layout_create_info =
//             vk::PipelineLayoutCreateInfo::default().set_layouts(set_layouts);
//         let pipeline_layout = unsafe {
//             device
//                 .create_pipeline_layout(&pipeline_layout_create_info, None)
//                 .unwrap()
//         };

//         let formats = &[swapchain_format.format];
//         let mut pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::default()
//             .color_attachment_formats(formats)
//             .depth_attachment_format(depth_format);

//         let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
//             .depth_test_enable(true)
//             .depth_write_enable(true)
//             .depth_compare_op(vk::CompareOp::LESS)
//             .depth_bounds_test_enable(false)
//             .stencil_test_enable(false);

//         let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
//             .stages(&shader_stages)
//             .vertex_input_state(&vertex_input_info)
//             .input_assembly_state(&input_assembly)
//             .viewport_state(&pipeline_viewport_state_create_info)
//             .rasterization_state(&rasterizer)
//             .multisample_state(&multisampling)
//             .color_blend_state(&color_blend)
//             .dynamic_state(&pipeline_dynamic_state_create_info)
//             .layout(pipeline_layout)
//             .depth_stencil_state(&depth_stencil)
//             .push_next(&mut pipeline_rendering_create_info);

//         let graphics_pipeline = unsafe {
//             device
//                 .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
//                 .unwrap()
//         };

//         unsafe { device.destroy_shader_module(shader_module, None) };

//         (graphics_pipeline[0], pipeline_layout)
//     }

//     fn create_command_pool(device: &Device, queue_family_index: u32) -> vk::CommandPool {
//         let create_info = vk::CommandPoolCreateInfo::default()
//             .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
//             .queue_family_index(queue_family_index);

//         unsafe { device.create_command_pool(&create_info, None).unwrap() }
//     }

//     fn create_texture_image(
//         device: &Device,
//         allocator: &mut Allocator,
//         command_pool: &vk::CommandPool,
//         queue: &vk::Queue,
//     ) -> (vk::Image, Allocation, vk::ImageView) {
//         let image = image::ImageReader::open("paint.jpg")
//             .unwrap()
//             .decode()
//             .unwrap();
//         let width = image.width();
//         let height = image.height();
//         let image_flat = image.into_rgba8();

//         let buf = image_flat.to_vec();
//         let size = dbg!(size_of_val(buf.as_slice())) as vk::DeviceSize;

//         let format = vk::Format::R8G8B8A8_SRGB;

//         let (staging_buffer, mut staging_allocation) = create_buffer(
//             device,
//             allocator,
//             size,
//             vk::BufferUsageFlags::TRANSFER_SRC,
//             MemoryLocation::CpuToGpu,
//         );

//         presser::copy_from_slice_to_offset(buf.as_slice(), &mut staging_allocation, 0).unwrap();

//         let (image, image_allocation) = create_image(
//             device,
//             allocator,
//             width,
//             height,
//             format,
//             vk::ImageTiling::OPTIMAL,
//             vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
//             MemoryLocation::GpuOnly,
//         );

//         single_time_commands(device, command_pool, queue, |device, command_buffer| {
//             transition_image_layout(
//                 device,
//                 command_buffer,
//                 image,
//                 vk::ImageLayout::UNDEFINED,
//                 vk::ImageLayout::TRANSFER_DST_OPTIMAL,
//                 vk::AccessFlags2::empty(),
//                 vk::AccessFlags2::TRANSFER_WRITE,
//                 vk::PipelineStageFlags2::TOP_OF_PIPE,
//                 vk::PipelineStageFlags2::TRANSFER,
//                 vk::ImageAspectFlags::COLOR,
//             );

//             copy_buffer_to_image(device, command_buffer, staging_buffer, image, width, height);

//             transition_image_layout(
//                 device,
//                 command_buffer,
//                 image,
//                 vk::ImageLayout::TRANSFER_DST_OPTIMAL,
//                 vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
//                 vk::AccessFlags2::TRANSFER_WRITE,
//                 vk::AccessFlags2::SHADER_READ,
//                 vk::PipelineStageFlags2::TRANSFER,
//                 vk::PipelineStageFlags2::FRAGMENT_SHADER,
//                 vk::ImageAspectFlags::COLOR,
//             );
//         });

//         unsafe { device.queue_wait_idle(*queue).unwrap() };

//         allocator.free(staging_allocation).unwrap();
//         unsafe { device.destroy_buffer(staging_buffer, None) };

//         // create ImageView
//         let view_info = vk::ImageViewCreateInfo::default()
//             .image(image)
//             .view_type(vk::ImageViewType::TYPE_2D)
//             .format(format)
//             .subresource_range(
//                 vk::ImageSubresourceRange::default()
//                     .aspect_mask(vk::ImageAspectFlags::COLOR)
//                     .base_mip_level(0)
//                     .base_array_layer(0)
//                     .level_count(1)
//                     .layer_count(1),
//             );
//         let image_view = unsafe { device.create_image_view(&view_info, None).unwrap() };

//         (image, image_allocation, image_view)
//     }

//     fn create_texture_sampler(
//         device: &Device,
//         instance: &Instance,
//         physical_device: vk::PhysicalDevice,
//     ) -> vk::Sampler {
//         let properties = unsafe { instance.get_physical_device_properties(physical_device) };
//         let sampler_info = vk::SamplerCreateInfo::default()
//             .mag_filter(vk::Filter::LINEAR)
//             .min_filter(vk::Filter::LINEAR)
//             .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
//             .address_mode_u(vk::SamplerAddressMode::REPEAT)
//             .address_mode_v(vk::SamplerAddressMode::REPEAT)
//             .address_mode_w(vk::SamplerAddressMode::REPEAT)
//             .anisotropy_enable(true)
//             .max_anisotropy(properties.limits.max_sampler_anisotropy)
//             .compare_enable(false)
//             .compare_op(vk::CompareOp::ALWAYS)
//             .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
//             .unnormalized_coordinates(false)
//             .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
//             .mip_lod_bias(0.0)
//             .min_lod(0.0)
//             .max_lod(0.0);

//         unsafe { device.create_sampler(&sampler_info, None).unwrap() }
//     }

//     fn create_vertex_buffer(
//         device: &Device,
//         allocator: &mut Allocator,
//         model: &Model,
//     ) -> (vk::Buffer, Allocation) {
//         let vertices = model.vertices.as_slice();
//         let size = dbg!(size_of_val(vertices)) as u64;
//         let (vertex_buffer, mut vertex_buffer_allocation) = create_buffer(
//             device,
//             allocator,
//             size,
//             vk::BufferUsageFlags::VERTEX_BUFFER,
//             MemoryLocation::CpuToGpu,
//         );

//         let _copy_record =
//             presser::copy_from_slice_to_offset(vertices, &mut vertex_buffer_allocation, 0).unwrap();

//         (vertex_buffer, vertex_buffer_allocation)
//     }

//     fn create_index_buffer(
//         device: &Device,
//         allocator: &mut Allocator,
//         model: &Model,
//     ) -> (vk::Buffer, Allocation) {
//         let indices = model.indices.as_slice();

//         let size = size_of_val(indices) as u64;
//         let (index_buffer, mut index_buffer_allocation) = create_buffer(
//             device,
//             allocator,
//             size,
//             vk::BufferUsageFlags::INDEX_BUFFER,
//             MemoryLocation::CpuToGpu,
//         );

//         dbg!(indices.len());
//         let copy_record =
//             presser::copy_from_slice_to_offset(indices, &mut index_buffer_allocation, 0).unwrap();
//         dbg!(copy_record);

//         (index_buffer, index_buffer_allocation)
//     }

//     fn create_uniform_buffers(
//         device: &Device,
//         allocator: &mut Allocator,
//     ) -> (Vec<vk::Buffer>, Vec<Allocation>) {
//         let size = size_of::<ViewUniform>() as vk::DeviceSize;
//         (0..MAX_FRAMES_IN_FLIGHT)
//             .map(|_| {
//                 create_buffer(
//                     device,
//                     allocator,
//                     size,
//                     vk::BufferUsageFlags::UNIFORM_BUFFER,
//                     MemoryLocation::CpuToGpu,
//                 )
//             })
//             .unzip()
//     }

//     fn create_descriptor_pool(device: &Device) -> vk::DescriptorPool {
//         let descriptor_pool_sizes = &[
//             vk::DescriptorPoolSize::default()
//                 .ty(vk::DescriptorType::UNIFORM_BUFFER)
//                 .descriptor_count(MAX_FRAMES_IN_FLIGHT),
//             vk::DescriptorPoolSize::default()
//                 .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
//                 .descriptor_count(MAX_FRAMES_IN_FLIGHT),
//         ];

//         let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
//             .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
//             .max_sets(MAX_FRAMES_IN_FLIGHT)
//             .pool_sizes(descriptor_pool_sizes);

//         unsafe {
//             device
//                 .create_descriptor_pool(&descriptor_pool_create_info, None)
//                 .unwrap()
//         }
//     }

//     fn create_descriptor_sets(
//         device: &Device,
//         descriptor_set_layout: vk::DescriptorSetLayout,
//         descriptor_pool: vk::DescriptorPool,
//         uniform_buffers: &[vk::Buffer],
//         sampler: vk::Sampler,
//         texture_image_view: vk::ImageView,
//     ) -> Vec<vk::DescriptorSet> {
//         let set_layouts = vec![descriptor_set_layout; MAX_FRAMES_IN_FLIGHT as usize];
//         let alloc_info = vk::DescriptorSetAllocateInfo::default()
//             .descriptor_pool(descriptor_pool)
//             .set_layouts(&set_layouts);

//         let sets = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() };

//         for (buffer, set) in uniform_buffers.iter().zip(sets.iter()) {
//             let buffer_info = &[vk::DescriptorBufferInfo::default()
//                 .buffer(buffer.clone())
//                 .offset(0)
//                 .range(size_of::<ViewUniform>() as u64)];
//             let image_info = &[vk::DescriptorImageInfo::default()
//                 .sampler(sampler)
//                 .image_view(texture_image_view)
//                 .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

//             let write_desc_set = &[
//                 vk::WriteDescriptorSet::default()
//                     .dst_set(set.clone())
//                     .dst_binding(0)
//                     .dst_array_element(0)
//                     .descriptor_count(1)
//                     .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
//                     .buffer_info(buffer_info),
//                 vk::WriteDescriptorSet::default()
//                     .dst_set(set.clone())
//                     .dst_binding(1)
//                     .dst_array_element(0)
//                     .descriptor_count(1)
//                     .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
//                     .image_info(image_info),
//             ];

//             unsafe { device.update_descriptor_sets(write_desc_set, &[]) };
//         }

//         sets
//     }

//     fn create_command_buffers(
//         device: &Device,
//         command_pool: vk::CommandPool,
//     ) -> Vec<vk::CommandBuffer> {
//         let command_buffer_alloc_info = vk::CommandBufferAllocateInfo::default()
//             .command_pool(command_pool)
//             .level(vk::CommandBufferLevel::PRIMARY)
//             .command_buffer_count(MAX_FRAMES_IN_FLIGHT);

//         unsafe {
//             device
//                 .allocate_command_buffers(&command_buffer_alloc_info)
//                 .unwrap()
//         }
//     }

//     fn update_uniform_buffers(&mut self, time: f32, camera_data: (&Camera, &Transform)) {
//         let model = Mat4::from_quat(Quat::from_rotation_y(FRAC_PI_8 * time));
//         let view = Mat4::look_to_rh(
//             camera_data.1.position,
//             (camera_data.1.rotation * Vec3::NEG_Z).normalize(),
//             Vec3::Y,
//         );

//         let view_uniform = ViewUniform {
//             model,
//             view,
//             projection: camera_data.0.projection,
//             normal_matrix: Mat3::from_mat4(model).inverse().transpose(),
//             eye_pos: camera_data.1.position,
//             light_pos: vec3(15.0, 0.0, 0.0),
//         };

//         presser::copy_to_offset(
//             &view_uniform,
//             &mut self.uniform_buffers_allocations[self.frame_index],
//             0,
//         )
//         .unwrap();
//     }

//     fn record_command_buffer(&self, swapchain_image_index: usize) {
//         unsafe {
//             let command_buffer = self.command_buffers[self.frame_index];

//             self.device
//                 .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
//                 .unwrap();

//             transition_image_layout(
//                 &self.device,
//                 self.command_buffers[self.frame_index],
//                 self.swapchain_images[swapchain_image_index],
//                 vk::ImageLayout::UNDEFINED,
//                 vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
//                 vk::AccessFlags2::empty(),
//                 vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
//                 vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
//                 vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
//                 vk::ImageAspectFlags::COLOR,
//             );

//             transition_image_layout(
//                 &self.device,
//                 self.command_buffers[self.frame_index],
//                 self.depth_image,
//                 vk::ImageLayout::UNDEFINED,
//                 vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
//                 vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
//                 vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
//                 vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
//                     | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
//                 vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
//                     | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
//                 vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
//             );

//             let clear_value = vk::ClearValue {
//                 color: vk::ClearColorValue {
//                     float32: [0.0, 0.0, 0.0, 0.0],
//                 },
//             };
//             let clear_depth = vk::ClearValue {
//                 depth_stencil: vk::ClearDepthStencilValue::default().depth(1.0).stencil(0),
//             };

//             let attachment_info = vk::RenderingAttachmentInfo::default()
//                 .image_view(self.swapchain_image_views[swapchain_image_index])
//                 .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
//                 .load_op(vk::AttachmentLoadOp::CLEAR)
//                 .store_op(vk::AttachmentStoreOp::STORE)
//                 .clear_value(clear_value);

//             let depth_attachment = vk::RenderingAttachmentInfo::default()
//                 .image_view(self.depth_image_view)
//                 .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
//                 .load_op(vk::AttachmentLoadOp::CLEAR)
//                 .store_op(vk::AttachmentStoreOp::DONT_CARE)
//                 .clear_value(clear_depth);

//             let color_attachments = &[attachment_info];
//             let rendering_info = vk::RenderingInfo::default()
//                 .render_area(
//                     vk::Rect2D::default()
//                         .offset(vk::Offset2D::default().x(0).y(0))
//                         .extent(self.swapchain_extent),
//                 )
//                 .layer_count(1)
//                 .color_attachments(color_attachments)
//                 .depth_attachment(&depth_attachment);

//             self.device
//                 .cmd_begin_rendering(command_buffer, &rendering_info);

//             self.device.cmd_bind_pipeline(
//                 command_buffer,
//                 vk::PipelineBindPoint::GRAPHICS,
//                 self.graphics_pipeline,
//             );

//             self.device
//                 .cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer], &[0]);

//             self.device.cmd_bind_index_buffer(
//                 command_buffer,
//                 self.index_buffer,
//                 0,
//                 vk::IndexType::UINT32,
//             );

//             self.device.cmd_set_viewport(
//                 command_buffer,
//                 0,
//                 &[vk::Viewport::default()
//                     .x(0.0)
//                     .y(0.0)
//                     .width(self.swapchain_extent.width as f32)
//                     .height(self.swapchain_extent.height as f32)
//                     .min_depth(0.0)
//                     .max_depth(1.0)],
//             );
//             self.device.cmd_set_scissor(
//                 command_buffer,
//                 0,
//                 &[vk::Rect2D::default()
//                     .offset(vk::Offset2D::default())
//                     .extent(self.swapchain_extent)],
//             );

//             self.device.cmd_bind_descriptor_sets(
//                 command_buffer,
//                 vk::PipelineBindPoint::GRAPHICS,
//                 self.pipeline_layout,
//                 0,
//                 &[self.descriptor_sets[self.frame_index]],
//                 &[],
//             );

//             self.device
//                 .cmd_draw_indexed(command_buffer, self.index_count as u32, 1, 0, 0, 0);

//             self.device.cmd_end_rendering(command_buffer);

//             transition_image_layout(
//                 &self.device,
//                 self.command_buffers[self.frame_index],
//                 self.swapchain_images[swapchain_image_index],
//                 vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
//                 vk::ImageLayout::PRESENT_SRC_KHR,
//                 vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
//                 vk::AccessFlags2::empty(),
//                 vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
//                 vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
//                 vk::ImageAspectFlags::COLOR,
//             );

//             self.device.end_command_buffer(command_buffer).unwrap();
//         }
//     }

//     fn draw_frame(&mut self, elapsed_time: f32, camera_data: (&Camera, &Transform)) {
//         unsafe {
//             self.device
//                 .wait_for_fences(&[self.in_flight_fences[self.frame_index]], true, u64::MAX)
//                 .unwrap();

//             let image_index = match self.swapchain.1.acquire_next_image(
//                 self.swapchain.0,
//                 u64::MAX,
//                 self.present_complete_semaphores[self.frame_index],
//                 vk::Fence::null(),
//             ) {
//                 Ok((image_index, _)) => image_index,
//                 Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
//                     self.swapchain_ok = false;
//                     return;
//                 }
//                 Err(e) => {
//                     panic!("failed to aquire swapchain image: {e}");
//                 }
//             };

//             self.device
//                 .reset_fences(&[self.in_flight_fences[self.frame_index]])
//                 .unwrap();

//             self.update_uniform_buffers(elapsed_time, camera_data);

//             self.device
//                 .reset_command_buffer(
//                     self.command_buffers[self.frame_index],
//                     vk::CommandBufferResetFlags::empty(),
//                 )
//                 .unwrap();
//             self.record_command_buffer(image_index as usize);

//             let wait_destination_stage_mask = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
//             let wait_semaphores = &[self.present_complete_semaphores[self.frame_index]];
//             let wait_dst_stage_mask = &[wait_destination_stage_mask];
//             let command_buffers = &[self.command_buffers[self.frame_index]];
//             let signal_semaphores = &[self.render_finished_semaphores[image_index as usize]];

//             let submit_info = vk::SubmitInfo::default()
//                 .wait_semaphores(wait_semaphores)
//                 .wait_dst_stage_mask(wait_dst_stage_mask)
//                 .command_buffers(command_buffers)
//                 .signal_semaphores(signal_semaphores);

//             self.device
//                 .queue_submit(
//                     self.queue,
//                     &[submit_info],
//                     self.in_flight_fences[self.frame_index],
//                 )
//                 .unwrap();

//             let wait_semaphores = &[self.render_finished_semaphores[image_index as usize]];
//             let swapchains = &[self.swapchain.0];
//             let image_indices = &[image_index];
//             let present_info = vk::PresentInfoKHR::default()
//                 .wait_semaphores(wait_semaphores)
//                 .swapchains(swapchains)
//                 .image_indices(image_indices);

//             match self.swapchain.1.queue_present(self.queue, &present_info) {
//                 Ok(false) => {}
//                 Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
//                     self.swapchain_ok = false;
//                 }
//                 e => panic!("queue_present error: {e:?}"),
//             }

//             self.frame_index = (self.frame_index + 1) % MAX_FRAMES_IN_FLIGHT as usize;
//         }
//     }

//     fn cleanup_swapchain(&self) {
//         unsafe {
//             for image_view in &self.swapchain_image_views {
//                 self.device.destroy_image_view(*image_view, None);
//             }

//             self.swapchain.1.destroy_swapchain(self.swapchain.0, None);
//         }
//     }

//     fn recreate_window_dependent_objects(&mut self, size: PhysicalSize<u32>) {
//         unsafe {
//             self.device.device_wait_idle().unwrap();
//             self.cleanup_swapchain();

//             let (
//                 swaphain,
//                 swapchain_device,
//                 swapchain_images,
//                 swapchain_surface_format,
//                 swapchain_extent,
//             ) = Self::create_swapchain(
//                 &self.instance,
//                 &self.device,
//                 self.physical_device,
//                 &self.surface,
//                 size,
//             );
//             let image_views =
//                 Self::create_image_views(&self.device, swapchain_surface_format, &swapchain_images);

//             self.swapchain = (swaphain, swapchain_device);
//             self.swapchain_images = swapchain_images;
//             self.swapchain_surface_format = swapchain_surface_format;
//             self.swapchain_extent = swapchain_extent;
//             self.swapchain_image_views = image_views;
//         }
//     }
// }
