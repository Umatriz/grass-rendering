use std::{borrow::Cow, ffi::CStr, io::Read, sync::Arc};

use ash::{Device, Entry, Instance, ext, khr, vk};
use bevy_app::{Plugin, PostUpdate, Startup};
use bevy_ecs::{
    resource::Resource,
    schedule::ScheduleLabel,
    system::{Commands, Res},
};
use itertools::Itertools;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use tracing::{Level, error, event, info, span, trace, trace_span, warn};
use winit::{event_loop::OwnedDisplayHandle, window::Window};

use crate::windowing::{AppWindows, WinitOwnedDisplayHandle};

pub struct RenderingPlugin;

impl Plugin for RenderingPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(Startup, setup_render_context);
        app.add_systems(PostUpdate, render);
        app.add_systems(CleanUp, destroy_render_context);
    }
}

/// This schedule runs after the window is closed but before `world.clear_all()`.
#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone)]
pub struct CleanUp;

fn setup_render_context(
    mut commands: Commands,
    windows: Res<AppWindows>,
    display_handle: Res<WinitOwnedDisplayHandle>,
) {
    let render_context = RenderContext::new(windows.primary.clone(), display_handle.0.clone());
    commands.insert_resource(render_context);
    info!("Render context was successfully created");
}

fn render(render_context: Res<RenderContext>) {
    render_context.draw_frame();
}

fn destroy_render_context(render_context: Res<RenderContext>) {
    unsafe {
        render_context.device.device_wait_idle().unwrap();

        render_context
            .device
            .destroy_fence(render_context.draw_fence, None);

        render_context
            .device
            .destroy_semaphore(render_context.present_complete_semaphore, None);

        render_context
            .device
            .destroy_semaphore(render_context.render_finished_semaphore, None);

        render_context
            .device
            .destroy_command_pool(render_context.command_pool, None);

        render_context
            .device
            .destroy_pipeline_layout(render_context.pipeline_layout, None);

        render_context
            .device
            .destroy_pipeline(render_context.graphics_pipeline, None);

        for image_view in &render_context.swapchain_image_views {
            render_context.device.destroy_image_view(*image_view, None);
        }

        // for image in &render_context.swapchain_images {
        //     render_context.device.destroy_image(*image, None);
        // }

        render_context
            .swapchain
            .1
            .destroy_swapchain(render_context.swapchain.0, None);

        render_context
            .surface
            .1
            .destroy_surface(render_context.surface.0, None);

        render_context
            .debug_utils_loader
            .destroy_debug_utils_messenger(render_context.debug_callback, None);

        render_context.device.destroy_device(None);
        render_context.instance.destroy_instance(None);
    }
}

#[derive(Resource)]
pub struct RenderContext {
    entry: ash::Entry,
    instance: ash::Instance,

    debug_utils_loader: ext::debug_utils::Instance,
    debug_callback: vk::DebugUtilsMessengerEXT,

    surface: (vk::SurfaceKHR, khr::surface::Instance),

    physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    queue: vk::Queue,

    swapchain: (vk::SwapchainKHR, khr::swapchain::Device),
    swapchain_images: Vec<vk::Image>,
    swapchain_surface_format: vk::SurfaceFormatKHR,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,

    graphics_pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,

    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,

    present_complete_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    draw_fence: vk::Fence,
}

impl RenderContext {
    fn new(window: Arc<Window>, display_hadle: OwnedDisplayHandle) -> Self {
        unsafe {
            let entry = ash::Entry::load().unwrap();

            let mut windowing_extension_names = ash_window::enumerate_required_extensions(
                display_hadle.display_handle().unwrap().as_raw(),
            )
            .unwrap();

            let mut extension_names =
                vec![khr::portability_enumeration::NAME, ext::debug_utils::NAME];

            let extension_properties = entry.enumerate_instance_extension_properties(None).unwrap();

            for ext in &extension_names {
                if !extension_properties
                    .iter()
                    .any(|prop| prop.extension_name_as_c_str().unwrap() == ext)
                {
                    panic!(
                        "Required extension not supported: {}",
                        ext.to_string_lossy()
                    );
                }
            }

            let mut extension_names = extension_names
                .into_iter()
                .map(|ext| ext.as_ptr())
                .collect_vec();
            extension_names.extend(windowing_extension_names);

            let validation_layers = vec![c"VK_LAYER_KHRONOS_validation"];

            let layer_properties = entry.enumerate_instance_layer_properties().unwrap();
            for layer in &validation_layers {
                if !layer_properties
                    .iter()
                    .any(|prop| prop.layer_name_as_c_str().unwrap() == layer)
                {
                    panic!("Required layer not supported: {}", layer.to_string_lossy());
                }
            }

            let validation_layers = validation_layers
                .into_iter()
                .map(|l| l.as_ptr())
                .collect_vec();

            let app_info = vk::ApplicationInfo::default()
                .api_version(vk::API_VERSION_1_3)
                .application_name(c"Grass Rendering")
                .engine_name(c"No Engine");

            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR)
                .enabled_extension_names(&extension_names)
                .enabled_layer_names(&validation_layers);

            let instance = entry.create_instance(&create_info, None).unwrap();

            let (debug_utils_loader, debug_callback) =
                Self::setup_debug_messenger(&entry, &instance);

            let surface = Self::create_surface(&entry, &instance, display_hadle, window.clone());
            let surface_instance = khr::surface::Instance::new(&entry, &instance);
            let surface = (surface, surface_instance);

            let physical_device = Self::pick_physical_device(&instance);
            let (device, queue, queue_family_index) =
                Self::create_logical_device(&instance, physical_device, &surface);

            let (
                swaphain,
                swapchain_device,
                swapchain_images,
                swapchain_surface_format,
                swapchain_extent,
            ) = Self::create_swapchain(&instance, &device, physical_device, &surface, &window);

            let swapchain_image_views =
                Self::create_image_views(&device, swapchain_surface_format, &swapchain_images);

            let (graphics_pipeline, pipeline_layout) =
                Self::create_graphics_pipeline(&device, swapchain_extent, swapchain_surface_format);

            let command_pool = Self::create_command_pool(&device, queue_family_index);
            let command_buffer = Self::create_command_buffer(&device, command_pool);
            let (present_complete_semaphore, render_finished_semaphore, draw_fence) =
                Self::create_sync_objects(&device);

            Self {
                entry,
                instance,
                debug_utils_loader,
                debug_callback,
                surface,
                physical_device,
                device,
                queue,
                swapchain: (swaphain, swapchain_device),
                swapchain_images,
                swapchain_surface_format,
                swapchain_extent,
                swapchain_image_views,
                graphics_pipeline,
                pipeline_layout,
                command_pool,
                command_buffer,
                present_complete_semaphore,
                render_finished_semaphore,
                draw_fence,
            }
        }
    }

    fn create_surface(
        entry: &Entry,
        instance: &Instance,
        display_handle: OwnedDisplayHandle,
        window: Arc<Window>,
    ) -> vk::SurfaceKHR {
        unsafe {
            ash_window::create_surface(
                entry,
                instance,
                display_handle.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .unwrap()
        }
    }

    fn setup_debug_messenger(
        entry: &Entry,
        instance: &Instance,
    ) -> (ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT) {
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let debug_utils_instance = ext::debug_utils::Instance::new(&entry, &instance);
        let debug_messenger =
            unsafe { debug_utils_instance.create_debug_utils_messenger(&debug_info, None) }
                .unwrap();

        (debug_utils_instance, debug_messenger)
    }

    fn pick_physical_device(instance: &ash::Instance) -> vk::PhysicalDevice {
        fn device_score(instance: &Instance, physical_device: vk::PhysicalDevice) -> Option<u32> {
            unsafe {
                let device_properties = instance.get_physical_device_properties(physical_device);
                let supports_vulkan_1_3 = device_properties.api_version >= vk::API_VERSION_1_3;

                let queue_families =
                    instance.get_physical_device_queue_family_properties(physical_device);
                let supports_graphics = queue_families
                    .into_iter()
                    .any(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS));

                let required_device_extensions = [khr::swapchain::NAME];
                let available_device_extensions = instance
                    .enumerate_device_extension_properties(physical_device)
                    .unwrap();
                let supports_all_required_extensions =
                    required_device_extensions.into_iter().all(|ext| {
                        available_device_extensions
                            .iter()
                            .any(|prop| prop.extension_name_as_c_str().unwrap() == ext)
                    });

                let mut vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features::default();
                let mut extended_dynamic_state =
                    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default();

                let mut features = vk::PhysicalDeviceFeatures2::default()
                    .push_next(&mut extended_dynamic_state)
                    .push_next(&mut vulkan_1_3_features);
                instance.get_physical_device_features2(physical_device, &mut features);

                let supports_required_features = extended_dynamic_state.extended_dynamic_state
                    == vk::TRUE
                    && vulkan_1_3_features.dynamic_rendering == vk::TRUE
                    && vulkan_1_3_features.synchronization2 == vk::TRUE;

                if supports_vulkan_1_3
                    && supports_graphics
                    && supports_all_required_extensions
                    && supports_required_features
                {
                    let score = match device_properties.device_type {
                        vk::PhysicalDeviceType::DISCRETE_GPU => 4,
                        vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
                        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                        vk::PhysicalDeviceType::CPU => 1,
                        vk::PhysicalDeviceType::OTHER => 0,
                        _ => unreachable!(),
                    };
                    Some(score)
                } else {
                    None
                }
            }
        }

        unsafe {
            let physical_devices = instance.enumerate_physical_devices().unwrap();
            let device = physical_devices
                .into_iter()
                .max_by_key(|device| device_score(instance, *device));

            device.unwrap()
        }
    }

    fn create_logical_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: &(vk::SurfaceKHR, khr::surface::Instance),
    ) -> (ash::Device, vk::Queue, u32) {
        unsafe {
            let queue_family_properties =
                instance.get_physical_device_queue_family_properties(physical_device);
            let (graphics_index, graphics_queue_family_property) = queue_family_properties
                .iter()
                .enumerate()
                .find(|(idx, props)| {
                    props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && surface
                            .1
                            .get_physical_device_surface_support(
                                physical_device,
                                *idx as u32,
                                surface.0,
                            )
                            .unwrap()
                })
                .unwrap();

            let device_queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(graphics_index as u32)
                .queue_priorities(&[0.5]);

            let mut vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features::default()
                .dynamic_rendering(true)
                .synchronization2(true);
            let mut extended_dynamic_state_features =
                vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default()
                    .extended_dynamic_state(true);
            let mut physical_device_features_2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut vulkan_1_3_features)
                .push_next(&mut extended_dynamic_state_features);

            // TODO: the same list is used when picking physical device
            let required_device_extensions = [khr::swapchain::NAME.as_ptr()];

            let queue_create_infos = &[device_queue_create_info];
            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(queue_create_infos)
                .enabled_extension_names(&required_device_extensions)
                .push_next(&mut physical_device_features_2);

            let device = instance
                .create_device(physical_device, &device_create_info, None)
                .unwrap();

            let queue = device.get_device_queue(graphics_index as u32, 0);

            (device, queue, graphics_index as u32)
        }
    }

    fn create_swapchain(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        surface: &(vk::SurfaceKHR, khr::surface::Instance),
        window: &Window,
    ) -> (
        vk::SwapchainKHR,
        khr::swapchain::Device,
        Vec<vk::Image>,
        vk::SurfaceFormatKHR,
        vk::Extent2D,
    ) {
        unsafe {
            let surface_capabilities = surface
                .1
                .get_physical_device_surface_capabilities(physical_device, surface.0)
                .unwrap();

            let swapchain_extent = Self::choose_swapchain_extent(surface_capabilities, window);
            let min_image_count = Self::choose_swapchain_min_image_count(surface_capabilities);

            let present_modes = surface
                .1
                .get_physical_device_surface_present_modes(physical_device, surface.0)
                .unwrap();

            let formats = surface
                .1
                .get_physical_device_surface_formats(physical_device, surface.0)
                .unwrap();

            let format = Self::choose_swapchain_format(&formats);
            let present_mode = Self::choose_swapchain_present_mode();

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(surface.0)
                .min_image_count(min_image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(swapchain_extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(surface_capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true);

            let swapchain_device = khr::swapchain::Device::new(instance, device);
            let swapchain = swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();
            let swapchain_images = swapchain_device.get_swapchain_images(swapchain).unwrap();

            (
                swapchain,
                swapchain_device,
                swapchain_images,
                format,
                swapchain_extent,
            )
        }
    }

    fn choose_swapchain_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        *formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&formats[0])
    }

    fn choose_swapchain_present_mode() -> vk::PresentModeKHR {
        vk::PresentModeKHR::FIFO
    }

    fn choose_swapchain_extent(
        capabilities: vk::SurfaceCapabilitiesKHR,
        window: &Window,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }

        let size = window.inner_size();

        vk::Extent2D::default()
            .width(size.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ))
            .height(size.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ))
    }

    fn choose_swapchain_min_image_count(surface_capabilities: vk::SurfaceCapabilitiesKHR) -> u32 {
        let mut min_image_count = surface_capabilities.min_image_count.max(3);
        if surface_capabilities.max_image_count > 0
            && surface_capabilities.max_image_count < min_image_count
        {
            min_image_count = surface_capabilities.max_image_count;
        }
        min_image_count
    }

    fn create_image_views(
        device: &Device,
        swapchain_format: vk::SurfaceFormatKHR,
        swapchain_images: &[vk::Image],
    ) -> Vec<vk::ImageView> {
        let mut image_view_create_info = vk::ImageViewCreateInfo::default()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(swapchain_format.format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let mut image_views = vec![];
        for image in swapchain_images {
            image_view_create_info = image_view_create_info.image(*image);
            let image_view = unsafe {
                device
                    .create_image_view(&image_view_create_info, None)
                    .unwrap()
            };
            image_views.push(image_view);
        }

        image_views
    }

    fn create_graphics_pipeline(
        device: &Device,
        swapchain_extent: vk::Extent2D,
        swapchain_format: vk::SurfaceFormatKHR,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let mut shader_code = std::fs::File::open("./shaders/slang.spv").unwrap();
        let mut buf = vec![];
        shader_code.read_to_end(&mut buf).unwrap();
        let shader_module = create_shader_module(device, &buf);

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
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
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
            .front_face(vk::FrontFace::CLOCKWISE)
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

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .unwrap()
        };

        let formats = &[swapchain_format.format];
        let mut pipeline_rendering_create_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(formats);

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
            .push_next(&mut pipeline_rendering_create_info);

        let graphics_pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .unwrap()
        };

        unsafe { device.destroy_shader_module(shader_module, None) };

        (graphics_pipeline[0], pipeline_layout)
    }

    fn create_command_pool(device: &Device, queue_family_index: u32) -> vk::CommandPool {
        let create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);

        unsafe { device.create_command_pool(&create_info, None).unwrap() }
    }

    fn create_command_buffer(device: &Device, command_pool: vk::CommandPool) -> vk::CommandBuffer {
        let command_buffer_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let buffer = unsafe {
            device
                .allocate_command_buffers(&command_buffer_alloc_info)
                .unwrap()[0]
        };

        buffer
    }

    fn record_command_buffer(&self, swapchain_image_index: usize) {
        unsafe {
            self.device
                .begin_command_buffer(self.command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();

            self.transition_image_layout(
                swapchain_image_index,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            );

            let clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            };

            let attachment_info = vk::RenderingAttachmentInfo::default()
                .image_view(self.swapchain_image_views[swapchain_image_index])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_value);

            let color_attachments = &[attachment_info];
            let rendering_info = vk::RenderingInfo::default()
                .render_area(
                    vk::Rect2D::default()
                        .offset(vk::Offset2D::default().x(0).y(0))
                        .extent(self.swapchain_extent),
                )
                .layer_count(1)
                .color_attachments(color_attachments);

            self.device
                .cmd_begin_rendering(self.command_buffer, &rendering_info);

            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );

            self.device.cmd_set_viewport(
                self.command_buffer,
                0,
                &[vk::Viewport::default()
                    .x(0.0)
                    .y(0.0)
                    .width(self.swapchain_extent.width as f32)
                    .height(self.swapchain_extent.height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0)],
            );
            self.device.cmd_set_scissor(
                self.command_buffer,
                0,
                &[vk::Rect2D::default()
                    .offset(vk::Offset2D::default())
                    .extent(self.swapchain_extent)],
            );

            self.device.cmd_draw(self.command_buffer, 3, 1, 0, 0);

            self.device.cmd_end_rendering(self.command_buffer);

            self.transition_image_layout(
                swapchain_image_index,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            );

            self.device.end_command_buffer(self.command_buffer).unwrap();
        }
    }

    fn transition_image_layout(
        &self,
        image_index: usize,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_access_mask: vk::AccessFlags2,
        dst_access_mask: vk::AccessFlags2,
        src_stage_mask: vk::PipelineStageFlags2,
        dst_stage_mask: vk::PipelineStageFlags2,
    ) {
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage_mask)
            .src_access_mask(src_access_mask)
            .dst_stage_mask(dst_stage_mask)
            .dst_access_mask(dst_access_mask)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.swapchain_images[image_index])
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let image_memory_barriers = &[barrier];
        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(image_memory_barriers);
        unsafe {
            self.device
                .cmd_pipeline_barrier2(self.command_buffer, &dependency_info)
        };
    }

    fn create_sync_objects(device: &Device) -> (vk::Semaphore, vk::Semaphore, vk::Fence) {
        unsafe {
            (
                device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .unwrap(),
                device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .unwrap(),
                device
                    .create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                    .unwrap(),
            )
        }
    }

    fn draw_frame(&self) {
        unsafe {
            self.device
                .wait_for_fences(&[self.draw_fence], true, u64::MAX)
                .unwrap();
            self.device.reset_fences(&[self.draw_fence]).unwrap();

            let (image_index, is_suboptimal) = self
                .swapchain
                .1
                .acquire_next_image(
                    self.swapchain.0,
                    u64::MAX,
                    self.present_complete_semaphore,
                    vk::Fence::null(),
                )
                .unwrap();

            self.record_command_buffer(image_index as usize);

            let wait_destination_stage_mask = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
            let wait_semaphores = &[self.present_complete_semaphore];
            let wait_dst_stage_mask = &[wait_destination_stage_mask];
            let command_buffers = &[self.command_buffer];
            let signal_semaphores = &[self.render_finished_semaphore];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(wait_dst_stage_mask)
                .command_buffers(command_buffers)
                .signal_semaphores(signal_semaphores);

            self.device
                .queue_submit(self.queue, &[submit_info], self.draw_fence)
                .unwrap();

            let wait_semaphores = &[self.render_finished_semaphore];
            let swapchains = &[self.swapchain.0];
            let image_indices = &[image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(wait_semaphores)
                .swapchains(swapchains)
                .image_indices(image_indices);

            self.swapchain
                .1
                .queue_present(self.queue, &present_info)
                .unwrap();
        }
    }
}

fn create_shader_module(device: &Device, buf: &[u8]) -> vk::ShaderModule {
    let code: &[u32] = bytemuck::cast_slice(buf);
    let create_info = vk::ShaderModuleCreateInfo::default().code(code);

    unsafe { device.create_shader_module(&create_info, None).unwrap() }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message_id_name) }.to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message) }.to_string_lossy()
    };

    // println!(
    //     "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
    // );

    // let span = span!(Level::ERROR, "VULKAN");
    // let _guard = span.enter();

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            trace!(target: "VULKAN", "{message_type:?} [{message_id_name} {message_id_number}]: {message}")
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            info!(target: "VULKAN", "{message_type:?} [{message_id_name} {message_id_number}]: {message}")
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            warn!(target: "VULKAN", "{message_type:?} [{message_id_name} {message_id_number}]: {message}")
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            error!(target: "VULKAN", "{message_type:?} [{message_id_name} {message_id_number}]: {message}");
        }
        _ => {
            warn!(target: "VULKAN", "{message_type:?} [{message_id_name} {message_id_number}]: {message}")
        }
    };

    vk::FALSE
}
