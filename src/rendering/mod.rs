use std::{
    borrow::Cow,
    f32::consts::{FRAC_PI_2, FRAC_PI_4},
    ffi::CStr,
    io::Read,
    mem,
    sync::Arc,
};

use ash::{Device, Entry, Instance, ext, khr, vk};
use bevy_app::{Plugin, PostUpdate, Startup};
use bevy_ecs::{
    message::MessageReader,
    resource::Resource,
    schedule::{IntoScheduleConfigs, ScheduleLabel},
    system::{Commands, Res, ResMut},
    world::World,
};
use bevy_time::Time;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3, vec3};
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings, MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};
use itertools::Itertools;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use tracing::{error, info, trace, warn};
use winit::{
    dpi::PhysicalSize, event::WindowEvent, event_loop::OwnedDisplayHandle, window::Window,
};

use crate::windowing::{AppWindows, RawWinitWindowEvent, WinitOwnedDisplayHandle};

pub const MAX_FRAMES_IN_FLIGHT: u32 = 2;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct ViewUniform {
    model: Mat4,
    view: Mat4,
    projection: Mat4,

    light_pos: Vec3,
}

#[derive(Pod, Zeroable, Clone, Copy, Debug)]
#[repr(C)]
pub struct Vertex {
    pos: Vec3,
    color: Vec3,
    normal: Vec3,
}

impl Vertex {
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(mem::offset_of!(Vertex, pos) as u32),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(mem::offset_of!(Vertex, color) as u32),
            vk::VertexInputAttributeDescription::default()
                .location(2)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(mem::offset_of!(Vertex, normal) as u32),
        ]
    }
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
        app.add_systems(Startup, (load_model, setup_render_context).chain());
        app.add_systems(PostUpdate, (resize, render).chain());
        app.add_systems(CleanUp, destroy_render_context);
    }
}

/// This schedule runs after the window is closed but before `world.clear_all()`.
#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone)]
pub struct CleanUp;

#[derive(Resource)]
pub struct Model(tobj::Model);

fn load_model(mut commands: Commands) {
    let (mut models, _materials) = tobj::load_obj(
        "monkey.obj",
        &tobj::LoadOptions {
            triangulate: true,
            ..Default::default()
        },
    )
    .unwrap();

    // println!("{models:#?}");
    // println!("{materials:#?}");

    commands.insert_resource(Model(models.pop().unwrap()));
}

fn setup_render_context(
    mut commands: Commands,
    windows: Res<AppWindows>,
    display_handle: Res<WinitOwnedDisplayHandle>,
    model: Res<Model>,
) {
    let render_context =
        RenderContext::new(windows.primary.clone(), display_handle.0.clone(), &model.0);
    commands.insert_resource(render_context);
    info!("Render context was successfully created");
}

fn render(mut render_context: ResMut<RenderContext>, _windows: Res<AppWindows>, time: Res<Time>) {
    render_context.draw_frame(time.elapsed_secs_wrapped());
}

fn resize(
    mut render_context: ResMut<RenderContext>,
    mut winit_events: MessageReader<RawWinitWindowEvent>,
    _windows: Res<AppWindows>,
    time: Res<Time>,
) {
    for event in winit_events.read() {
        match event.event {
            WindowEvent::Resized(size) => {
                render_context.recreate_window_dependent_objects(size);
                render_context.swapchain_ok = true;
                render_context.draw_frame(time.elapsed_secs_wrapped());
            }
            _ => {}
        }
    }
}

fn destroy_render_context(world: &mut World) {
    let mut rc = world.remove_resource::<RenderContext>().unwrap();
    unsafe {
        rc.device.device_wait_idle().unwrap();

        for fence in &rc.in_flight_fences {
            rc.device.destroy_fence(*fence, None);
        }

        for semaphore in &rc.present_complete_semaphores {
            rc.device.destroy_semaphore(*semaphore, None);
        }

        for semaphore in &rc.render_finished_semaphores {
            rc.device.destroy_semaphore(*semaphore, None);
        }

        rc.device.destroy_command_pool(rc.command_pool, None);

        rc.device.destroy_pipeline_layout(rc.pipeline_layout, None);

        rc.device.destroy_pipeline(rc.graphics_pipeline, None);

        for image_view in &rc.swapchain_image_views {
            rc.device.destroy_image_view(*image_view, None);
        }

        // for image in &render_context.swapchain_images {
        //     render_context.device.destroy_image(*image, None);
        // }

        rc.swapchain.1.destroy_swapchain(rc.swapchain.0, None);

        rc.surface.1.destroy_surface(rc.surface.0, None);

        rc.debug_utils_loader
            .destroy_debug_utils_messenger(rc.debug_callback, None);

        rc.allocator.free(rc.depth_image_allocation).unwrap();
        rc.device.destroy_image(rc.depth_image, None);
        rc.device.destroy_image_view(rc.depth_image_view, None);

        rc.allocator.free(rc.index_buffer_allocation).unwrap();
        rc.device.destroy_buffer(rc.index_buffer, None);

        rc.allocator.free(rc.vertex_buffer_allocation).unwrap();
        rc.device.destroy_buffer(rc.vertex_buffer, None);

        for (allocation, buffer) in rc
            .uniform_buffers_allocations
            .into_iter()
            .zip(rc.uniform_buffers)
        {
            rc.allocator.free(allocation).unwrap();
            rc.device.destroy_buffer(buffer, None);
        }

        rc.device
            .free_descriptor_sets(rc.descriptor_pool, &rc.descriptor_sets)
            .unwrap();
        rc.device
            .destroy_descriptor_set_layout(rc.descriptor_set_layout, None);
        rc.device.destroy_descriptor_pool(rc.descriptor_pool, None);

        // Drop allocator before destroying device because it holds memory
        drop(rc.allocator);

        rc.device.destroy_device(None);
        rc.instance.destroy_instance(None);
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
    min_memory_map_alignment: usize,
    pub device: ash::Device,
    queue: vk::Queue,

    allocator: Allocator,

    swapchain: (vk::SwapchainKHR, khr::swapchain::Device),
    swapchain_images: Vec<vk::Image>,
    swapchain_surface_format: vk::SurfaceFormatKHR,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,

    depth_image: vk::Image,
    depth_image_allocation: Allocation,
    depth_image_view: vk::ImageView,

    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,

    graphics_pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,

    vertex_buffer: vk::Buffer,
    vertex_buffer_allocation: Allocation,
    index_buffer: vk::Buffer,
    index_buffer_allocation: Allocation,
    index_count: usize,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_allocations: Vec<Allocation>,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    present_complete_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,

    frame_index: usize,
    swapchain_ok: bool,
}

impl RenderContext {
    fn new(window: Arc<Window>, display_hadle: OwnedDisplayHandle, model: &tobj::Model) -> Self {
        unsafe {
            let entry = ash::Entry::linked();

            let windowing_extension_names = ash_window::enumerate_required_extensions(
                display_hadle.display_handle().unwrap().as_raw(),
            )
            .unwrap();

            let extension_names = vec![ext::debug_utils::NAME];

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
                .enabled_extension_names(&extension_names)
                .enabled_layer_names(&validation_layers);

            let instance = entry.create_instance(&create_info, None).unwrap();

            let (debug_utils_loader, debug_callback) =
                Self::setup_debug_messenger(&entry, &instance);

            let surface = Self::create_surface(&entry, &instance, display_hadle, window.clone());
            let surface_instance = khr::surface::Instance::new(&entry, &instance);
            let surface = (surface, surface_instance);

            let (physical_device, min_memory_map_alignment) = Self::pick_physical_device(&instance);
            let (device, queue, queue_family_index) =
                Self::create_logical_device(&instance, physical_device, &surface);

            let mut allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: AllocatorDebugSettings::default(),
                buffer_device_address: false,
                allocation_sizes: AllocationSizes::default(),
            })
            .unwrap();

            let (
                swaphain,
                swapchain_device,
                swapchain_images,
                swapchain_surface_format,
                swapchain_extent,
            ) = Self::create_swapchain(
                &instance,
                &device,
                physical_device,
                &surface,
                window.inner_size(),
            );

            let swapchain_image_views =
                Self::create_image_views(&device, swapchain_surface_format, &swapchain_images);

            let (depth_image, depth_image_allocation, depth_image_view, depth_format) =
                Self::create_depth_attachment(
                    &instance,
                    &device,
                    &mut allocator,
                    physical_device,
                    window.inner_size(),
                );

            let descriptor_set_layout = Self::create_descriptor_set_layout(&device);
            let (uniform_buffers, uniform_buffers_allocations) =
                Self::create_uniform_buffers(&device, &mut allocator);
            let descriptor_pool = Self::create_descriptor_pool(&device);
            let descriptor_sets = Self::create_descriptor_sets(
                &device,
                descriptor_set_layout,
                descriptor_pool,
                &uniform_buffers,
            );

            let (graphics_pipeline, pipeline_layout) = Self::create_graphics_pipeline(
                &device,
                swapchain_extent,
                swapchain_surface_format,
                depth_format,
                descriptor_set_layout,
            );

            let command_pool = Self::create_command_pool(&device, queue_family_index);

            let (vertex_buffer, vertex_buffer_allocation) =
                Self::create_vertex_buffer(&device, &mut allocator, &model);
            let (index_buffer, index_buffer_allocation) =
                Self::create_index_buffer(&device, &mut allocator, &model);

            let command_buffers = Self::create_command_buffers(&device, command_pool);
            let (present_complete_semaphores, render_finished_semaphores, in_flight_fences) =
                Self::create_sync_objects(&device, swapchain_images.len());

            Self {
                entry,
                instance,

                debug_utils_loader,
                debug_callback,

                surface,

                physical_device,
                min_memory_map_alignment,

                device,
                queue,
                allocator,

                swapchain: (swaphain, swapchain_device),
                swapchain_images,
                swapchain_surface_format,
                swapchain_extent,
                swapchain_image_views,

                depth_image,
                depth_image_allocation,
                depth_image_view,

                descriptor_set_layout,
                descriptor_sets,
                descriptor_pool,

                graphics_pipeline,
                pipeline_layout,

                vertex_buffer,
                vertex_buffer_allocation,
                index_buffer,
                index_buffer_allocation,
                index_count: model.mesh.indices.len(),

                uniform_buffers,
                uniform_buffers_allocations,

                command_pool,
                command_buffers,

                present_complete_semaphores,
                render_finished_semaphores,

                in_flight_fences,
                frame_index: 0,
                swapchain_ok: false,
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

    fn pick_physical_device(instance: &ash::Instance) -> (vk::PhysicalDevice, usize) {
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
                .max_by_key(|device| device_score(instance, *device))
                .unwrap();

            // this is needed for proper data copy to GPU allocated memoty
            let min_memory_map_alignment = instance
                .get_physical_device_properties(device)
                .limits
                .min_memory_map_alignment;

            (device, min_memory_map_alignment)
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
            let (graphics_index, _graphics_queue_family_property) = queue_family_properties
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
        size: PhysicalSize<u32>,
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

            let swapchain_extent = Self::choose_swapchain_extent(surface_capabilities, size);
            let min_image_count = Self::choose_swapchain_min_image_count(surface_capabilities);

            let _present_modes = surface
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
        size: PhysicalSize<u32>,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }

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

    fn create_depth_attachment(
        instance: &Instance,
        device: &Device,
        allocator: &mut Allocator,
        physical_device: vk::PhysicalDevice,
        size: PhysicalSize<u32>,
    ) -> (vk::Image, Allocation, vk::ImageView, vk::Format) {
        let depth_format_list = &[
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        let mut depth_format = vk::Format::UNDEFINED;
        for format in depth_format_list {
            let mut format_properties = vk::FormatProperties2::default();
            unsafe {
                instance.get_physical_device_format_properties2(
                    physical_device,
                    *format,
                    &mut format_properties,
                )
            };
            if format_properties
                .format_properties
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            {
                depth_format = *format;
                break;
            }
        }

        let depth_image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(depth_format)
            .extent(
                vk::Extent3D::default()
                    .width(size.width)
                    .height(size.height)
                    .depth(1),
            )
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let depth_image = unsafe { device.create_image(&depth_image_info, None).unwrap() };
        let mem_req = unsafe { device.get_image_memory_requirements(depth_image) };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "Depth image allocation",
                requirements: mem_req,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            device
                .bind_image_memory(depth_image, allocation.memory(), allocation.offset())
                .unwrap()
        }

        let depth_image_view_info = vk::ImageViewCreateInfo::default()
            .image(depth_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(depth_format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .level_count(1)
                    .layer_count(1),
            );

        let depth_image_view = unsafe {
            device
                .create_image_view(&depth_image_view_info, None)
                .unwrap()
        };

        (depth_image, allocation, depth_image_view, depth_format)
    }

    fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
        let bindings = &[vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];

        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);
        unsafe {
            device
                .create_descriptor_set_layout(&create_info, None)
                .unwrap()
        }
    }

    fn create_graphics_pipeline(
        device: &Device,
        _swapchain_extent: vk::Extent2D,
        swapchain_format: vk::SurfaceFormatKHR,
        depth_format: vk::Format,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let mut shader_code = std::fs::File::open("./shaders/triangle.spv").unwrap();
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

        let set_layouts = &[descriptor_set_layout];
        let pipeline_layout_create_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(set_layouts);
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .unwrap()
        };

        let formats = &[swapchain_format.format];
        let mut pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(formats)
            .depth_attachment_format(depth_format);

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

    fn create_vertex_buffer(
        device: &Device,
        allocator: &mut Allocator,
        model: &tobj::Model,
    ) -> (vk::Buffer, Allocation) {
        let mut vertices = vec![];
        dbg!(model.mesh.normals.len());
        dbg!(model.mesh.positions.len());
        for i in 0..model.mesh.indices.len() {
            let pos = vec3(
                model.mesh.positions[3 * i],
                model.mesh.positions[3 * i + 1],
                model.mesh.positions[3 * i + 2],
            );

            let color = model
                .mesh
                .vertex_color
                .get((3 * i)..=(3 * i + 2))
                .map(Vec3::from_slice)
                .unwrap_or(Vec3::splat(1.0));

            let normal = model
                .mesh
                .normals
                .get((3 * i)..=(3 * i + 2))
                .map(Vec3::from_slice)
                .expect("normals are empty");

            let vertex = Vertex { pos, color, normal };
            vertices.push(vertex);
        }

        let size = dbg!(size_of_val(vertices.as_slice())) as u64;
        let (vertex_buffer, mut vertex_buffer_allocation) = create_buffer(
            device,
            allocator,
            size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryLocation::CpuToGpu,
        );

        let _copy_record =
            presser::copy_from_slice_to_offset(&vertices, &mut vertex_buffer_allocation, 0)
                .unwrap();

        (vertex_buffer, vertex_buffer_allocation)
    }

    fn create_index_buffer(
        device: &Device,
        allocator: &mut Allocator,
        model: &tobj::Model,
    ) -> (vk::Buffer, Allocation) {
        // let mut next_face = 0;
        // for f in 0..mesh.face_arities.len() {
        //     let end = next_face + mesh.face_arities[f] as usize;
        //     let face_indices: Vec<_> = mesh.indices[next_face..end].iter().collect();
        //     println!("    face[{}] = {:?}", f, face_indices);
        //     next_face = end;
        // }
        let indices = model.mesh.indices.as_slice();

        let size = size_of_val(indices) as u64;
        let (index_buffer, mut index_buffer_allocation) = create_buffer(
            device,
            allocator,
            size,
            vk::BufferUsageFlags::INDEX_BUFFER,
            MemoryLocation::CpuToGpu,
        );

        dbg!(indices.len());
        let copy_record =
            presser::copy_from_slice_to_offset(indices, &mut index_buffer_allocation, 0).unwrap();
        dbg!(copy_record);

        (index_buffer, index_buffer_allocation)
    }

    fn create_uniform_buffers(
        device: &Device,
        allocator: &mut Allocator,
    ) -> (Vec<vk::Buffer>, Vec<Allocation>) {
        let size = size_of::<ViewUniform>() as vk::DeviceSize;
        (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                create_buffer(
                    device,
                    allocator,
                    size,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    MemoryLocation::CpuToGpu,
                )
            })
            .unzip()
    }

    fn create_descriptor_pool(device: &Device) -> vk::DescriptorPool {
        let descriptor_pool_sizes = &[vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(MAX_FRAMES_IN_FLIGHT)];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(MAX_FRAMES_IN_FLIGHT)
            .pool_sizes(descriptor_pool_sizes);

        unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .unwrap()
        }
    }

    fn create_descriptor_sets(
        device: &Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
        descriptor_pool: vk::DescriptorPool,
        uniform_buffers: &[vk::Buffer],
    ) -> Vec<vk::DescriptorSet> {
        let set_layouts = vec![descriptor_set_layout; MAX_FRAMES_IN_FLIGHT as usize];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let sets = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() };

        for (buffer, set) in uniform_buffers.iter().zip(sets.iter()) {
            let buffer_info = &[vk::DescriptorBufferInfo::default()
                .buffer(buffer.clone())
                .offset(0)
                .range(size_of::<ViewUniform>() as u64)];
            let write_desc_set = &[vk::WriteDescriptorSet::default()
                .dst_set(set.clone())
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(buffer_info)];

            unsafe { device.update_descriptor_sets(write_desc_set, &[]) };
        }

        sets
    }

    fn create_command_buffers(
        device: &Device,
        command_pool: vk::CommandPool,
    ) -> Vec<vk::CommandBuffer> {
        let command_buffer_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT);

        unsafe {
            device
                .allocate_command_buffers(&command_buffer_alloc_info)
                .unwrap()
        }
    }

    fn update_uniform_buffers(&mut self, time: f32) {
        let model = Mat4::from_quat(
            Quat::from_rotation_z(FRAC_PI_2 * time) * Quat::from_rotation_y(FRAC_PI_2 * time),
        );
        let view = Mat4::look_at_rh(vec3(0.0, 4.0, -4.0), Vec3::ZERO, Vec3::Y);
        let mut projection = Mat4::perspective_rh(
            FRAC_PI_4,
            self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32,
            0.0001,
            1000.0,
        );
        projection.y_axis *= -1.0;

        let view_uniform = ViewUniform {
            model,
            view,
            projection,
            light_pos: vec3(3.0, 5.0, 0.0),
        };

        presser::copy_to_offset(
            &view_uniform,
            &mut self.uniform_buffers_allocations[self.frame_index],
            0,
        )
        .unwrap();
    }

    fn record_command_buffer(&self, swapchain_image_index: usize) {
        unsafe {
            let command_buffer = self.command_buffers[self.frame_index];

            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();

            self.transition_image_layout(
                self.swapchain_images[swapchain_image_index],
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::ImageAspectFlags::COLOR,
            );

            self.transition_image_layout(
                self.depth_image,
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
                .image_view(self.swapchain_image_views[swapchain_image_index])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_value);

            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(self.depth_image_view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(clear_depth);

            let color_attachments = &[attachment_info];
            let rendering_info = vk::RenderingInfo::default()
                .render_area(
                    vk::Rect2D::default()
                        .offset(vk::Offset2D::default().x(0).y(0))
                        .extent(self.swapchain_extent),
                )
                .layer_count(1)
                .color_attachments(color_attachments)
                .depth_attachment(&depth_attachment);

            self.device
                .cmd_begin_rendering(command_buffer, &rendering_info);

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );

            self.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer], &[0]);

            self.device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer,
                0,
                vk::IndexType::UINT32,
            );

            self.device.cmd_set_viewport(
                command_buffer,
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
                command_buffer,
                0,
                &[vk::Rect2D::default()
                    .offset(vk::Offset2D::default())
                    .extent(self.swapchain_extent)],
            );

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[self.frame_index]],
                &[],
            );

            self.device
                .cmd_draw_indexed(command_buffer, self.index_count as u32, 1, 0, 0, 0);

            self.device.cmd_end_rendering(command_buffer);

            self.transition_image_layout(
                self.swapchain_images[swapchain_image_index],
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                vk::ImageAspectFlags::COLOR,
            );

            self.device.end_command_buffer(command_buffer).unwrap();
        }
    }

    fn transition_image_layout(
        &self,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_access_mask: vk::AccessFlags2,
        dst_access_mask: vk::AccessFlags2,
        src_stage_mask: vk::PipelineStageFlags2,
        dst_stage_mask: vk::PipelineStageFlags2,
        image_aspect_mask: vk::ImageAspectFlags,
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
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(image_aspect_mask)
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
                .cmd_pipeline_barrier2(self.command_buffers[self.frame_index], &dependency_info)
        };
    }

    fn create_sync_objects(
        device: &Device,
        num_swapchain_images: usize,
    ) -> (Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>) {
        unsafe {
            let (present_complete_semaphores, in_flight_fences) = (0..MAX_FRAMES_IN_FLIGHT)
                .map(|_| {
                    (
                        device
                            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                            .unwrap(),
                        device
                            .create_fence(
                                &vk::FenceCreateInfo::default()
                                    .flags(vk::FenceCreateFlags::SIGNALED),
                                None,
                            )
                            .unwrap(),
                    )
                })
                .unzip();

            let render_finished_semaphores = (0..num_swapchain_images)
                .map(|_| {
                    device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .unwrap()
                })
                .collect_vec();

            (
                present_complete_semaphores,
                render_finished_semaphores,
                in_flight_fences,
            )
        }
    }

    fn draw_frame(&mut self, elapsed_time: f32) {
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight_fences[self.frame_index]], true, u64::MAX)
                .unwrap();

            let image_index = match self.swapchain.1.acquire_next_image(
                self.swapchain.0,
                u64::MAX,
                self.present_complete_semaphores[self.frame_index],
                vk::Fence::null(),
            ) {
                Ok((image_index, _)) => image_index,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.swapchain_ok = false;
                    return;
                }
                Err(e) => {
                    panic!("failed to aquire swapchain image: {e}");
                }
            };

            self.device
                .reset_fences(&[self.in_flight_fences[self.frame_index]])
                .unwrap();

            self.update_uniform_buffers(elapsed_time);

            self.device
                .reset_command_buffer(
                    self.command_buffers[self.frame_index],
                    vk::CommandBufferResetFlags::empty(),
                )
                .unwrap();
            self.record_command_buffer(image_index as usize);

            let wait_destination_stage_mask = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
            let wait_semaphores = &[self.present_complete_semaphores[self.frame_index]];
            let wait_dst_stage_mask = &[wait_destination_stage_mask];
            let command_buffers = &[self.command_buffers[self.frame_index]];
            let signal_semaphores = &[self.render_finished_semaphores[image_index as usize]];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(wait_dst_stage_mask)
                .command_buffers(command_buffers)
                .signal_semaphores(signal_semaphores);

            self.device
                .queue_submit(
                    self.queue,
                    &[submit_info],
                    self.in_flight_fences[self.frame_index],
                )
                .unwrap();

            let wait_semaphores = &[self.render_finished_semaphores[image_index as usize]];
            let swapchains = &[self.swapchain.0];
            let image_indices = &[image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(wait_semaphores)
                .swapchains(swapchains)
                .image_indices(image_indices);

            match self.swapchain.1.queue_present(self.queue, &present_info) {
                Ok(false) => {}
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.swapchain_ok = false;
                }
                e => panic!("queue_present error: {e:?}"),
            }

            self.frame_index = (self.frame_index + 1) % MAX_FRAMES_IN_FLIGHT as usize;
        }
    }

    fn cleanup_swapchain(&self) {
        unsafe {
            for image_view in &self.swapchain_image_views {
                self.device.destroy_image_view(*image_view, None);
            }

            self.swapchain.1.destroy_swapchain(self.swapchain.0, None);
        }
    }

    fn recreate_window_dependent_objects(&mut self, size: PhysicalSize<u32>) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.cleanup_swapchain();

            let (
                swaphain,
                swapchain_device,
                swapchain_images,
                swapchain_surface_format,
                swapchain_extent,
            ) = Self::create_swapchain(
                &self.instance,
                &self.device,
                self.physical_device,
                &self.surface,
                size,
            );
            let image_views =
                Self::create_image_views(&self.device, swapchain_surface_format, &swapchain_images);

            self.swapchain = (swaphain, swapchain_device);
            self.swapchain_images = swapchain_images;
            self.swapchain_surface_format = swapchain_surface_format;
            self.swapchain_extent = swapchain_extent;
            self.swapchain_image_views = image_views;
        }
    }
}

fn create_shader_module(device: &Device, buf: &[u8]) -> vk::ShaderModule {
    let code: &[u32] = bytemuck::cast_slice(buf);
    let create_info = vk::ShaderModuleCreateInfo::default().code(code);

    unsafe { device.create_shader_module(&create_info, None).unwrap() }
}

fn create_buffer(
    device: &Device,
    allocator: &mut Allocator,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    memory_location: MemoryLocation,
) -> (vk::Buffer, Allocation) {
    unsafe {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = device.create_buffer(&buffer_info, None).unwrap();
        let mem_requirements = device.get_buffer_memory_requirements(buffer);

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "Buffer allocation",
                requirements: mem_requirements,
                location: memory_location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        device
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            .unwrap();

        (buffer, allocation)
    }
}

fn find_memory_type(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> u32 {
    unsafe {
        let mem_properties = instance.get_physical_device_memory_properties(physical_device);

        for i in 0..mem_properties.memory_type_count {
            if (type_filter & (1 << i) > 0)
                && mem_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return i;
            }
        }

        panic!("failed to find suitable memory type")
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    unsafe {
        let callback_data = *p_callback_data;
        let message_id_number = callback_data.message_id_number;

        let message_id_name = if callback_data.p_message_id_name.is_null() {
            Cow::from("")
        } else {
            CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
        };

        let message = if callback_data.p_message.is_null() {
            Cow::from("")
        } else {
            CStr::from_ptr(callback_data.p_message).to_string_lossy()
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
}
