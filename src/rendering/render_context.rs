use std::{borrow::Cow, ffi::CStr, sync::Arc};

use ash::{Device, Entry, Instance, ext, khr, vk};
use bevy_app::{Plugin, Startup};
use bevy_ecs::prelude::*;
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings, vulkan::Allocator, vulkan::AllocatorCreateDesc,
};
use itertools::Itertools;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use tracing::{error, info, trace, warn};
use winit::{dpi::PhysicalSize, event_loop::OwnedDisplayHandle, window::Window};

use crate::windowing::{AppWindows, WinitOwnedDisplayHandle};

use super::CleanUp;

pub struct RenderContextPlugin;

impl Plugin for RenderContextPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(Startup, setup_render_context)
            .add_systems(CleanUp, destroy_render_context);
    }
}

fn setup_render_context(
    mut commands: Commands,
    windows: Res<AppWindows>,
    display_handle: Res<WinitOwnedDisplayHandle>,
) {
    let render_context = RenderContext::new(windows.primary.clone(), display_handle.0.clone());
    commands.insert_resource(render_context);
    info!("Render context was successfully created");
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

    pub allocator: Allocator,

    swapchain: (vk::SwapchainKHR, khr::swapchain::Device),
    swapchain_images: Vec<vk::Image>,
    swapchain_surface_format: vk::SurfaceFormatKHR,
    pub swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,

    // depth_image: vk::Image,
    // depth_image_allocation: Allocation,
    // depth_image_view: vk::ImageView,

    // texture_image: vk::Image,
    // texture_image_allocation: Allocation,
    // texture_image_view: vk::ImageView,

    // sampler: vk::Sampler,

    // descriptor_set_layout: vk::DescriptorSetLayout,
    // descriptor_sets: Vec<vk::DescriptorSet>,
    // descriptor_pool: vk::DescriptorPool,

    // graphics_pipeline: vk::Pipeline,
    // pipeline_layout: vk::PipelineLayout,

    // vertex_buffer: vk::Buffer,
    // vertex_buffer_allocation: Allocation,
    // index_buffer: vk::Buffer,
    // index_buffer_allocation: Allocation,
    // index_count: usize,

    // uniform_buffers: Vec<vk::Buffer>,
    // uniform_buffers_allocations: Vec<Allocation>,

    // command_pool: vk::CommandPool,
    // command_buffers: Vec<vk::CommandBuffer>,

    // present_complete_semaphores: Vec<vk::Semaphore>,
    // render_finished_semaphores: Vec<vk::Semaphore>,
    // in_flight_fences: Vec<vk::Fence>,
    frame_index: usize,
    swapchain_ok: bool,
}

impl RenderContext {
    fn new(window: Arc<Window>, display_hadle: OwnedDisplayHandle) -> Self {
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

            let (debug_utils_loader, debug_callback) = setup_debug_messenger(&entry, &instance);

            let surface = create_surface(&entry, &instance, display_hadle, window.clone());
            let surface_instance = khr::surface::Instance::new(&entry, &instance);
            let surface = (surface, surface_instance);

            let (physical_device, min_memory_map_alignment) = pick_physical_device(&instance);
            let (device, queue, queue_family_index) =
                create_logical_device(&instance, physical_device, &surface);

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
            ) = create_swapchain(
                &instance,
                &device,
                physical_device,
                &surface,
                window.inner_size(),
            );

            let swapchain_image_views =
                create_image_views(&device, swapchain_surface_format, &swapchain_images);

            Self {
                entry,
                instance,

                debug_utils_loader,
                debug_callback,

                surface,

                physical_device,
                device,
                queue,
                allocator,

                swapchain: (swaphain, swapchain_device),
                swapchain_images,
                swapchain_surface_format,
                swapchain_extent,
                swapchain_image_views,

                frame_index: 0,
                swapchain_ok: false,
            }
        }
    }
}

pub fn destroy_render_context(world: &mut World) {
    let mut rc = world.remove_resource::<RenderContext>().unwrap();
    unsafe {
        rc.device.device_wait_idle().unwrap();

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

        // Drop allocator before destroying device because it holds memory
        drop(rc.allocator);

        rc.device.destroy_device(None);
        rc.instance.destroy_instance(None);
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
        unsafe { debug_utils_instance.create_debug_utils_messenger(&debug_info, None) }.unwrap();

    (debug_utils_instance, debug_messenger)
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

            let mut vulkan_1_2_features = vk::PhysicalDeviceVulkan12Features::default();
            let mut vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features::default();
            let mut extended_dynamic_state =
                vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default();

            let mut features = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut extended_dynamic_state)
                .push_next(&mut vulkan_1_2_features)
                .push_next(&mut vulkan_1_3_features);
            instance.get_physical_device_features2(physical_device, &mut features);

            let supports_required_features = features.features.sampler_anisotropy == vk::TRUE
                && extended_dynamic_state.extended_dynamic_state == vk::TRUE
                && vulkan_1_2_features.scalar_block_layout == vk::TRUE
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

        let mut vulkan_1_2_features =
            vk::PhysicalDeviceVulkan12Features::default().scalar_block_layout(true);
        let mut vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let mut extended_dynamic_state_features =
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default()
                .extended_dynamic_state(true);
        let mut physical_device_features_2 = vk::PhysicalDeviceFeatures2::default()
            .features(vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true))
            .push_next(&mut vulkan_1_2_features)
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

        let swapchain_extent = choose_swapchain_extent(surface_capabilities, size);
        let min_image_count = choose_swapchain_min_image_count(surface_capabilities);

        let _present_modes = surface
            .1
            .get_physical_device_surface_present_modes(physical_device, surface.0)
            .unwrap();

        let formats = surface
            .1
            .get_physical_device_surface_formats(physical_device, surface.0)
            .unwrap();

        let format = choose_swapchain_format(&formats);
        let present_mode = choose_swapchain_present_mode();

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
