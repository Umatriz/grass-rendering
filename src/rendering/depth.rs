use ash::vk;
use bevy_app::{Plugin, Startup};
use bevy_ecs::{
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Res, ResMut},
};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{AllocationCreateDesc, AllocationScheme},
};

use crate::windowing::AppWindows;

use super::{
    asset::{Handle, RenderAssets, common::SimpleImage},
    render_context::{RenderContext, create_render_context},
};

pub struct DepthAttachmentPlugin;

impl Plugin for DepthAttachmentPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(
            Startup,
            // TODO: use SystemSets
            create_depth_attachment.after(create_render_context),
        );
    }
}

#[derive(Resource)]
pub struct DepthAttachment {
    pub image: Handle<SimpleImage>,
    pub format: vk::Format,
}

fn create_depth_attachment(
    mut commands: Commands,
    mut rc: ResMut<RenderContext>,
    mut simple_image_assets: ResMut<RenderAssets<SimpleImage>>,
    windows: Res<AppWindows>,
) {
    let depth_format_list = &[
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];
    let mut depth_format = vk::Format::UNDEFINED;
    for format in depth_format_list {
        let mut format_properties = vk::FormatProperties2::default();
        unsafe {
            rc.instance.get_physical_device_format_properties2(
                rc.physical_device,
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

    let size = windows.primary.inner_size();
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

    let depth_image = unsafe { rc.device.create_image(&depth_image_info, None).unwrap() };
    let mem_req = unsafe { rc.device.get_image_memory_requirements(depth_image) };

    let allocation = rc
        .allocator
        .allocate(&AllocationCreateDesc {
            name: "Depth image allocation",
            requirements: mem_req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .unwrap();

    unsafe {
        rc.device
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
        rc.device
            .create_image_view(&depth_image_view_info, None)
            .unwrap()
    };

    let image = SimpleImage {
        image: depth_image,
        view: depth_image_view,
        allocation,
    };
    let image = simple_image_assets.add(image);

    commands.insert_resource(DepthAttachment {
        image,
        format: depth_format,
    });
}
