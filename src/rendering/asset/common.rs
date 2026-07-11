use ash::{Device, Instance, vk};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};
use itertools::Itertools;

use crate::rendering::{
    MAX_FRAMES_IN_FLIGHT,
    render_context::RenderContext,
    utils::{copy_buffer_to_image, create_image, single_time_commands, transition_image_layout},
};

use super::RenderAsset;

pub struct SimpleImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: Allocation,
}

pub struct SimpleImageCreateInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub memory_location: MemoryLocation,
    pub data: &'a [u8],
}

impl SimpleImage {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        command_pool: &vk::CommandPool,
        queue: &vk::Queue,
        create_info: SimpleImageCreateInfo<'_>,
    ) -> Self {
        let SimpleImageCreateInfo {
            width,
            height,
            format,
            tiling,
            usage,
            memory_location,
            data,
        } = create_info;

        let (image, image_allocation) = create_image(
            device,
            allocator,
            width,
            height,
            format,
            tiling,
            usage,
            memory_location,
        );

        let mut staging_buffer = Buffer::new(
            device,
            allocator,
            size_of_val(data) as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );

        presser::copy_from_slice_to_offset(data, &mut staging_buffer.allocation, 0).unwrap();

        single_time_commands(device, command_pool, queue, |device, command_buffer| {
            transition_image_layout(
                device,
                command_buffer,
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::PipelineStageFlags2::TRANSFER,
                vk::ImageAspectFlags::COLOR,
            );

            copy_buffer_to_image(
                device,
                command_buffer,
                staging_buffer.buffer,
                image,
                width,
                height,
            );

            transition_image_layout(
                device,
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::AccessFlags2::SHADER_READ,
                vk::PipelineStageFlags2::TRANSFER,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::ImageAspectFlags::COLOR,
            );
        });
        unsafe { device.queue_wait_idle(*queue).unwrap() };

        allocator.free(staging_buffer.allocation).unwrap();
        unsafe { device.destroy_buffer(staging_buffer.buffer, None) };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .base_array_layer(0)
                    .level_count(1)
                    .layer_count(1),
            );
        let image_view = unsafe { device.create_image_view(&view_info, None).unwrap() };

        Self {
            image,
            view: image_view,
            allocation: image_allocation,
        }
    }
}

impl RenderAsset for SimpleImage {
    fn destroy(self, world: &mut bevy_ecs::world::World) -> anyhow::Result<()> {
        let mut rc = world.resource_mut::<RenderContext>();
        unsafe {
            rc.allocator.free(self.allocation)?;
            rc.device.destroy_image_view(self.view, None);
            rc.device.destroy_image(self.image, None);
        }

        Ok(())
    }
}

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
}

impl Buffer {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
    ) -> Self {
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

            Self { buffer, allocation }
        }
    }
}

impl RenderAsset for Buffer {
    fn destroy(self, world: &mut bevy_ecs::world::World) -> anyhow::Result<()> {
        let mut rc = world.resource_mut::<RenderContext>();
        unsafe {
            rc.allocator.free(self.allocation)?;
            rc.device.destroy_buffer(self.buffer, None);
        }

        Ok(())
    }
}

// TODO: keep type information
pub struct UniformBuffers([Buffer; MAX_FRAMES_IN_FLIGHT]);

impl UniformBuffers {
    pub fn new<T>(device: &Device, allocator: &mut Allocator) -> Self {
        let size = size_of::<T>() as vk::DeviceSize;
        let buffers = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                Buffer::new(
                    device,
                    allocator,
                    size,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    MemoryLocation::CpuToGpu,
                )
            })
            .collect_array()
            .unwrap();
        Self(buffers)
    }

    pub fn raw_buffers(&self) -> [vk::Buffer; MAX_FRAMES_IN_FLIGHT] {
        self.0.iter().map(|b| b.buffer).collect_array().unwrap()
    }
}

impl RenderAsset for UniformBuffers {
    fn destroy(self, world: &mut bevy_ecs::world::World) -> anyhow::Result<()> {
        for buffer in self.0 {
            buffer.destroy(world)?;
        }

        Ok(())
    }
}

pub struct Sampler(pub vk::Sampler);

impl Sampler {
    pub fn basic(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        device: &Device,
    ) -> Self {
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(properties.limits.max_sampler_anisotropy)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);

        let sampler = unsafe { device.create_sampler(&sampler_info, None).unwrap() };
        Sampler(sampler)
    }
}

impl RenderAsset for Sampler {
    fn destroy(self, world: &mut bevy_ecs::world::World) -> anyhow::Result<()> {
        let rc = world.resource_mut::<RenderContext>();
        unsafe {
            rc.device.destroy_sampler(self.0, None);
        }

        Ok(())
    }
}
