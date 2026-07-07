use ash::{Device, vk};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};
use itertools::Itertools;

use crate::rendering::{MAX_FRAMES_IN_FLIGHT, render_context::RenderContext};

use super::RenderAsset;

pub struct SimpleImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: Allocation,
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
pub struct UniformBuffers(Vec<Buffer>);

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
            .collect_vec();
        Self(buffers)
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

pub struct Sampler(vk::Sampler);

impl RenderAsset for Sampler {
    fn destroy(self, world: &mut bevy_ecs::world::World) -> anyhow::Result<()> {
        let rc = world.resource_mut::<RenderContext>();
        unsafe {
            rc.device.destroy_sampler(self.0, None);
        }

        Ok(())
    }
}
