use ash::vk;
use gpu_allocator::vulkan::Allocation;

use crate::rendering::render_context::RenderContext;

use super::RenderAsset;

pub struct SimpleImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: Allocation,
}

impl RenderAsset for SimpleImage {
    fn destroy(self, world: &mut bevy_ecs::world::World) -> crate::Result<()> {
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

impl RenderAsset for Buffer {
    fn destroy(self, world: &mut bevy_ecs::world::World) -> crate::Result<()> {
        let mut rc = world.resource_mut::<RenderContext>();
        unsafe {
            rc.allocator.free(self.allocation);
            rc.device.destroy_buffer(self.buffer, None);
        }

        Ok(())
    }
}

pub struct Sampler(vk::Sampler);

impl RenderAsset for Sampler {
    fn destroy(self, world: &mut bevy_ecs::world::World) -> crate::Result<()> {
        let rc = world.resource_mut::<RenderContext>();
        unsafe {
            rc.device.destroy_sampler(self.0, None);
        }

        Ok(())
    }
}
