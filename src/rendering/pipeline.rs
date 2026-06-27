use ash::vk;
use bevy_ecs::world::World;

use super::descriptor_management::DescriptorChache;

pub trait GraphicsPipeline {
    fn create(world: &mut World);
    fn destroy(self, world: &mut World);
}

pub struct MeshPipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_cache: DescriptorChache,
}
