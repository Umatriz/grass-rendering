use ash::vk;
use bevy_app::Plugin;
use bevy_ecs::{prelude::*, system::SystemParam};
use itertools::Itertools;

use crate::dense_storage::{DenseStorage, Index};

use super::{MAX_FRAMES_IN_FLIGHT, render_context::RenderContext};

// pub struct DescriptorManagementPlugin;

// impl Plugin for DescriptorManagementPlugin {
//     fn build(&self, app: &mut bevy_app::App) {}
// }

pub struct DescriptorSetBinding {
    pub binding: u32,
    pub descriptor_kind: DescriptorKind,
    pub descriptor_count: u32,
    pub stage_flags: vk::ShaderStageFlags,
}

pub enum DescriptorKind {
    CombinedSampler {
        sampler: vk::Sampler,
        image_view: vk::ImageView,
        image_layout: vk::ImageLayout,
    },
    UniformBuffer {
        // TODO: Rethink this
        buffers: [vk::Buffer; MAX_FRAMES_IN_FLIGHT],
        offset: u64,
        range: u64,
    },
}

impl DescriptorKind {
    pub fn as_type(&self) -> vk::DescriptorType {
        match self {
            DescriptorKind::CombinedSampler { .. } => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            DescriptorKind::UniformBuffer { .. } => vk::DescriptorType::UNIFORM_BUFFER,
        }
    }
}

pub struct DescriptorCache {
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl DescriptorCache {
    pub fn new(device: &ash::Device, bindings: &[DescriptorSetBinding]) -> anyhow::Result<Self> {
        // Layout
        let vk_bindings = bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(b.descriptor_kind.as_type())
                    .descriptor_count(b.descriptor_count)
                    .stage_flags(b.stage_flags)
            })
            .collect_vec();
        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_bindings);
        let layout = unsafe { device.create_descriptor_set_layout(&create_info, None)? };

        // Pool
        let pool_sizes = bindings
            .iter()
            .map(|b| {
                vk::DescriptorPoolSize::default()
                    .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
                    .ty(b.descriptor_kind.as_type())
            })
            .collect_vec();
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(MAX_FRAMES_IN_FLIGHT as u32)
            .pool_sizes(&pool_sizes);
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        // Allocate
        let set_layouts = vec![layout; MAX_FRAMES_IN_FLIGHT as usize];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&set_layouts);
        let sets = unsafe { device.allocate_descriptor_sets(&allocate_info)? };

        for (idx, set) in sets.iter().enumerate() {
            let mut infos = vec![(Vec::new(), Vec::new()); bindings.len()];
            let writes = bindings
                .iter()
                .zip(infos.iter_mut())
                .map(|(b, (image_info, buffer_info))| {
                    let write_info = vk::WriteDescriptorSet::default()
                        .dst_set(set.clone())
                        .dst_binding(b.binding)
                        .dst_array_element(0)
                        .descriptor_count(b.descriptor_count)
                        .descriptor_type(b.descriptor_kind.as_type());

                    match b.descriptor_kind {
                        DescriptorKind::CombinedSampler {
                            sampler,
                            image_view,
                            image_layout,
                        } => {
                            image_info.push(
                                vk::DescriptorImageInfo::default()
                                    .sampler(sampler)
                                    .image_view(image_view)
                                    .image_layout(image_layout),
                            );

                            write_info.image_info(image_info)
                        }
                        DescriptorKind::UniformBuffer {
                            buffers,
                            offset,
                            range,
                        } => {
                            buffer_info.push(
                                vk::DescriptorBufferInfo::default()
                                    .buffer(buffers[idx])
                                    .offset(offset)
                                    .range(range),
                            );

                            write_info.buffer_info(buffer_info)
                        }
                    }
                })
                .collect_vec();

            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        Ok(Self {
            pool,
            layout,
            descriptor_sets: sets,
        })
    }

    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    pub fn descriptor_sets(&self) -> &[vk::DescriptorSet] {
        &self.descriptor_sets
    }

    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            device.free_descriptor_sets(self.pool, self.descriptor_sets());
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
