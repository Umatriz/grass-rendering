use std::collections::BTreeMap;

use ash::vk;
use glam::{Vec2, Vec3};
use itertools::Itertools;

pub struct Mesh {
    indices: Vec<u32>,
    attributes: BTreeMap<MeshVertexAttributeKey, MeshVertexAttribute>,
}

impl Mesh {
    pub fn get_binding_description(&self) -> vk::VertexInputBindingDescription {
        // WARNING: tightly packed data. Make sure the alignment is correct on the GPU
        let size = self
            .attributes
            .values()
            .map(MeshVertexAttribute::size)
            .sum::<usize>();

        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn get_attribute_descriptions(&self) -> Vec<vk::VertexInputAttributeDescription> {
        self.attributes
            .iter()
            .scan(0, |offset, (key, attr)| {
                let desc = vk::VertexInputAttributeDescription::default()
                    .location(*key as u32)
                    .binding(0)
                    .format(attr.format())
                    .offset(*offset);

                *offset += attr.size() as u32;
                Some(desc)
            })
            .collect_vec()
    }
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum MeshVertexAttributeKey {
    Position = 0,
    Normal = 1,
    Uv = 2,
    Color = 3,
}

#[derive(Clone)]
pub enum MeshVertexAttribute {
    Float3x32(Vec<Vec3>),
    Float2x32(Vec<Vec2>),
}

impl MeshVertexAttribute {
    pub fn size(&self) -> usize {
        match self {
            MeshVertexAttribute::Float3x32(..) => size_of::<Vec3>(),
            MeshVertexAttribute::Float2x32(..) => size_of::<Vec2>(),
        }
    }

    pub fn format(&self) -> vk::Format {
        match self {
            MeshVertexAttribute::Float3x32(..) => vk::Format::R32G32B32_SFLOAT,
            MeshVertexAttribute::Float2x32(..) => vk::Format::R32G32_SFLOAT,
        }
    }
}
