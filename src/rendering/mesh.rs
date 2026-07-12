use std::{collections::BTreeMap, mem};

use ash::vk;
use bevy_app::Plugin;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    lifecycle::Insert,
    observer::On,
    query::Changed,
    system::{Commands, Query, Res, ResMut},
};
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use gltf::{buffer, mesh::util::ReadIndices};
use gpu_allocator::MemoryLocation;
use itertools::{Itertools, multizip};

use super::{
    asset::{Handle, RenderAssets, common::Buffer},
    render_context::RenderContext,
};

pub struct MeshPlugin;

impl Plugin for MeshPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_observer(generate_mesh_render_data);
    }
}

#[derive(Component)]
pub struct Mesh {
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
}

impl Mesh {
    pub fn from_gltf(buffers: &[buffer::Data], mesh: gltf::Mesh) -> Self {
        let mut indices = Vec::new();
        let mut vertices = Vec::new();

        for primitive in mesh.primitives() {
            let reader = primitive.reader(|b| Some(&buffers[b.index()]));

            indices = reader
                .read_indices()
                .map(|iter| iter.into_u32().collect_vec())
                .unwrap_or_default();

            let positions = reader
                .read_positions()
                .map(|iter| iter.map(Vec3::from_array).collect_vec())
                .unwrap_or_default();

            let normals = reader
                .read_normals()
                .map(|iter| iter.map(Vec3::from_array).collect_vec())
                .unwrap_or_default();

            let uvs = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().map(Vec2::from_array).collect_vec())
                .unwrap_or_default();

            assert_eq!(positions.len(), normals.len());
            assert_eq!(positions.len(), uvs.len());

            for (p, n, uv) in multizip((positions, normals, uvs)) {
                vertices.push(Vertex {
                    pos: p,
                    color: Vec3::ONE,
                    normal: n,
                    uv,
                });
            }
        }

        Self { indices, vertices }
    }
}

#[derive(Pod, Zeroable, Clone, Copy, Debug)]
#[repr(C)]
pub struct Vertex {
    pos: Vec3,
    color: Vec3,
    normal: Vec3,
    uv: Vec2,
}

impl Vertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 4] {
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
            vk::VertexInputAttributeDescription::default()
                .location(3)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(mem::offset_of!(Vertex, uv) as u32),
        ]
    }
}

#[derive(Component)]
pub struct MeshRenderData {
    pub index_buffer: Handle<Buffer>,
    pub vertex_buffer: Handle<Buffer>,
}

fn generate_mesh_render_data(
    on: On<Insert, Mesh>,
    mut commands: Commands,
    meshes: Query<&Mesh>,
    mut rc: ResMut<RenderContext>,
    mut buffers: ResMut<RenderAssets<Buffer>>,
) {
    let mesh = meshes.get(on.entity).unwrap();

    let indices = mesh.indices.as_slice();
    let mut index_buffer = Buffer::new(
        &rc.device.clone(),
        &mut rc.allocator,
        size_of_val(indices) as vk::DeviceSize,
        vk::BufferUsageFlags::INDEX_BUFFER,
        MemoryLocation::CpuToGpu,
    );
    index_buffer.write(indices).unwrap();

    let vertices = mesh.vertices.as_slice();
    let mut vertex_buffer = Buffer::new(
        &rc.device.clone(),
        &mut rc.allocator,
        size_of_val(vertices) as vk::DeviceSize,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        MemoryLocation::CpuToGpu,
    );
    vertex_buffer.write(vertices).unwrap();

    let mesh_render_data = MeshRenderData {
        index_buffer: buffers.add(index_buffer),
        vertex_buffer: buffers.add(vertex_buffer),
    };
}
