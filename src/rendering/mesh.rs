use std::collections::BTreeMap;

pub struct Mesh {
    indices: Vec<u32>,
    attributes: BTreeMap<MeshVertexAttributeKey, MeshVertexAttribute>,
}

pub enum MeshVertexAttributeKey {
    Position,
    Normal,
    Uv,
    Color,
}

pub struct MeshVertexAttribute {}
