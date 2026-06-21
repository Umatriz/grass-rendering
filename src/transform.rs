use bevy_ecs::component::Component;
use glam::{Quat, Vec3};

#[derive(Component, Default)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
}
