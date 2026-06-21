use std::f32::consts::FRAC_PI_4;

use bevy_app::{Plugin, PostUpdate, Startup, Update};
use bevy_ecs::prelude::*;
use bevy_time::Time;
use dolly::{
    prelude::{Arm, LookAt, Position, Smooth, YawPitch},
    rig::CameraRig,
};
use glam::{Mat4, Quat, Vec3};
use winit::{
    event::{KeyEvent, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
};

use crate::{
    transform::Transform,
    windowing::{AppWindows, RawWinitWindowEvent},
};

use super::{RenderContext, render, resize};

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(Startup, spawn_camera);
        app.add_systems(Update, move_camera);
        app.add_systems(PostUpdate, update_camera.after(resize).before(render));
    }
}

#[derive(Component)]
pub struct Camera {
    pub projection: Mat4,
    camera_rig: CameraRig,
}

fn projection(width: u32, height: u32) -> Mat4 {
    let mut projection =
        Mat4::perspective_rh(FRAC_PI_4, width as f32 / height as f32, 0.0001, 1000.0);
    projection.y_axis *= -1.0;

    projection
}

fn spawn_camera(mut commands: Commands, window: Res<AppWindows>) {
    let camera_rig: CameraRig = CameraRig::builder()
        .with(YawPitch::new().yaw_degrees(45.0).pitch_degrees(-30.0))
        .with(Smooth::new_rotation(1.5))
        .with(Arm::new(Vec3::Z * 4.0))
        .build();

    // let camera_rig = CameraRig::builder()
    //     .with(Position::new(Vec3::Y))
    //     .with(YawPitch::new())
    //     .with(Smooth::new_position_rotation(1.0, 1.0))
    //     .build();

    let size = window.primary.inner_size();
    let projection = projection(size.width, size.height);

    commands.spawn((
        Camera {
            projection,
            camera_rig,
        },
        Transform::default(),
    ));
}

fn key_pressed(key_event: &KeyEvent, key_code: KeyCode) -> bool {
    if let PhysicalKey::Code(code) = key_event.physical_key {
        key_code == code
    } else {
        false
    }
}

fn move_camera(
    mut camera: Single<&mut Camera>,
    mut winit_events: MessageReader<RawWinitWindowEvent>,
    time: Res<Time>,
) {
    for event in winit_events.read() {
        let WindowEvent::KeyboardInput {
            device_id,
            event,
            is_synthetic,
        } = &event.event
        else {
            continue;
        };

        let camera_driver = camera.camera_rig.driver_mut::<YawPitch>();

        if key_pressed(event, KeyCode::KeyW) {
            camera_driver.rotate_yaw_pitch(0.0, -30.0);
        }
        if key_pressed(event, KeyCode::KeyS) {
            camera_driver.rotate_yaw_pitch(0.0, 30.0);
        }
        if key_pressed(event, KeyCode::KeyA) {
            camera_driver.rotate_yaw_pitch(-30.0, 0.0);
        }
        if key_pressed(event, KeyCode::KeyD) {
            camera_driver.rotate_yaw_pitch(30.0, 0.0);
        }

        // let mut velocity = Vec3::ZERO;
        // if key_pressed(event, KeyCode::KeyW) {
        //     velocity += Vec3::NEG_Z;
        // }
        // if key_pressed(event, KeyCode::KeyS) {
        //     velocity += Vec3::Z;
        // }
        // if key_pressed(event, KeyCode::KeyA) {
        //     velocity += Vec3::NEG_X;
        // }
        // if key_pressed(event, KeyCode::KeyD) {
        //     velocity += Vec3::X;
        // }
        // if key_pressed(event, KeyCode::KeyE) {
        //     velocity += Vec3::Y;
        // }
        // if key_pressed(event, KeyCode::KeyQ) {
        //     velocity += Vec3::NEG_Y;
        // }

        // let move_vec = Quat::from(camera.camera_rig.final_transform.rotation)
        //     * velocity.clamp_length_max(1.0)
        //     * 10.0f32.powi(key_pressed(event, KeyCode::ShiftLeft) as i32);

        let delta_secs = time.delta_secs();

        // camera
        //     .camera_rig
        //     .driver_mut::<Position>()
        //     .translate(move_vec * delta_secs * 10.0);

        camera.camera_rig.update(delta_secs);
    }
}

fn update_camera(
    camera: Single<(&mut Camera, &mut Transform)>,
    render_ctx: Res<RenderContext>,
    time: Res<Time>,
) {
    let (mut camera, mut camera_transform) = camera.into_inner();

    camera.projection = projection(
        render_ctx.swapchain_extent.width,
        render_ctx.swapchain_extent.height,
    );

    let (position, rotation) = camera
        .camera_rig
        .update(time.delta_secs())
        .into_position_rotation();
    camera_transform.position = position;
    camera_transform.rotation = rotation;
}
