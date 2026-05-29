use std::{borrow::Cow, collections::HashMap, sync::Arc};

use bevy_app::{App, AppExit, Plugin, PluginsState};
use bevy_ecs::{event::Event, message::Message, resource::Resource};
use tracing::error;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop, OwnedDisplayHandle},
    window::{Window, WindowAttributes, WindowId},
};

use crate::rendering::{CleanUp, RenderContext};

pub struct WindowingPlugin;

impl Plugin for WindowingPlugin {
    fn build(&self, app: &mut App) {
        let event_loop = EventLoop::new().unwrap();

        event_loop.set_control_flow(ControlFlow::Poll);

        app.set_runner(|app| runner(app, event_loop));
    }
}

fn runner(mut app: App, event_loop: EventLoop<()>) -> AppExit {
    if app.plugins_state() == PluginsState::Ready {
        app.finish();
        app.cleanup();
    }

    app.add_message::<RawWinitWindowEvent>();

    app.world_mut()
        .insert_resource(WinitOwnedDisplayHandle(event_loop.owned_display_handle()));

    let mut runner_state = WinitAppRunnerState::new(app);

    if let Err(err) = event_loop.run_app(&mut runner_state) {
        error!("winit event loop returned an error: {err}");
    };

    // unsafe {
    //     runner_state
    //         .app
    //         .world_mut()
    //         .resource::<RenderContext>()
    //         .device
    //         .device_wait_idle()
    //         .unwrap()
    // };

    runner_state.app.world_mut().run_schedule(CleanUp);

    runner_state.app.world_mut().clear_all();

    runner_state.app_exit.unwrap_or_else(|| {
        error!("Failed to receive an app exit code! This is a bug");
        AppExit::error()
    })
}

#[derive(Resource)]
pub struct AppWindows {
    pub primary: Arc<Window>,

    /// Secondary windows that can be accessed by a string ID.
    pub secondary: HashMap<Cow<'static, str>, Arc<Window>>,
}

#[derive(Message)]
pub struct RawWinitWindowEvent {
    pub event: WindowEvent,
    pub window_id: WindowId,
}

#[derive(Resource)]
pub struct WinitOwnedDisplayHandle(pub OwnedDisplayHandle);

struct WinitAppRunnerState {
    app: App,
    app_exit: Option<AppExit>,
    // system_state: SystemState<(ResMut<'static, AppWindows>)>,
}

impl WinitAppRunnerState {
    fn new(app: App) -> Self {
        // let system_state = SystemState::new(app.world_mut());

        Self {
            app,
            app_exit: None,
            // system_state,
        }
    }
}

impl ApplicationHandler for WinitAppRunnerState {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let primary_window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_resizable(true)
                    .with_inner_size(LogicalSize::new(1280, 720)),
            )
            .unwrap();

        self.app.world_mut().insert_resource(AppWindows {
            primary: Arc::new(primary_window),
            secondary: HashMap::new(),
        });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        // let system_state = &mut self.system_state;

        match event {
            WindowEvent::CloseRequested => {
                self.app_exit = Some(AppExit::Success);
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.app.update();
                self.app
                    .world()
                    .resource::<AppWindows>()
                    .primary
                    .request_redraw();
            }
            event => {
                self.app
                    .world_mut()
                    .write_message(RawWinitWindowEvent { event, window_id });
            }
        }
    }

    fn exiting(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {}
}
