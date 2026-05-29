use std::{
    fs::{File, OpenOptions},
    io::Write,
};

use bevy_app::{App, AppExit};
use bevy_time::TimePlugin;
use rendering::RenderingPlugin;
use tracing::{Level, error, info, warn};
use tracing_subscriber::{EnvFilter, Layer, filter, layer::SubscriberExt, util::SubscriberInitExt};
use windowing::WindowingPlugin;

mod rendering;
mod windowing;

fn main() -> AppExit {
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open("log.txt")
        .unwrap();

    std::panic::set_hook(Box::new(|info| {
        error!("Panic occured: {info}");
    }));

    tracing_subscriber::registry()
        // .with(
        //     EnvFilter::try_from_default_env()
        //         .unwrap_or_else(|_| format!("{}=debug", env!("CARGO_CRATE_NAME")).into()),
        // )
        .with(tracing_subscriber::fmt::layer())
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(file)
                .with_ansi(false),
        )
        .with(
            filter::Targets::new()
                .with_target(env!("CARGO_CRATE_NAME"), Level::DEBUG)
                .with_target("VULKAN", Level::TRACE),
        )
        .init();

    File::create("CRASHED.txt").unwrap();

    info!("Logging is successfully initialized");
    error!("Error!");
    warn!("Warn!");

    App::new()
        .add_plugins(TimePlugin)
        .add_plugins((WindowingPlugin, RenderingPlugin))
        .run()
}
