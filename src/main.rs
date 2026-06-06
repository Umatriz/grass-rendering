use std::{
    fs::{File, OpenOptions},
    io,
    process::Command,
};

use bevy_app::{App, AppExit, PreStartup};
use bevy_time::TimePlugin;
use rendering::{RenderingPlugin, camera::CameraPlugin};
use tracing::{Level, error, info, warn};
use tracing_subscriber::{filter, layer::SubscriberExt, util::SubscriberInitExt};
use windowing::WindowingPlugin;

mod rendering;
mod transform;
mod windowing;

fn main() -> AppExit {
    let file = OpenOptions::new()
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

    info!("Logging is successfully initialized");
    error!("Error!");
    warn!("Warn!");

    App::new()
        .add_plugins(TimePlugin)
        .add_plugins((WindowingPlugin, RenderingPlugin, CameraPlugin))
        .add_systems(PreStartup, compile_shaders)
        .run()
}

fn compile_shaders() {
    compile_slang("shaders/triangle.slang", &["vertMain", "fragMain"]).unwrap();
}

fn compile_slang(input: &str, entries: &[&str]) -> io::Result<()> {
    let mut output = input.strip_suffix(".slang").map(|s| s.to_string()).unwrap();
    output.push_str(".spv");

    info!("Compiling {input}");
    Command::new("slangc")
        .arg(input)
        .args([
            "-target",
            "spirv",
            "-profile",
            "spirv_1_4",
            "-emit-spirv-directly",
            "-fvk-use-entrypoint-name",
        ])
        .args(entries.iter().flat_map(|entry| ["-entry", entry]))
        .arg("-o")
        .arg(output)
        .stderr(io::stderr())
        .stdout(io::stdout())
        .status()?;

    Ok(())
}
