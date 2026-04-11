use bevy_app::{App, AppExit};
use rendering::RenderingPlugin;
use tracing::{error, info, warn};
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};
use windowing::WindowingPlugin;

mod rendering;
mod windowing;

fn main() -> AppExit {
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("{}=debug", env!("CARGO_CRATE_NAME")).into()),
        )
        .with(tracing_subscriber::fmt::layer().pretty())
        .init();

    info!("Logging is successfully initialized");
    error!("Error!");
    warn!("Warn!");

    App::new()
        .add_plugins((WindowingPlugin, RenderingPlugin))
        .run()
}
