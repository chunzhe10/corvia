// tonic::Status is >=176 bytes (required by gRPC trait signatures); boxing is impractical.
#![allow(clippy::result_large_err)]

mod backend;
mod chat_service;
mod embedding_service;
mod model_manager;

use clap::Parser;
use corvia_common::config::TelemetryConfig;
use corvia_proto::chat_service_server::ChatServiceServer;
use corvia_proto::embedding_service_server::EmbeddingServiceServer;
use corvia_proto::model_manager_server::ModelManagerServer;
use corvia_telemetry::propagation::MetadataExtractor;
use opentelemetry::global;
use tonic::transport::Server;

#[derive(Parser)]
#[command(name = "corvia-inference")]
#[command(about = "Corvia inference server — gRPC embedding + chat")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Start the gRPC server
    Serve {
        #[arg(long, default_value = "8030")]
        port: u16,
    },
}

/// Tonic interceptor that extracts W3C trace context from incoming gRPC metadata.
fn accept_trace(
    request: tonic::Request<()>,
) -> std::result::Result<tonic::Request<()>, tonic::Status> {
    let parent_cx = global::get_text_map_propagator(|propagator| {
        propagator.extract(&MetadataExtractor(request.metadata()))
    });
    // Attach the parent context so spans created in this request inherit it.
    // The guard is intentionally dropped — tracing-opentelemetry captures the
    // current OTel context at span creation time, so child spans created
    // immediately after will pick up the parent.
    parent_cx.attach();
    Ok(request)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let telemetry_config = TelemetryConfig {
        service_name: "corvia-inference".into(),
        ..Default::default()
    };
    let _telemetry_guard = corvia_telemetry::init_telemetry(&telemetry_config)?;

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { port } => {
            let addr = format!("0.0.0.0:{port}").parse()?;

            // Probe GPU capabilities once at startup (wrapped for runtime re-probing)
            let gpu = backend::GpuCapabilities::probe();
            tracing::info!(
                cuda = gpu.cuda_available,
                openvino = gpu.openvino_available,
                "GPU capabilities"
            );
            let gpu = std::sync::Arc::new(std::sync::RwLock::new(gpu));

            let embed_svc = embedding_service::EmbeddingServiceImpl::new();
            let chat_svc = chat_service::ChatServiceImpl::new();
            let model_mgr = model_manager::ModelManagerService::new(
                embed_svc.clone(),
                chat_svc.clone(),
                gpu,
            );

            tracing::info!(port, "inference_server_starting");

            Server::builder()
                .add_service(ModelManagerServer::new(model_mgr))
                .add_service(EmbeddingServiceServer::with_interceptor(embed_svc, accept_trace))
                .add_service(ChatServiceServer::with_interceptor(chat_svc, accept_trace))
                .serve(addr)
                .await?;
        }
    }

    Ok(())
}
