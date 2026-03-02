mod embedding_service;
mod model_manager;

use clap::Parser;
use corvia_proto::embedding_service_server::EmbeddingServiceServer;
use corvia_proto::model_manager_server::ModelManagerServer;
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "corvia_inference=info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { port } => {
            let addr = format!("0.0.0.0:{port}").parse()?;
            let model_mgr = model_manager::ModelManagerService::new();
            let embed_svc = embedding_service::EmbeddingServiceImpl::new();

            tracing::info!("corvia-inference listening on {addr}");

            Server::builder()
                .add_service(ModelManagerServer::new(model_mgr))
                .add_service(EmbeddingServiceServer::new(embed_svc))
                .serve(addr)
                .await?;
        }
    }

    Ok(())
}
