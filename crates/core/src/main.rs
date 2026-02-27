use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tracing::info;

mod api;
mod config;
mod memory_engine;
mod channels;
mod types;

#[derive(Parser)]
#[command(name = "hiveminddb", about = "Distributed AI agent memory system")]
struct Cli {
    /// Listen address for agent connections (REST + WebSocket)
    #[arg(long, default_value = "0.0.0.0:8100", env = "HIVEMIND_LISTEN_ADDR")]
    listen_addr: String,

    /// RaftTimeDB WebSocket URL (the SpacetimeDB proxy)
    #[arg(long, default_value = "ws://127.0.0.1:3001", env = "HIVEMIND_RTDB_URL")]
    rtdb_url: String,

    /// LLM provider for extraction pipeline (anthropic, openai, ollama)
    #[arg(long, default_value = "anthropic", env = "HIVEMIND_LLM_PROVIDER")]
    llm_provider: String,

    /// LLM API key (for extraction pipeline)
    #[arg(long, env = "HIVEMIND_LLM_API_KEY")]
    llm_api_key: Option<String>,

    /// LLM model name for extraction
    #[arg(long, default_value = "claude-sonnet-4-20250514", env = "HIVEMIND_LLM_MODEL")]
    llm_model: String,

    /// Embedding model (openai:text-embedding-3-small, ollama:nomic-embed-text, etc.)
    #[arg(long, default_value = "openai:text-embedding-3-small", env = "HIVEMIND_EMBEDDING_MODEL")]
    embedding_model: String,

    /// Embedding API key (if different from LLM key)
    #[arg(long, env = "HIVEMIND_EMBEDDING_API_KEY")]
    embedding_api_key: Option<String>,

    /// Data directory for local state
    #[arg(long, default_value = "./data", env = "HIVEMIND_DATA_DIR")]
    data_dir: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hiveminddb=info".into()),
        )
        .init();

    let cli = Cli::parse();

    info!(
        listen_addr = %cli.listen_addr,
        rtdb_url = %cli.rtdb_url,
        llm_provider = %cli.llm_provider,
        "Starting HiveMindDB"
    );

    let config = config::HiveMindConfig {
        listen_addr: cli.listen_addr.clone(),
        rtdb_url: cli.rtdb_url,
        llm_provider: cli.llm_provider,
        llm_api_key: cli.llm_api_key,
        llm_model: cli.llm_model,
        embedding_model: cli.embedding_model,
        embedding_api_key: cli.embedding_api_key,
        data_dir: cli.data_dir,
    };

    let engine = Arc::new(memory_engine::MemoryEngine::new(config.clone()));
    let channel_hub = Arc::new(channels::ChannelHub::new());

    // Build and start the API server
    let app = api::router(engine.clone(), channel_hub.clone());

    let listener = tokio::net::TcpListener::bind(&cli.listen_addr).await?;
    info!(addr = %cli.listen_addr, "HiveMindDB API listening");

    axum::serve(listener, app).await?;

    Ok(())
}
