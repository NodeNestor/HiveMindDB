use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tokio::sync::watch;
use tracing::info;

mod api;
mod channels;
mod config;
mod embeddings;
mod extraction;
mod memory_engine;
mod persistence;
mod types;
mod websocket;

#[derive(Parser)]
#[command(name = "hiveminddb", about = "Distributed AI agent memory system")]
struct Cli {
    /// Listen address for agent connections (REST + WebSocket)
    #[arg(long, default_value = "0.0.0.0:8100", env = "HIVEMIND_LISTEN_ADDR")]
    listen_addr: String,

    /// RaftTimeDB WebSocket URL (the SpacetimeDB proxy)
    #[arg(long, default_value = "ws://127.0.0.1:3001", env = "HIVEMIND_RTDB_URL")]
    rtdb_url: String,

    /// LLM provider for extraction pipeline (openai, anthropic, ollama, codegate, or a URL)
    #[arg(long, default_value = "anthropic", env = "HIVEMIND_LLM_PROVIDER")]
    llm_provider: String,

    /// LLM API key (for extraction pipeline)
    #[arg(long, env = "HIVEMIND_LLM_API_KEY")]
    llm_api_key: Option<String>,

    /// LLM model name for extraction
    #[arg(long, default_value = "claude-sonnet-4-20250514", env = "HIVEMIND_LLM_MODEL")]
    llm_model: String,

    /// Embedding model (provider:model, e.g. openai:text-embedding-3-small, ollama:nomic-embed-text)
    #[arg(long, default_value = "openai:text-embedding-3-small", env = "HIVEMIND_EMBEDDING_MODEL")]
    embedding_model: String,

    /// Embedding API key (if different from LLM key)
    #[arg(long, env = "HIVEMIND_EMBEDDING_API_KEY")]
    embedding_api_key: Option<String>,

    /// Data directory for snapshots and local state
    #[arg(long, default_value = "./data", env = "HIVEMIND_DATA_DIR")]
    data_dir: String,

    /// Snapshot interval in seconds (0 to disable)
    #[arg(long, default_value = "60", env = "HIVEMIND_SNAPSHOT_INTERVAL")]
    snapshot_interval: u64,

    /// Enable RaftTimeDB replication
    #[arg(long, env = "HIVEMIND_ENABLE_REPLICATION")]
    enable_replication: bool,
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
        snapshot_interval = cli.snapshot_interval,
        "Starting HiveMindDB"
    );

    let config = config::HiveMindConfig {
        listen_addr: cli.listen_addr.clone(),
        rtdb_url: cli.rtdb_url.clone(),
        llm_provider: cli.llm_provider,
        llm_api_key: cli.llm_api_key,
        llm_model: cli.llm_model,
        embedding_model: cli.embedding_model,
        embedding_api_key: cli.embedding_api_key,
        data_dir: cli.data_dir.clone(),
    };

    // Shutdown signal
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Initialize engine
    let mut engine = memory_engine::MemoryEngine::new(config.clone());

    // Restore from snapshot if available
    let snapshot_mgr = persistence::SnapshotManager::new(&cli.data_dir);
    if let Some(snapshot) = snapshot_mgr.load().await? {
        engine.restore_from_snapshot(snapshot);
    }

    // Set up replication if enabled
    if cli.enable_replication {
        let (repl_tx, repl_rx) = tokio::sync::mpsc::unbounded_channel();
        engine.set_replication_tx(repl_tx);

        let repl_client =
            persistence::ReplicationClient::new(&cli.rtdb_url, shutdown_rx.clone());
        tokio::spawn(async move {
            repl_client.run(repl_rx).await;
        });
        info!("Replication client started");
    }

    let engine = Arc::new(engine);
    let channel_hub = Arc::new(channels::ChannelHub::new());

    // Start periodic snapshot task
    if cli.snapshot_interval > 0 {
        let engine_clone = engine.clone();
        let snapshot_fn = move || engine_clone.create_snapshot();
        tokio::spawn(persistence::snapshot_loop(
            persistence::SnapshotManager::new(&cli.data_dir),
            snapshot_fn,
            cli.snapshot_interval,
            shutdown_rx.clone(),
        ));
        info!(interval = cli.snapshot_interval, "Snapshot task started");
    }

    // Build and start the API server
    let app = api::router(engine.clone(), channel_hub.clone());

    let listener = tokio::net::TcpListener::bind(&cli.listen_addr).await?;
    info!(addr = %cli.listen_addr, "HiveMindDB API listening");

    // Graceful shutdown
    let server = axum::serve(listener, app).with_graceful_shutdown(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl+C");
        info!("Shutdown signal received");
        let _ = shutdown_tx.send(true);
    });

    server.await?;

    info!("HiveMindDB stopped");
    Ok(())
}
