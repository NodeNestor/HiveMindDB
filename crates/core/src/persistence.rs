use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

use crate::types::*;

/// Persistence layer for HiveMindDB.
///
/// Provides two mechanisms:
/// 1. **Snapshot persistence**: Periodic JSON snapshots to disk for restart recovery
/// 2. **RaftTimeDB replication**: Forward writes to RaftTimeDB via WebSocket for
///    cross-node replication through Raft consensus
///
/// The in-memory DashMap stores remain the source of truth for reads.
/// Persistence is async and best-effort — the system works without a running
/// RaftTimeDB cluster (standalone mode).

/// Snapshot of all in-memory state, serialized to disk.
#[derive(Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub version: u32,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub memories: Vec<Memory>,
    pub entities: Vec<Entity>,
    pub relationships: Vec<Relationship>,
    pub episodes: Vec<Episode>,
    pub agents: Vec<Agent>,
    pub history: Vec<(u64, Vec<MemoryHistory>)>,
    pub channels: Vec<Channel>,
    #[serde(default)]
    pub tasks: Vec<Task>,
    #[serde(default)]
    pub task_events: Vec<(u64, Vec<TaskEvent>)>,
}

impl Snapshot {
    pub const CURRENT_VERSION: u32 = 2;
}

/// Manages snapshot persistence to disk.
pub struct SnapshotManager {
    data_dir: PathBuf,
}

impl SnapshotManager {
    pub fn new(data_dir: &str) -> Self {
        Self {
            data_dir: PathBuf::from(data_dir),
        }
    }

    fn snapshot_path(&self) -> PathBuf {
        self.data_dir.join("hivemind-snapshot.json")
    }

    fn snapshot_tmp_path(&self) -> PathBuf {
        self.data_dir.join("hivemind-snapshot.json.tmp")
    }

    /// Save a snapshot to disk (atomic write via temp file + rename).
    pub async fn save(&self, snapshot: &Snapshot) -> anyhow::Result<()> {
        // Ensure data dir exists
        tokio::fs::create_dir_all(&self.data_dir).await?;

        let json = serde_json::to_string_pretty(snapshot)?;
        let tmp = self.snapshot_tmp_path();
        let target = self.snapshot_path();

        tokio::fs::write(&tmp, json).await?;
        tokio::fs::rename(&tmp, &target).await?;

        info!(
            path = %target.display(),
            memories = snapshot.memories.len(),
            entities = snapshot.entities.len(),
            "Snapshot saved"
        );
        Ok(())
    }

    /// Load a snapshot from disk.
    pub async fn load(&self) -> anyhow::Result<Option<Snapshot>> {
        let path = self.snapshot_path();
        if !path.exists() {
            debug!(path = %path.display(), "No snapshot found");
            return Ok(None);
        }

        let json = tokio::fs::read_to_string(&path).await?;
        let snapshot: Snapshot = serde_json::from_str(&json)?;

        info!(
            version = snapshot.version,
            memories = snapshot.memories.len(),
            entities = snapshot.entities.len(),
            "Snapshot loaded"
        );
        Ok(Some(snapshot))
    }
}

/// RaftTimeDB replication client.
///
/// Connects to a RaftTimeDB node's WebSocket endpoint and forwards
/// memory write operations as SpacetimeDB reducer calls through Raft consensus.
///
/// This ensures all HiveMindDB nodes in a cluster maintain identical state.
pub struct ReplicationClient {
    rtdb_url: String,
    connected: std::sync::atomic::AtomicBool,
    shutdown: watch::Receiver<bool>,
}

/// Replication event sent to RaftTimeDB.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReplicationEvent {
    MemoryAdded { memory: Memory },
    MemoryUpdated { memory: Memory },
    MemoryInvalidated { memory_id: u64, reason: String },
    EntityAdded { entity: Entity },
    RelationshipAdded { relationship: Relationship },
    AgentRegistered { agent: Agent },
    ChannelCreated { channel: Channel },
    TaskCreated { task: Task },
    TaskClaimed { task: Task },
    TaskCompleted { task: Task },
    TaskFailed { task: Task },
}

impl ReplicationClient {
    pub fn new(rtdb_url: &str, shutdown: watch::Receiver<bool>) -> Self {
        Self {
            rtdb_url: rtdb_url.to_string(),
            connected: std::sync::atomic::AtomicBool::new(false),
            shutdown,
        }
    }

    pub fn is_connected(&self) -> bool {
        self.connected
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Start the replication client — connects to RaftTimeDB and forwards events.
    ///
    /// Runs until the shutdown signal is received. Reconnects automatically on failure.
    pub async fn run(
        &self,
        mut event_rx: tokio::sync::mpsc::UnboundedReceiver<ReplicationEvent>,
    ) {
        info!(url = %self.rtdb_url, "Starting replication client");

        loop {
            // Check shutdown
            if *self.shutdown.borrow() {
                info!("Replication client shutting down");
                break;
            }

            match self.connect_and_forward(&mut event_rx).await {
                Ok(()) => {
                    info!("Replication session ended cleanly");
                    break;
                }
                Err(e) => {
                    warn!(error = %e, "Replication connection failed, retrying in 5s");
                    self.connected
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                    let mut shutdown = self.shutdown.clone();
                    tokio::select! {
                        _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {},
                        _ = shutdown.changed() => break,
                    }
                }
            }
        }
    }

    async fn connect_and_forward(
        &self,
        event_rx: &mut tokio::sync::mpsc::UnboundedReceiver<ReplicationEvent>,
    ) -> anyhow::Result<()> {
        use futures_util::SinkExt;
        use tokio_tungstenite::tungstenite;

        // Connect to RaftTimeDB WebSocket
        let ws_url = format!(
            "{}/database/subscribe/hivemind",
            self.rtdb_url.replace("http://", "ws://").replace("https://", "wss://")
        );

        let (ws_stream, _) = tokio_tungstenite::connect_async(&ws_url).await?;
        let (mut ws_tx, _ws_rx) = futures_util::StreamExt::split(ws_stream);

        self.connected
            .store(true, std::sync::atomic::Ordering::Relaxed);
        info!("Connected to RaftTimeDB for replication");

        let mut shutdown = self.shutdown.clone();
        loop {
            tokio::select! {
                event = event_rx.recv() => {
                    match event {
                        Some(evt) => {
                            let json = serde_json::to_string(&evt)?;
                            ws_tx.send(tungstenite::Message::Text(json.into())).await?;
                            debug!(event_type = ?std::mem::discriminant(&evt), "Replicated event");
                        }
                        None => break, // Channel closed
                    }
                }
                _ = shutdown.changed() => break,
            }
        }

        Ok(())
    }
}

/// Periodic snapshot task — saves snapshots at regular intervals.
pub async fn snapshot_loop(
    manager: SnapshotManager,
    snapshot_fn: impl Fn() -> Snapshot + Send + 'static,
    interval_secs: u64,
    mut shutdown: watch::Receiver<bool>,
) {
    let interval = std::time::Duration::from_secs(interval_secs);

    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {
                let snapshot = snapshot_fn();
                if let Err(e) = manager.save(&snapshot).await {
                    error!(error = %e, "Failed to save snapshot");
                }
            }
            _ = shutdown.changed() => {
                // Final snapshot on shutdown
                info!("Saving final snapshot before shutdown");
                let snapshot = snapshot_fn();
                if let Err(e) = manager.save(&snapshot).await {
                    error!(error = %e, "Failed to save final snapshot");
                }
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_serialize_deserialize() {
        let snapshot = Snapshot {
            version: Snapshot::CURRENT_VERSION,
            created_at: chrono::Utc::now(),
            memories: vec![Memory {
                id: 1,
                content: "test memory".into(),
                memory_type: MemoryType::Fact,
                agent_id: Some("agent-1".into()),
                user_id: Some("user-1".into()),
                session_id: None,
                confidence: 0.95,
                tags: vec!["test".into()],
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                valid_from: chrono::Utc::now(),
                valid_until: None,
                source: "test".into(),
                metadata: serde_json::json!({"key": "value"}),
            }],
            entities: vec![Entity {
                id: 1,
                name: "TestEntity".into(),
                entity_type: "Thing".into(),
                description: Some("A test entity".into()),
                agent_id: None,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                metadata: serde_json::Value::Null,
            }],
            relationships: vec![],
            episodes: vec![],
            agents: vec![],
            history: vec![],
            channels: vec![],
            tasks: vec![],
            task_events: vec![],
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        let loaded: Snapshot = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.version, Snapshot::CURRENT_VERSION);
        assert_eq!(loaded.memories.len(), 1);
        assert_eq!(loaded.memories[0].content, "test memory");
        assert_eq!(loaded.entities.len(), 1);
    }

    #[tokio::test]
    async fn test_snapshot_save_load() {
        let dir = std::env::temp_dir().join(format!("hivemind-test-{}", std::process::id()));
        let manager = SnapshotManager::new(dir.to_str().unwrap());

        let snapshot = Snapshot {
            version: Snapshot::CURRENT_VERSION,
            created_at: chrono::Utc::now(),
            memories: vec![Memory {
                id: 42,
                content: "Rust is great".into(),
                memory_type: MemoryType::Fact,
                agent_id: None,
                user_id: None,
                session_id: None,
                confidence: 1.0,
                tags: vec![],
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                valid_from: chrono::Utc::now(),
                valid_until: None,
                source: "test".into(),
                metadata: serde_json::Value::Null,
            }],
            entities: vec![],
            relationships: vec![],
            episodes: vec![],
            agents: vec![],
            history: vec![],
            channels: vec![],
            tasks: vec![],
            task_events: vec![],
        };

        manager.save(&snapshot).await.unwrap();

        let loaded = manager.load().await.unwrap().unwrap();
        assert_eq!(loaded.memories.len(), 1);
        assert_eq!(loaded.memories[0].id, 42);
        assert_eq!(loaded.memories[0].content, "Rust is great");

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn test_snapshot_load_missing() {
        let dir = std::env::temp_dir().join("hivemind-nonexistent-dir-12345");
        let manager = SnapshotManager::new(dir.to_str().unwrap());
        let loaded = manager.load().await.unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn test_replication_event_serialize() {
        let evt = ReplicationEvent::MemoryAdded {
            memory: Memory {
                id: 1,
                content: "test".into(),
                memory_type: MemoryType::Fact,
                agent_id: None,
                user_id: None,
                session_id: None,
                confidence: 1.0,
                tags: vec![],
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                valid_from: chrono::Utc::now(),
                valid_until: None,
                source: "test".into(),
                metadata: serde_json::Value::Null,
            },
        };
        let json = serde_json::to_string(&evt).unwrap();
        assert!(json.contains("\"type\":\"memory_added\""));
    }
}
