use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ============================================================================
// Memory Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    Fact,
    Episodic,
    Procedural,
    Semantic,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Add,
    Update,
    Invalidate,
    Merge,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: u64,
    pub content: String,
    pub memory_type: MemoryType,
    pub agent_id: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub confidence: f32,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub valid_from: DateTime<Utc>,
    pub valid_until: Option<DateTime<Utc>>,
    pub source: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHistory {
    pub id: u64,
    pub memory_id: u64,
    pub operation: Operation,
    pub old_content: Option<String>,
    pub new_content: String,
    pub reason: String,
    pub changed_by: String,
    pub timestamp: DateTime<Utc>,
}

// ============================================================================
// Knowledge Graph Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: u64,
    pub name: String,
    pub entity_type: String,
    pub description: Option<String>,
    pub agent_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub id: u64,
    pub source_entity_id: u64,
    pub target_entity_id: u64,
    pub relation_type: String,
    pub description: Option<String>,
    pub weight: f32,
    pub valid_from: DateTime<Utc>,
    pub valid_until: Option<DateTime<Utc>>,
    pub created_by: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

// ============================================================================
// Episode Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: u64,
    pub agent_id: String,
    pub user_id: Option<String>,
    pub session_id: String,
    pub summary: String,
    pub key_decisions: Vec<String>,
    pub tools_used: Vec<String>,
    pub outcome: String,
    pub started_at: DateTime<Utc>,
    pub ended_at: DateTime<Utc>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

// ============================================================================
// Channel Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChannelType {
    Public,
    Private,
    Agent,
    User,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Channel {
    pub id: u64,
    pub name: String,
    pub description: Option<String>,
    pub channel_type: ChannelType,
    pub created_by: String,
    pub created_at: DateTime<Utc>,
}

// ============================================================================
// Agent Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    Online,
    Offline,
    Busy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub agent_id: String,
    pub name: String,
    pub agent_type: String,
    pub capabilities: Vec<String>,
    pub status: AgentStatus,
    pub last_seen: DateTime<Utc>,
    pub memory_count: u64,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

// ============================================================================
// Search Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub include_graph: bool,
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub memory: Memory,
    pub score: f32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    pub related_entities: Vec<Entity>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    pub related_relationships: Vec<Relationship>,
}

// ============================================================================
// API Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct AddMemoryRequest {
    pub content: String,
    #[serde(default = "default_memory_type")]
    pub memory_type: MemoryType,
    pub agent_id: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

fn default_memory_type() -> MemoryType {
    MemoryType::Fact
}

#[derive(Debug, Deserialize)]
pub struct BulkAddMemoryRequest {
    pub memories: Vec<AddMemoryRequest>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateMemoryRequest {
    pub content: Option<String>,
    pub tags: Option<Vec<String>>,
    pub confidence: Option<f32>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ExtractRequest {
    pub messages: Vec<ConversationMessage>,
    pub agent_id: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct AddEntityRequest {
    pub name: String,
    pub entity_type: String,
    pub description: Option<String>,
    pub agent_id: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct AddRelationshipRequest {
    pub source_entity_id: u64,
    pub target_entity_id: u64,
    pub relation_type: String,
    pub description: Option<String>,
    #[serde(default = "default_weight")]
    pub weight: f32,
    pub created_by: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

fn default_weight() -> f32 {
    1.0
}

#[derive(Debug, Deserialize)]
pub struct CreateChannelRequest {
    pub name: String,
    pub description: Option<String>,
    #[serde(default = "default_channel_type")]
    pub channel_type: ChannelType,
    pub created_by: String,
}

fn default_channel_type() -> ChannelType {
    ChannelType::Public
}

#[derive(Debug, Deserialize)]
pub struct ShareToChannelRequest {
    pub memory_id: u64,
    pub shared_by: String,
}

#[derive(Debug, Deserialize)]
pub struct RegisterAgentRequest {
    pub agent_id: String,
    pub name: String,
    pub agent_type: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

// ============================================================================
// Task Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Claimed,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: u64,
    pub title: String,
    pub description: String,
    pub status: TaskStatus,
    pub priority: u32,
    #[serde(default)]
    pub required_capabilities: Vec<String>,
    pub assigned_agent: Option<String>,
    pub created_by: String,
    #[serde(default)]
    pub dependencies: Vec<u64>,
    pub result: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub deadline: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskEventType {
    Created,
    Claimed,
    Started,
    Progress,
    Completed,
    Failed,
    Cancelled,
    Reassigned,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEvent {
    pub id: u64,
    pub task_id: u64,
    pub event_type: TaskEventType,
    pub agent_id: Option<String>,
    pub details: Option<String>,
    pub timestamp: DateTime<Utc>,
}

// ============================================================================
// Task API Request Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct CreateTaskRequest {
    pub title: String,
    pub description: String,
    #[serde(default)]
    pub priority: u32,
    #[serde(default)]
    pub required_capabilities: Vec<String>,
    pub created_by: String,
    #[serde(default)]
    pub dependencies: Vec<u64>,
    #[serde(default)]
    pub deadline: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct ClaimTaskRequest {
    pub agent_id: String,
}

#[derive(Debug, Deserialize)]
pub struct CompleteTaskRequest {
    pub agent_id: String,
    pub result: String,
}

#[derive(Debug, Deserialize)]
pub struct FailTaskRequest {
    pub agent_id: String,
    pub reason: String,
}

// ============================================================================
// Extraction Response Types
// ============================================================================

/// Response from the extraction pipeline endpoint.
#[derive(Debug, Serialize)]
pub struct ExtractResponse {
    pub memories_added: Vec<Memory>,
    pub memories_updated: Vec<Memory>,
    pub entities_added: Vec<Entity>,
    pub relationships_added: Vec<Relationship>,
    pub skipped: usize,
}

// ============================================================================
// WebSocket Message Types
// ============================================================================

// ============================================================================
// Bulk Search Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct BulkSearchRequest {
    pub queries: Vec<SearchRequest>,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
}

fn default_max_concurrent() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct BulkSearchResponse {
    pub results: Vec<BulkSearchQueryResult>,
    pub total_results: usize,
    pub elapsed_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct BulkSearchQueryResult {
    pub query_index: usize,
    pub query: String,
    pub results: Vec<SearchResult>,
    pub error: Option<String>,
}

// ============================================================================
// Benchmark Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct BenchmarkRequest {
    #[serde(default = "default_benchmark_operations")]
    pub operations: Vec<String>,
    #[serde(default = "default_iterations")]
    pub iterations: usize,
    #[serde(default = "default_bench_concurrency")]
    pub concurrency: usize,
    #[serde(default = "default_true")]
    pub cleanup: bool,
}

fn default_benchmark_operations() -> Vec<String> {
    vec![
        "write".into(),
        "bulk_write".into(),
        "keyword_search".into(),
        "semantic_search".into(),
        "entity_create".into(),
        "graph_traverse".into(),
    ]
}

fn default_iterations() -> usize {
    100
}

fn default_bench_concurrency() -> usize {
    1
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Serialize)]
pub struct BenchmarkResponse {
    pub results: Vec<BenchmarkOperationResult>,
    pub system_info: BenchmarkSystemInfo,
    pub total_elapsed_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct BenchmarkOperationResult {
    pub operation: String,
    pub iterations: usize,
    pub total_ms: f64,
    pub latency: LatencyStats,
    pub ops_per_second: f64,
    pub errors: usize,
}

#[derive(Debug, Serialize)]
pub struct LatencyStats {
    pub min_us: f64,
    pub max_us: f64,
    pub avg_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
}

#[derive(Debug, Serialize)]
pub struct BenchmarkSystemInfo {
    pub memories_before: usize,
    pub memories_after: usize,
    pub entities_before: usize,
    pub entities_after: usize,
    pub embedding_provider: String,
    pub embedding_model: String,
}

// ============================================================================
// System Introspection Types
// ============================================================================

#[derive(Debug, Serialize)]
pub struct SystemConfigResponse {
    pub listen_addr: String,
    pub data_dir: String,
    pub embedding_model: String,
    pub embedding_pool_size: usize,
    pub llm_provider: String,
    pub llm_model: String,
    pub snapshot_interval: u64,
    pub replication_enabled: bool,
}

#[derive(Debug, Serialize)]
pub struct SystemTopologyResponse {
    pub node_id: String,
    pub listen_addr: String,
    pub rtdb_url: String,
    pub replication_enabled: bool,
    pub role: String,
}

#[derive(Debug, Serialize)]
pub struct SystemHealthResponse {
    pub status: String,
    pub uptime_seconds: u64,
    pub embedding: EmbeddingHealthInfo,
    pub inverted_index: InvertedIndexHealthInfo,
    pub memory_store: MemoryStoreHealthInfo,
    pub knowledge_graph: KnowledgeGraphHealthInfo,
    pub tasks: TasksHealthInfo,
    pub websocket: WebSocketHealthInfo,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingHealthInfo {
    pub available: bool,
    pub provider: String,
    pub model: String,
    pub indexed_count: usize,
    pub dimensions: u32,
    pub pool_size: usize,
}

#[derive(Debug, Serialize)]
pub struct InvertedIndexHealthInfo {
    pub unique_words: usize,
}

#[derive(Debug, Serialize)]
pub struct MemoryStoreHealthInfo {
    pub total_memories: usize,
    pub valid_memories: usize,
    pub invalidated_memories: usize,
    pub history_entries: usize,
}

#[derive(Debug, Serialize)]
pub struct KnowledgeGraphHealthInfo {
    pub entities: usize,
    pub relationships: usize,
}

#[derive(Debug, Serialize)]
pub struct TasksHealthInfo {
    pub total: usize,
    pub pending: usize,
    pub in_progress: usize,
    pub completed: usize,
    pub failed: usize,
}

#[derive(Debug, Serialize)]
pub struct WebSocketHealthInfo {
    pub active_connections: u64,
}

#[derive(Debug, Serialize)]
pub struct SystemEmbeddingResponse {
    pub provider: String,
    pub model: String,
    pub dimensions: u32,
    pub indexed_count: usize,
    pub pool_size: usize,
    pub available: bool,
}

// ============================================================================
// WebSocket Message Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsClientMessage {
    Subscribe {
        channels: Vec<String>,
        #[serde(default)]
        agent_id: Option<String>,
    },
    Unsubscribe {
        channels: Vec<String>,
    },
    SubscribeTasks {
        #[serde(default)]
        capabilities: Vec<String>,
        agent_id: String,
    },
    Ping,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsServerMessage {
    MemoryAdded { channel: String, memory: Memory },
    MemoryUpdated { channel: String, memory: Memory },
    MemoryInvalidated { channel: String, memory_id: u64, reason: String },
    EntityUpdated { channel: String, entity: Entity },
    Subscribed { channels: Vec<String> },
    TaskCreated { task: Task },
    TaskClaimed { task: Task },
    TaskUpdated { task: Task },
    TaskCompleted { task: Task },
    TaskFailed { task: Task },
    Pong,
    Error { message: String },
}
