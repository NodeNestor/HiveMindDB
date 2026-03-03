use axum::extract::ws::WebSocketUpgrade;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::channels::ChannelHub;
use crate::memory_engine::MemoryEngine;
use crate::types::*;
use crate::websocket;

pub struct AppState {
    pub engine: Arc<MemoryEngine>,
    pub channels: Arc<ChannelHub>,
    pub started_at: DateTime<Utc>,
}

pub fn router(engine: Arc<MemoryEngine>, channels: Arc<ChannelHub>) -> Router {
    let state = Arc::new(AppState {
        engine,
        channels,
        started_at: Utc::now(),
    });

    Router::new()
        // Memory endpoints
        .route("/api/v1/memories", post(add_memory))
        .route("/api/v1/bulk/memories", post(add_memories_bulk))
        .route("/api/v1/memories/{id}", get(get_memory))
        .route("/api/v1/memories/{id}", put(update_memory))
        .route("/api/v1/memories/{id}", delete(invalidate_memory))
        .route("/api/v1/memories/{id}/history", get(memory_history))
        .route("/api/v1/memories", get(list_memories))
        // Search
        .route("/api/v1/search", post(search))
        .route("/api/v1/search/bulk", post(bulk_search))
        // Extraction
        .route("/api/v1/extract", post(extract))
        // Knowledge Graph
        .route("/api/v1/entities", post(add_entity))
        .route("/api/v1/entities/{id}", get(get_entity))
        .route("/api/v1/entities/find", post(find_entity))
        .route("/api/v1/relationships", post(add_relationship))
        .route("/api/v1/entities/{id}/relationships", get(entity_relationships))
        .route("/api/v1/graph/traverse", post(graph_traverse))
        // Channels
        .route("/api/v1/channels", post(create_channel))
        .route("/api/v1/channels", get(list_channels))
        .route("/api/v1/channels/{id}/share", post(share_to_channel))
        // Tasks
        .route("/api/v1/tasks", post(create_task))
        .route("/api/v1/tasks", get(list_tasks))
        .route("/api/v1/tasks/{id}", get(get_task))
        .route("/api/v1/tasks/{id}/claim", post(claim_task))
        .route("/api/v1/tasks/{id}/start", post(start_task))
        .route("/api/v1/tasks/{id}/complete", post(complete_task))
        .route("/api/v1/tasks/{id}/fail", post(fail_task))
        .route("/api/v1/tasks/{id}/events", get(task_events))
        // Agents
        .route("/api/v1/agents/register", post(register_agent))
        .route("/api/v1/agents", get(list_agents))
        .route("/api/v1/agents/{agent_id}/heartbeat", post(agent_heartbeat))
        // WebSocket
        .route("/ws", get(ws_upgrade))
        // Benchmark
        .route("/api/v1/benchmark/run", post(benchmark_run))
        // System introspection
        .route("/api/v1/system/config", get(system_config))
        .route("/api/v1/system/topology", get(system_topology))
        .route("/api/v1/system/health", get(system_health))
        .route("/api/v1/system/embedding", get(system_embedding))
        // Status
        .route("/api/v1/status", get(status))
        .route("/health", get(health))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

// ============================================================================
// Memory Endpoints
// ============================================================================

async fn add_memory(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AddMemoryRequest>,
) -> (StatusCode, Json<Memory>) {
    let memory = state.engine.add_memory(req);

    // Broadcast to relevant channels
    if let Some(ref user_id) = memory.user_id {
        let channel_name = format!("user:{}", user_id);
        state.channels.broadcast_to_channel_by_name(
            &channel_name,
            WsServerMessage::MemoryAdded {
                channel: channel_name.clone(),
                memory: memory.clone(),
            },
        );
    }

    // Broadcast to global channel
    state.channels.broadcast_to_channel_by_name(
        "global",
        WsServerMessage::MemoryAdded {
            channel: "global".into(),
            memory: memory.clone(),
        },
    );

    (StatusCode::CREATED, Json(memory))
}

async fn add_memories_bulk(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BulkAddMemoryRequest>,
) -> (StatusCode, Json<Vec<Memory>>) {
    let memories = state.engine.add_memories_bulk(req.memories);

    // Broadcast to global channel
    for memory in &memories {
        state.channels.broadcast_to_channel_by_name(
            "global",
            WsServerMessage::MemoryAdded {
                channel: "global".into(),
                memory: memory.clone(),
            },
        );
    }

    (StatusCode::CREATED, Json(memories))
}

async fn get_memory(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<Memory>, StatusCode> {
    state
        .engine
        .get_memory(id)
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

async fn update_memory(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
    Json(req): Json<UpdateMemoryRequest>,
) -> Result<Json<Memory>, StatusCode> {
    let memory = state
        .engine
        .update_memory(id, req, "api")
        .ok_or(StatusCode::NOT_FOUND)?;

    // Broadcast update
    if let Some(ref user_id) = memory.user_id {
        let channel_name = format!("user:{}", user_id);
        state.channels.broadcast_to_channel_by_name(
            &channel_name,
            WsServerMessage::MemoryUpdated {
                channel: channel_name.clone(),
                memory: memory.clone(),
            },
        );
    }

    Ok(Json(memory))
}

#[derive(serde::Deserialize)]
struct InvalidateRequest {
    reason: String,
    #[serde(default = "default_api")]
    changed_by: String,
}

fn default_api() -> String {
    "api".into()
}

async fn invalidate_memory(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
    Json(req): Json<InvalidateRequest>,
) -> Result<Json<Memory>, StatusCode> {
    let memory = state
        .engine
        .invalidate_memory(id, &req.reason, &req.changed_by)
        .ok_or(StatusCode::NOT_FOUND)?;

    // Broadcast invalidation
    if let Some(ref user_id) = memory.user_id {
        let channel_name = format!("user:{}", user_id);
        state.channels.broadcast_to_channel_by_name(
            &channel_name,
            WsServerMessage::MemoryInvalidated {
                channel: channel_name.clone(),
                memory_id: id,
                reason: req.reason.clone(),
            },
        );
    }

    Ok(Json(memory))
}

async fn memory_history(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Json<Vec<MemoryHistory>> {
    Json(state.engine.get_memory_history(id))
}

#[derive(serde::Deserialize)]
struct ListMemoriesQuery {
    agent_id: Option<String>,
    user_id: Option<String>,
    #[serde(default)]
    include_invalidated: bool,
}

async fn list_memories(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(query): axum::extract::Query<ListMemoriesQuery>,
) -> Json<Vec<Memory>> {
    Json(state.engine.list_memories(
        query.agent_id.as_deref(),
        query.user_id.as_deref(),
        query.include_invalidated,
    ))
}

// ============================================================================
// Search (Hybrid: keyword + vector)
// ============================================================================

async fn search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> Json<Vec<SearchResult>> {
    // Use hybrid search (includes vector similarity when available)
    Json(state.engine.search_hybrid(&req).await)
}

// ============================================================================
// Bulk Search
// ============================================================================

async fn bulk_search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BulkSearchRequest>,
) -> Result<Json<BulkSearchResponse>, (StatusCode, String)> {
    let num_queries = req.queries.len();
    if num_queries > 100 {
        return Err((StatusCode::BAD_REQUEST, "Maximum 100 queries per bulk request".into()));
    }

    let max_concurrent = req.max_concurrent.min(50);
    let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent));
    let start = std::time::Instant::now();

    let mut handles = Vec::with_capacity(num_queries);
    for (idx, query) in req.queries.into_iter().enumerate() {
        let engine = state.engine.clone();
        let sem = semaphore.clone();
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let query_text = query.query.clone();
            let results = engine.search_hybrid(&query).await;
            BulkSearchQueryResult {
                query_index: idx,
                query: query_text,
                results,
                error: None,
            }
        }));
    }

    let mut query_results = Vec::with_capacity(num_queries);
    for handle in handles {
        match handle.await {
            Ok(result) => query_results.push(result),
            Err(e) => query_results.push(BulkSearchQueryResult {
                query_index: query_results.len(),
                query: String::new(),
                results: vec![],
                error: Some(format!("Task failed: {}", e)),
            }),
        }
    }

    query_results.sort_by_key(|r| r.query_index);
    let total_results: usize = query_results.iter().map(|r| r.results.len()).sum();

    Ok(Json(BulkSearchResponse {
        results: query_results,
        total_results,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

// ============================================================================
// Extraction (LLM-powered)
// ============================================================================

async fn extract(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExtractRequest>,
) -> Result<(StatusCode, Json<ExtractResponse>), (StatusCode, String)> {
    match state.engine.extract_and_store(&req).await {
        Ok(response) => {
            // Broadcast new memories to channels
            for memory in &response.memories_added {
                if let Some(ref user_id) = memory.user_id {
                    let channel_name = format!("user:{}", user_id);
                    state.channels.broadcast_to_channel_by_name(
                        &channel_name,
                        WsServerMessage::MemoryAdded {
                            channel: channel_name.clone(),
                            memory: memory.clone(),
                        },
                    );
                }
            }
            Ok((StatusCode::OK, Json(response)))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Extraction failed: {}", e),
        )),
    }
}

// ============================================================================
// Knowledge Graph
// ============================================================================

async fn add_entity(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AddEntityRequest>,
) -> (StatusCode, Json<Entity>) {
    (StatusCode::CREATED, Json(state.engine.add_entity(req)))
}

async fn get_entity(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<Entity>, StatusCode> {
    state
        .engine
        .get_entity(id)
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

#[derive(serde::Deserialize)]
struct FindEntityRequest {
    name: String,
}

async fn find_entity(
    State(state): State<Arc<AppState>>,
    Json(req): Json<FindEntityRequest>,
) -> Result<Json<Entity>, StatusCode> {
    state
        .engine
        .find_entity_by_name(&req.name)
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

async fn add_relationship(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AddRelationshipRequest>,
) -> (StatusCode, Json<Relationship>) {
    (
        StatusCode::CREATED,
        Json(state.engine.add_relationship(req)),
    )
}

async fn entity_relationships(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Json<Vec<(Relationship, Entity)>> {
    Json(state.engine.get_entity_relationships(id))
}

#[derive(serde::Deserialize)]
struct TraverseRequest {
    entity_id: u64,
    #[serde(default = "default_depth")]
    depth: usize,
}

fn default_depth() -> usize {
    2
}

async fn graph_traverse(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TraverseRequest>,
) -> Json<Vec<(Entity, Vec<Relationship>)>> {
    Json(state.engine.traverse(req.entity_id, req.depth))
}

// ============================================================================
// Channels
// ============================================================================

async fn create_channel(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateChannelRequest>,
) -> (StatusCode, Json<Channel>) {
    (StatusCode::CREATED, Json(state.channels.create_channel(req)))
}

async fn list_channels(State(state): State<Arc<AppState>>) -> Json<Vec<Channel>> {
    Json(state.channels.list_channels())
}

async fn share_to_channel(
    State(state): State<Arc<AppState>>,
    Path(channel_id): Path<u64>,
    Json(req): Json<ShareToChannelRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    let memory = state
        .engine
        .get_memory(req.memory_id)
        .ok_or(StatusCode::NOT_FOUND)?;

    let channel = state
        .channels
        .get_channel(channel_id)
        .ok_or(StatusCode::NOT_FOUND)?;

    state.channels.broadcast_to_channel(
        channel_id,
        WsServerMessage::MemoryAdded {
            channel: channel.name,
            memory,
        },
    );

    Ok((StatusCode::OK, "Memory shared to channel"))
}

// ============================================================================
// Agents
// ============================================================================

async fn register_agent(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterAgentRequest>,
) -> (StatusCode, Json<Agent>) {
    (
        StatusCode::CREATED,
        Json(state.engine.register_agent(req)),
    )
}

async fn list_agents(State(state): State<Arc<AppState>>) -> Json<Vec<Agent>> {
    Json(state.engine.list_agents())
}

async fn agent_heartbeat(
    State(state): State<Arc<AppState>>,
    Path(agent_id): Path<String>,
) -> StatusCode {
    state.engine.heartbeat_agent(&agent_id);
    StatusCode::OK
}

// ============================================================================
// Tasks
// ============================================================================

async fn create_task(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateTaskRequest>,
) -> (StatusCode, Json<Task>) {
    let task = state.engine.create_task(req);

    // Broadcast to tasks channel
    state.channels.broadcast_to_channel_by_name(
        "tasks",
        WsServerMessage::TaskCreated { task: task.clone() },
    );

    (StatusCode::CREATED, Json(task))
}

#[derive(serde::Deserialize)]
struct ListTasksQuery {
    status: Option<String>,
    agent_id: Option<String>,
}

async fn list_tasks(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(query): axum::extract::Query<ListTasksQuery>,
) -> Json<Vec<Task>> {
    let status = query.status.as_deref().and_then(|s| match s {
        "pending" => Some(TaskStatus::Pending),
        "claimed" => Some(TaskStatus::Claimed),
        "in_progress" => Some(TaskStatus::InProgress),
        "completed" => Some(TaskStatus::Completed),
        "failed" => Some(TaskStatus::Failed),
        "cancelled" => Some(TaskStatus::Cancelled),
        _ => None,
    });
    Json(state.engine.list_tasks(status.as_ref(), query.agent_id.as_deref(), None))
}

async fn get_task(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let task = state.engine.get_task(id).ok_or(StatusCode::NOT_FOUND)?;
    let events = state.engine.get_task_events(id);
    Ok(Json(serde_json::json!({
        "task": task,
        "events": events,
    })))
}

async fn claim_task(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
    Json(req): Json<ClaimTaskRequest>,
) -> Result<Json<Task>, (StatusCode, String)> {
    match state.engine.claim_task(id, &req.agent_id) {
        Ok(task) => {
            state.channels.broadcast_to_channel_by_name(
                "tasks",
                WsServerMessage::TaskClaimed { task: task.clone() },
            );
            Ok(Json(task))
        }
        Err(e) => Err((StatusCode::CONFLICT, e)),
    }
}

async fn start_task(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
    Json(req): Json<ClaimTaskRequest>,
) -> Result<Json<Task>, (StatusCode, String)> {
    match state.engine.start_task(id, &req.agent_id) {
        Ok(task) => {
            state.channels.broadcast_to_channel_by_name(
                "tasks",
                WsServerMessage::TaskUpdated { task: task.clone() },
            );
            Ok(Json(task))
        }
        Err(e) => Err((StatusCode::CONFLICT, e)),
    }
}

async fn complete_task(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
    Json(req): Json<CompleteTaskRequest>,
) -> Result<Json<Task>, (StatusCode, String)> {
    match state.engine.complete_task(id, &req.agent_id, req.result) {
        Ok(task) => {
            state.channels.broadcast_to_channel_by_name(
                "tasks",
                WsServerMessage::TaskCompleted { task: task.clone() },
            );
            Ok(Json(task))
        }
        Err(e) => Err((StatusCode::CONFLICT, e)),
    }
}

async fn fail_task(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
    Json(req): Json<FailTaskRequest>,
) -> Result<Json<Task>, (StatusCode, String)> {
    match state.engine.fail_task(id, &req.agent_id, req.reason) {
        Ok(task) => {
            state.channels.broadcast_to_channel_by_name(
                "tasks",
                WsServerMessage::TaskFailed { task: task.clone() },
            );
            Ok(Json(task))
        }
        Err(e) => Err((StatusCode::CONFLICT, e)),
    }
}

async fn task_events(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Json<Vec<TaskEvent>> {
    Json(state.engine.get_task_events(id))
}

// ============================================================================
// WebSocket
// ============================================================================

async fn ws_upgrade(
    State(state): State<Arc<AppState>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    let channels = state.channels.clone();
    ws.on_upgrade(move |socket| websocket::handle_ws_connection(socket, channels))
}

// ============================================================================
// Benchmark
// ============================================================================

async fn benchmark_run(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BenchmarkRequest>,
) -> Json<BenchmarkResponse> {
    let engine = state.engine.clone();
    let result = tokio::spawn(async move { engine.run_benchmark(&req).await })
        .await
        .unwrap();
    Json(result)
}

// ============================================================================
// System Introspection
// ============================================================================

async fn system_config(State(state): State<Arc<AppState>>) -> Json<SystemConfigResponse> {
    let config = state.engine.config();
    Json(SystemConfigResponse {
        listen_addr: config.listen_addr.clone(),
        data_dir: config.data_dir.clone(),
        embedding_model: config.embedding_model.clone(),
        embedding_pool_size: state.engine.embeddings().pool_size(),
        llm_provider: config.llm_provider.clone(),
        llm_model: config.llm_model.clone(),
        snapshot_interval: config.snapshot_interval,
        replication_enabled: config.replication_enabled,
    })
}

async fn system_topology(State(state): State<Arc<AppState>>) -> Json<SystemTopologyResponse> {
    let config = state.engine.config();
    Json(SystemTopologyResponse {
        node_id: hostname(),
        listen_addr: config.listen_addr.clone(),
        rtdb_url: config.rtdb_url.clone(),
        replication_enabled: config.replication_enabled,
        role: if config.replication_enabled { "replica".into() } else { "standalone".into() },
    })
}

async fn system_health(State(state): State<Arc<AppState>>) -> Json<SystemHealthResponse> {
    let (embedding, inverted_index, memory_store, knowledge_graph, tasks) =
        state.engine.health_details();

    let uptime = Utc::now()
        .signed_duration_since(state.started_at)
        .num_seconds()
        .max(0) as u64;

    Json(SystemHealthResponse {
        status: "healthy".into(),
        uptime_seconds: uptime,
        embedding,
        inverted_index,
        memory_store,
        knowledge_graph,
        tasks,
        websocket: WebSocketHealthInfo {
            active_connections: state.channels.active_ws_count(),
        },
    })
}

async fn system_embedding(State(state): State<Arc<AppState>>) -> Json<SystemEmbeddingResponse> {
    Json(state.engine.embedding_info())
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".into())
}

// ============================================================================
// Status
// ============================================================================

async fn status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(state.engine.stats())
}

async fn health() -> &'static str {
    "ok"
}
