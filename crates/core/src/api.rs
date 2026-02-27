use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::channels::ChannelHub;
use crate::memory_engine::MemoryEngine;
use crate::types::*;

pub struct AppState {
    pub engine: Arc<MemoryEngine>,
    pub channels: Arc<ChannelHub>,
}

pub fn router(engine: Arc<MemoryEngine>, channels: Arc<ChannelHub>) -> Router {
    let state = Arc::new(AppState { engine, channels });

    Router::new()
        // Memory endpoints
        .route("/api/v1/memories", post(add_memory))
        .route("/api/v1/memories/{id}", get(get_memory))
        .route("/api/v1/memories/{id}", put(update_memory))
        .route("/api/v1/memories/{id}", delete(invalidate_memory))
        .route("/api/v1/memories/{id}/history", get(memory_history))
        .route("/api/v1/memories", get(list_memories))
        // Search
        .route("/api/v1/search", post(search))
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
        // Agents
        .route("/api/v1/agents/register", post(register_agent))
        .route("/api/v1/agents", get(list_agents))
        .route("/api/v1/agents/{agent_id}/heartbeat", post(agent_heartbeat))
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

    (StatusCode::CREATED, Json(memory))
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
    state
        .engine
        .update_memory(id, req, "api")
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
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
    state
        .engine
        .invalidate_memory(id, &req.reason, &req.changed_by)
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
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
// Search
// ============================================================================

async fn search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> Json<Vec<SearchResult>> {
    Json(state.engine.search(&req))
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
// Status
// ============================================================================

async fn status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(state.engine.stats())
}

async fn health() -> &'static str {
    "ok"
}
