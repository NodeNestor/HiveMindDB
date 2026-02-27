//! HiveMindDB SpacetimeDB Module
//!
//! This WASM module runs inside SpacetimeDB and provides the persistent,
//! Raft-replicated storage layer for HiveMindDB. All state mutations happen
//! through reducers, which are replicated via RaftTimeDB consensus.
//!
//! Phase 1: Core tables and CRUD reducers.
//! Phase 2+: Entity extraction, vector indexing, channel pub/sub.

use spacetimedb::{ReducerContext, Table, Timestamp};

// ============================================================================
// Tables
// ============================================================================

#[spacetimedb::table(name = memories, public)]
pub struct Memory {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub content: String,
    pub memory_type: String, // "fact", "episodic", "procedural", "semantic"
    pub agent_id: String,    // empty = shared
    pub user_id: String,     // empty = no user scope
    pub session_id: String,
    pub confidence: f64,
    pub tags: String, // JSON array
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub valid_from: Timestamp,
    pub valid_until: String, // empty = still valid, otherwise ISO timestamp
    pub source: String,
    pub metadata: String, // JSON
}

#[spacetimedb::table(name = memory_history, public)]
pub struct MemoryHistory {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub memory_id: u64,
    pub operation: String, // "add", "update", "invalidate", "merge"
    pub old_content: String,
    pub new_content: String,
    pub reason: String,
    pub changed_by: String,
    pub timestamp: Timestamp,
}

#[spacetimedb::table(name = entities, public)]
pub struct Entity {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub name: String,
    pub entity_type: String,
    pub description: String,
    pub agent_id: String,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub metadata: String,
}

#[spacetimedb::table(name = relationships, public)]
pub struct Relationship {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub source_entity_id: u64,
    pub target_entity_id: u64,
    pub relation_type: String,
    pub description: String,
    pub weight: f64,
    pub valid_from: Timestamp,
    pub valid_until: String,
    pub created_by: String,
    pub metadata: String,
}

#[spacetimedb::table(name = episodes, public)]
pub struct Episode {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub agent_id: String,
    pub user_id: String,
    pub session_id: String,
    pub summary: String,
    pub key_decisions: String, // JSON array
    pub tools_used: String,   // JSON array
    pub outcome: String,
    pub started_at: Timestamp,
    pub ended_at: Timestamp,
    pub metadata: String,
}

#[spacetimedb::table(name = channels, public)]
pub struct Channel {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub name: String,
    pub description: String,
    pub channel_type: String, // "public", "private", "agent", "user"
    pub created_by: String,
    pub created_at: Timestamp,
}

#[spacetimedb::table(name = channel_memories, public)]
pub struct ChannelMemory {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub channel_id: u64,
    pub memory_id: u64,
    pub shared_by: String,
    pub shared_at: Timestamp,
}

#[spacetimedb::table(name = agents, public)]
pub struct Agent {
    #[primary_key]
    pub agent_id: String,
    pub name: String,
    pub agent_type: String,
    pub capabilities: String, // JSON array
    pub status: String,       // "online", "offline", "busy"
    pub last_seen: Timestamp,
    pub memory_count: u64,
    pub metadata: String,
}

// ============================================================================
// Reducers
// ============================================================================

#[spacetimedb::reducer]
pub fn add_memory(
    ctx: &ReducerContext,
    content: String,
    memory_type: String,
    agent_id: String,
    user_id: String,
    session_id: String,
    tags: String,
    metadata: String,
) {
    let now = Timestamp::now();
    let memory = Memory {
        id: 0, // auto_inc
        content: content.clone(),
        memory_type,
        agent_id: agent_id.clone(),
        user_id,
        session_id,
        confidence: 1.0,
        tags,
        created_at: now,
        updated_at: now,
        valid_from: now,
        valid_until: String::new(),
        source: if agent_id.is_empty() {
            "unknown".to_string()
        } else {
            agent_id
        },
        metadata,
    };
    ctx.db.memories().insert(memory);
    log::info!("Memory added: {}", &content[..content.len().min(50)]);
}

#[spacetimedb::reducer]
pub fn add_entity(
    ctx: &ReducerContext,
    name: String,
    entity_type: String,
    description: String,
    agent_id: String,
    metadata: String,
) {
    let now = Timestamp::now();
    let entity = Entity {
        id: 0,
        name: name.clone(),
        entity_type,
        description,
        agent_id,
        created_at: now,
        updated_at: now,
        metadata,
    };
    ctx.db.entities().insert(entity);
    log::info!("Entity added: {}", name);
}

#[spacetimedb::reducer]
pub fn add_relationship(
    ctx: &ReducerContext,
    source_entity_id: u64,
    target_entity_id: u64,
    relation_type: String,
    description: String,
    weight: f64,
    created_by: String,
    metadata: String,
) {
    let now = Timestamp::now();
    let rel = Relationship {
        id: 0,
        source_entity_id,
        target_entity_id,
        relation_type: relation_type.clone(),
        description,
        weight,
        valid_from: now,
        valid_until: String::new(),
        created_by,
        metadata,
    };
    ctx.db.relationships().insert(rel);
    log::info!(
        "Relationship added: {} --{}--> {}",
        source_entity_id,
        relation_type,
        target_entity_id
    );
}

#[spacetimedb::reducer]
pub fn register_agent(
    ctx: &ReducerContext,
    agent_id: String,
    name: String,
    agent_type: String,
    capabilities: String,
    metadata: String,
) {
    let now = Timestamp::now();
    let agent = Agent {
        agent_id: agent_id.clone(),
        name,
        agent_type,
        capabilities,
        status: "online".to_string(),
        last_seen: now,
        memory_count: 0,
        metadata,
    };
    ctx.db.agents().insert(agent);
    log::info!("Agent registered: {}", agent_id);
}

#[spacetimedb::reducer]
pub fn create_channel(
    ctx: &ReducerContext,
    name: String,
    description: String,
    channel_type: String,
    created_by: String,
) {
    let now = Timestamp::now();
    let channel = Channel {
        id: 0,
        name: name.clone(),
        description,
        channel_type,
        created_by,
        created_at: now,
    };
    ctx.db.channels().insert(channel);
    log::info!("Channel created: {}", name);
}

#[spacetimedb::reducer]
pub fn share_to_channel(
    ctx: &ReducerContext,
    channel_id: u64,
    memory_id: u64,
    shared_by: String,
) {
    let now = Timestamp::now();
    let cm = ChannelMemory {
        id: 0,
        channel_id,
        memory_id,
        shared_by,
        shared_at: now,
    };
    ctx.db.channel_memories().insert(cm);
    log::info!("Memory {} shared to channel {}", memory_id, channel_id);
}
