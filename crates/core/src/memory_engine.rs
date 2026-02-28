use crate::config::HiveMindConfig;
use crate::embeddings::{self, EmbeddingEngine};
use crate::extraction::{ExtractionOperation, ExtractionPipeline};
use crate::persistence::{ReplicationEvent, Snapshot};
use crate::types::*;
use chrono::Utc;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{info, warn};

/// Core memory engine — manages memories, entities, relationships, and search.
///
/// Integrates:
/// - In-memory stores (DashMap) for low-latency access
/// - LLM extraction pipeline for automatic knowledge extraction
/// - Vector embeddings for semantic search
/// - Snapshot persistence for restart recovery
/// - Replication events for multi-node sync via RaftTimeDB
pub struct MemoryEngine {
    config: HiveMindConfig,
    // In-memory stores
    memories: DashMap<u64, Memory>,
    entities: DashMap<u64, Entity>,
    relationships: DashMap<u64, Relationship>,
    episodes: DashMap<u64, Episode>,
    agents: DashMap<String, Agent>,
    history: DashMap<u64, Vec<MemoryHistory>>,
    next_memory_id: AtomicU64,
    next_entity_id: AtomicU64,
    next_relationship_id: AtomicU64,
    next_episode_id: AtomicU64,
    next_history_id: AtomicU64,
    // Extraction pipeline (LLM-powered)
    extraction: ExtractionPipeline,
    // Embedding engine (vector search)
    embeddings: Arc<EmbeddingEngine>,
    // Replication event sender (optional)
    replication_tx: Option<tokio::sync::mpsc::UnboundedSender<ReplicationEvent>>,
}

impl MemoryEngine {
    pub fn new(config: HiveMindConfig) -> Self {
        info!("Initializing memory engine");
        let extraction = ExtractionPipeline::from_hivemind_config(&config);
        let embeddings = Arc::new(EmbeddingEngine::from_hivemind_config(&config));

        Self {
            config,
            memories: DashMap::new(),
            entities: DashMap::new(),
            relationships: DashMap::new(),
            episodes: DashMap::new(),
            agents: DashMap::new(),
            history: DashMap::new(),
            next_memory_id: AtomicU64::new(1),
            next_entity_id: AtomicU64::new(1),
            next_relationship_id: AtomicU64::new(1),
            next_episode_id: AtomicU64::new(1),
            next_history_id: AtomicU64::new(1),
            extraction,
            embeddings,
            replication_tx: None,
        }
    }

    /// Set the replication event sender for RaftTimeDB sync.
    pub fn set_replication_tx(
        &mut self,
        tx: tokio::sync::mpsc::UnboundedSender<ReplicationEvent>,
    ) {
        self.replication_tx = Some(tx);
    }

    /// Get a reference to the embedding engine (for async operations).
    pub fn embeddings(&self) -> &Arc<EmbeddingEngine> {
        &self.embeddings
    }

    /// Restore state from a snapshot (called at startup).
    pub fn restore_from_snapshot(&mut self, snapshot: Snapshot) {
        let mut max_memory_id = 0u64;
        let mut max_entity_id = 0u64;
        let mut max_rel_id = 0u64;
        let mut max_episode_id = 0u64;
        let mut max_history_id = 0u64;

        for memory in snapshot.memories {
            max_memory_id = max_memory_id.max(memory.id);
            self.memories.insert(memory.id, memory);
        }
        for entity in snapshot.entities {
            max_entity_id = max_entity_id.max(entity.id);
            self.entities.insert(entity.id, entity);
        }
        for rel in snapshot.relationships {
            max_rel_id = max_rel_id.max(rel.id);
            self.relationships.insert(rel.id, rel);
        }
        for episode in snapshot.episodes {
            max_episode_id = max_episode_id.max(episode.id);
            self.episodes.insert(episode.id, episode);
        }
        for agent in snapshot.agents {
            self.agents.insert(agent.agent_id.clone(), agent);
        }
        for (memory_id, hist_entries) in snapshot.history {
            for h in &hist_entries {
                max_history_id = max_history_id.max(h.id);
            }
            self.history.insert(memory_id, hist_entries);
        }

        // Set counters past the max existing IDs
        self.next_memory_id.store(max_memory_id + 1, Ordering::Relaxed);
        self.next_entity_id.store(max_entity_id + 1, Ordering::Relaxed);
        self.next_relationship_id.store(max_rel_id + 1, Ordering::Relaxed);
        self.next_episode_id.store(max_episode_id + 1, Ordering::Relaxed);
        self.next_history_id.store(max_history_id + 1, Ordering::Relaxed);

        info!(
            memories = self.memories.len(),
            entities = self.entities.len(),
            relationships = self.relationships.len(),
            agents = self.agents.len(),
            "State restored from snapshot"
        );
    }

    /// Create a snapshot of current state.
    pub fn create_snapshot(&self) -> Snapshot {
        Snapshot {
            version: Snapshot::CURRENT_VERSION,
            created_at: Utc::now(),
            memories: self.memories.iter().map(|m| m.value().clone()).collect(),
            entities: self.entities.iter().map(|e| e.value().clone()).collect(),
            relationships: self.relationships.iter().map(|r| r.value().clone()).collect(),
            episodes: self.episodes.iter().map(|e| e.value().clone()).collect(),
            agents: self.agents.iter().map(|a| a.value().clone()).collect(),
            history: self.history.iter().map(|h| (*h.key(), h.value().clone())).collect(),
            channels: vec![], // Channels are managed by ChannelHub
        }
    }

    fn emit_replication(&self, event: ReplicationEvent) {
        if let Some(ref tx) = self.replication_tx {
            let _ = tx.send(event);
        }
    }

    // ========================================================================
    // Memory CRUD
    // ========================================================================

    pub fn add_memory(&self, req: AddMemoryRequest) -> Memory {
        let id = self.next_memory_id.fetch_add(1, Ordering::Relaxed);
        let now = Utc::now();

        let memory = Memory {
            id,
            content: req.content.clone(),
            memory_type: req.memory_type,
            agent_id: req.agent_id.clone(),
            user_id: req.user_id.clone(),
            session_id: req.session_id,
            confidence: 1.0,
            tags: req.tags,
            created_at: now,
            updated_at: now,
            valid_from: now,
            valid_until: None,
            source: req.agent_id.unwrap_or_else(|| "unknown".into()),
            metadata: req.metadata,
        };

        // Record history
        let hist_id = self.next_history_id.fetch_add(1, Ordering::Relaxed);
        let hist = MemoryHistory {
            id: hist_id,
            memory_id: id,
            operation: Operation::Add,
            old_content: None,
            new_content: req.content,
            reason: "Initial creation".into(),
            changed_by: memory.source.clone(),
            timestamp: now,
        };
        self.history.entry(id).or_default().push(hist);
        self.memories.insert(id, memory.clone());

        // Async: index embedding (fire-and-forget)
        if self.embeddings.is_available() {
            let emb = self.embeddings.clone();
            let mem = memory.clone();
            tokio::spawn(async move {
                if let Err(e) = emb.index_memory(&mem).await {
                    warn!(memory_id = mem.id, error = %e, "Failed to index memory embedding");
                }
            });
        }

        self.emit_replication(ReplicationEvent::MemoryAdded {
            memory: memory.clone(),
        });

        info!(id, "Memory added");
        memory
    }

    pub fn get_memory(&self, id: u64) -> Option<Memory> {
        self.memories.get(&id).map(|m| m.clone())
    }

    pub fn update_memory(
        &self,
        id: u64,
        req: UpdateMemoryRequest,
        changed_by: &str,
    ) -> Option<Memory> {
        let mut entry = self.memories.get_mut(&id)?;
        let old_content = entry.content.clone();

        if let Some(content) = &req.content {
            entry.content = content.clone();
        }
        if let Some(tags) = req.tags {
            entry.tags = tags;
        }
        if let Some(confidence) = req.confidence {
            entry.confidence = confidence;
        }
        if let Some(metadata) = req.metadata {
            entry.metadata = metadata;
        }
        entry.updated_at = Utc::now();

        // Record history
        let hist_id = self.next_history_id.fetch_add(1, Ordering::Relaxed);
        let hist = MemoryHistory {
            id: hist_id,
            memory_id: id,
            operation: Operation::Update,
            old_content: Some(old_content),
            new_content: entry.content.clone(),
            reason: "Manual update".into(),
            changed_by: changed_by.into(),
            timestamp: Utc::now(),
        };
        self.history.entry(id).or_default().push(hist);

        let memory = entry.clone();

        // Re-index embedding if content changed
        if req.content.is_some() && self.embeddings.is_available() {
            let emb = self.embeddings.clone();
            let mem = memory.clone();
            tokio::spawn(async move {
                if let Err(e) = emb.index_memory(&mem).await {
                    warn!(memory_id = mem.id, error = %e, "Failed to re-index memory embedding");
                }
            });
        }

        self.emit_replication(ReplicationEvent::MemoryUpdated {
            memory: memory.clone(),
        });

        info!(id, "Memory updated");
        Some(memory)
    }

    pub fn invalidate_memory(&self, id: u64, reason: &str, changed_by: &str) -> Option<Memory> {
        let mut entry = self.memories.get_mut(&id)?;
        entry.valid_until = Some(Utc::now());
        entry.updated_at = Utc::now();

        let hist_id = self.next_history_id.fetch_add(1, Ordering::Relaxed);
        let hist = MemoryHistory {
            id: hist_id,
            memory_id: id,
            operation: Operation::Invalidate,
            old_content: Some(entry.content.clone()),
            new_content: entry.content.clone(),
            reason: reason.into(),
            changed_by: changed_by.into(),
            timestamp: Utc::now(),
        };
        self.history.entry(id).or_default().push(hist);

        // Remove from embedding index
        self.embeddings.remove_memory(id);

        let memory = entry.clone();

        self.emit_replication(ReplicationEvent::MemoryInvalidated {
            memory_id: id,
            reason: reason.into(),
        });

        info!(id, reason, "Memory invalidated");
        Some(memory)
    }

    pub fn get_memory_history(&self, memory_id: u64) -> Vec<MemoryHistory> {
        self.history
            .get(&memory_id)
            .map(|h| h.clone())
            .unwrap_or_default()
    }

    // ========================================================================
    // Search (Hybrid: keyword + vector)
    // ========================================================================

    /// Hybrid search combining keyword matching and vector similarity.
    ///
    /// If embeddings are available, uses 70% vector + 30% keyword scoring.
    /// Falls back to pure keyword search if embeddings are not configured.
    pub fn search(&self, req: &SearchRequest) -> Vec<SearchResult> {
        self.search_keyword(req)
    }

    /// Async search that includes vector similarity when embeddings are available.
    pub async fn search_hybrid(&self, req: &SearchRequest) -> Vec<SearchResult> {
        // Get keyword results
        let keyword_results = self.search_keyword(req);

        // If embeddings aren't available, return keyword results
        if !self.embeddings.is_available() || self.embeddings.indexed_count() == 0 {
            return keyword_results;
        }

        // Get vector similarity scores
        let vector_scores = match self.embeddings.search(&req.query, req.limit * 2).await {
            Ok(scores) => scores,
            Err(e) => {
                warn!(error = %e, "Vector search failed, using keyword only");
                return keyword_results;
            }
        };

        // Build a map of memory_id → vector_score
        let vector_map: std::collections::HashMap<u64, f32> =
            vector_scores.into_iter().collect();

        // Merge: for each keyword result, enhance with vector score
        let mut results: Vec<SearchResult> = keyword_results
            .into_iter()
            .map(|mut r| {
                if let Some(&vec_score) = vector_map.get(&r.memory.id) {
                    r.score = embeddings::hybrid_score(r.score, vec_score, 0.7);
                }
                r
            })
            .collect();

        // Add any vector-only results (high vector score but no keyword match)
        for (memory_id, vec_score) in &vector_map {
            if !results.iter().any(|r| r.memory.id == *memory_id) {
                if let Some(memory) = self.get_memory(*memory_id) {
                    // Apply filters
                    if memory.valid_until.is_some() {
                        continue;
                    }
                    if let Some(ref agent_id) = req.agent_id {
                        if memory.agent_id.as_ref() != Some(agent_id) && memory.agent_id.is_some() {
                            continue;
                        }
                    }
                    if let Some(ref user_id) = req.user_id {
                        if memory.user_id.as_ref() != Some(user_id) && memory.user_id.is_some() {
                            continue;
                        }
                    }
                    if *vec_score > 0.3 {
                        // Minimum threshold for vector-only results
                        results.push(SearchResult {
                            memory,
                            score: embeddings::hybrid_score(0.0, *vec_score, 0.7),
                            related_entities: vec![],
                            related_relationships: vec![],
                        });
                    }
                }
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(req.limit);
        results
    }

    /// Keyword-only search (synchronous, always available).
    fn search_keyword(&self, req: &SearchRequest) -> Vec<SearchResult> {
        let query_lower = req.query.to_lowercase();

        let mut results: Vec<SearchResult> = self
            .memories
            .iter()
            .filter(|entry| {
                let m = entry.value();
                if m.valid_until.is_some() {
                    return false;
                }
                if let Some(ref agent_id) = req.agent_id {
                    if m.agent_id.as_ref() != Some(agent_id) && m.agent_id.is_some() {
                        return false;
                    }
                }
                if let Some(ref user_id) = req.user_id {
                    if m.user_id.as_ref() != Some(user_id) && m.user_id.is_some() {
                        return false;
                    }
                }
                if !req.tags.is_empty()
                    && !req.tags.iter().any(|t| m.tags.contains(t))
                {
                    return false;
                }
                m.content.to_lowercase().contains(&query_lower)
                    || m.tags.iter().any(|t| t.to_lowercase().contains(&query_lower))
            })
            .map(|entry| {
                let m = entry.value();
                let content_lower = m.content.to_lowercase();
                let score = if content_lower == query_lower {
                    1.0
                } else {
                    let words: Vec<&str> = query_lower.split_whitespace().collect();
                    let matches = words
                        .iter()
                        .filter(|w| content_lower.contains(*w))
                        .count();
                    matches as f32 / words.len().max(1) as f32
                };

                SearchResult {
                    memory: m.clone(),
                    score,
                    related_entities: vec![],
                    related_relationships: vec![],
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(req.limit);
        results
    }

    /// Get all memories, optionally filtered by agent/user.
    pub fn list_memories(
        &self,
        agent_id: Option<&str>,
        user_id: Option<&str>,
        include_invalidated: bool,
    ) -> Vec<Memory> {
        self.memories
            .iter()
            .filter(|entry| {
                let m = entry.value();
                if !include_invalidated && m.valid_until.is_some() {
                    return false;
                }
                if let Some(aid) = agent_id {
                    if m.agent_id.as_deref() != Some(aid) {
                        return false;
                    }
                }
                if let Some(uid) = user_id {
                    if m.user_id.as_deref() != Some(uid) {
                        return false;
                    }
                }
                true
            })
            .map(|entry| entry.value().clone())
            .collect()
    }

    // ========================================================================
    // LLM Extraction
    // ========================================================================

    /// Extract knowledge from conversation using the LLM pipeline.
    ///
    /// Automatically:
    /// - Extracts facts and stores them as memories
    /// - Creates entities in the knowledge graph
    /// - Creates relationships between entities
    /// - Handles conflict resolution (add/update/noop)
    pub async fn extract_and_store(
        &self,
        req: &ExtractRequest,
    ) -> anyhow::Result<ExtractResponse> {
        if !self.extraction.is_available() {
            anyhow::bail!("Extraction pipeline not configured — set LLM API key or use a local provider");
        }

        // Gather existing memories for conflict resolution
        let existing: Vec<Memory> = self.list_memories(
            req.agent_id.as_deref(),
            req.user_id.as_deref(),
            false,
        );

        let result = self
            .extraction
            .extract(&req.messages, &existing)
            .await?;

        let mut response = ExtractResponse {
            memories_added: vec![],
            memories_updated: vec![],
            entities_added: vec![],
            relationships_added: vec![],
            skipped: 0,
        };

        // Process extracted facts
        for fact in &result.facts {
            match fact.operation {
                ExtractionOperation::Add => {
                    let memory = self.add_memory(AddMemoryRequest {
                        content: fact.content.clone(),
                        memory_type: fact.memory_type.clone(),
                        agent_id: req.agent_id.clone(),
                        user_id: req.user_id.clone(),
                        session_id: req.session_id.clone(),
                        tags: fact.tags.clone(),
                        metadata: serde_json::json!({
                            "confidence": fact.confidence,
                            "extracted": true,
                        }),
                    });
                    response.memories_added.push(memory);
                }
                ExtractionOperation::Update => {
                    if let Some(target_id) = fact.updates_memory_id {
                        if let Some(updated) = self.update_memory(
                            target_id,
                            UpdateMemoryRequest {
                                content: Some(fact.content.clone()),
                                tags: Some(fact.tags.clone()),
                                confidence: Some(fact.confidence),
                                metadata: None,
                            },
                            req.agent_id.as_deref().unwrap_or("extraction"),
                        ) {
                            response.memories_updated.push(updated);
                        }
                    } else {
                        // No target ID — add as new memory
                        let memory = self.add_memory(AddMemoryRequest {
                            content: fact.content.clone(),
                            memory_type: fact.memory_type.clone(),
                            agent_id: req.agent_id.clone(),
                            user_id: req.user_id.clone(),
                            session_id: req.session_id.clone(),
                            tags: fact.tags.clone(),
                            metadata: serde_json::json!({
                                "confidence": fact.confidence,
                                "extracted": true,
                            }),
                        });
                        response.memories_added.push(memory);
                    }
                }
                ExtractionOperation::Noop => {
                    response.skipped += 1;
                }
            }
        }

        // Process extracted entities
        for entity in &result.entities {
            // Check if entity already exists
            if self.find_entity_by_name(&entity.name).is_some() {
                continue;
            }
            let e = self.add_entity(AddEntityRequest {
                name: entity.name.clone(),
                entity_type: entity.entity_type.clone(),
                description: entity.description.clone(),
                agent_id: req.agent_id.clone(),
                metadata: serde_json::json!({"extracted": true}),
            });
            response.entities_added.push(e);
        }

        // Process extracted relationships
        for rel in &result.relationships {
            let source = self.find_entity_by_name(&rel.source_entity);
            let target = self.find_entity_by_name(&rel.target_entity);

            if let (Some(src), Some(tgt)) = (source, target) {
                let r = self.add_relationship(AddRelationshipRequest {
                    source_entity_id: src.id,
                    target_entity_id: tgt.id,
                    relation_type: rel.relation_type.clone(),
                    description: rel.description.clone(),
                    weight: 1.0,
                    created_by: req.agent_id.clone().unwrap_or_else(|| "extraction".into()),
                    metadata: serde_json::json!({"extracted": true}),
                });
                response.relationships_added.push(r);
            }
        }

        info!(
            added = response.memories_added.len(),
            updated = response.memories_updated.len(),
            entities = response.entities_added.len(),
            relationships = response.relationships_added.len(),
            skipped = response.skipped,
            "Extraction processed"
        );

        Ok(response)
    }

    // ========================================================================
    // Knowledge Graph
    // ========================================================================

    pub fn add_entity(&self, req: AddEntityRequest) -> Entity {
        let id = self.next_entity_id.fetch_add(1, Ordering::Relaxed);
        let now = Utc::now();

        let entity = Entity {
            id,
            name: req.name,
            entity_type: req.entity_type,
            description: req.description,
            agent_id: req.agent_id,
            created_at: now,
            updated_at: now,
            metadata: req.metadata,
        };

        self.entities.insert(id, entity.clone());

        self.emit_replication(ReplicationEvent::EntityAdded {
            entity: entity.clone(),
        });

        info!(id, name = %entity.name, "Entity added");
        entity
    }

    pub fn get_entity(&self, id: u64) -> Option<Entity> {
        self.entities.get(&id).map(|e| e.clone())
    }

    pub fn find_entity_by_name(&self, name: &str) -> Option<Entity> {
        let name_lower = name.to_lowercase();
        self.entities
            .iter()
            .find(|e| e.value().name.to_lowercase() == name_lower)
            .map(|e| e.value().clone())
    }

    pub fn add_relationship(&self, req: AddRelationshipRequest) -> Relationship {
        let id = self.next_relationship_id.fetch_add(1, Ordering::Relaxed);
        let now = Utc::now();

        let rel = Relationship {
            id,
            source_entity_id: req.source_entity_id,
            target_entity_id: req.target_entity_id,
            relation_type: req.relation_type,
            description: req.description,
            weight: req.weight,
            valid_from: now,
            valid_until: None,
            created_by: req.created_by,
            metadata: req.metadata,
        };

        self.relationships.insert(id, rel.clone());

        self.emit_replication(ReplicationEvent::RelationshipAdded {
            relationship: rel.clone(),
        });

        info!(id, src = req.source_entity_id, dst = req.target_entity_id, "Relationship added");
        rel
    }

    pub fn get_entity_relationships(&self, entity_id: u64) -> Vec<(Relationship, Entity)> {
        self.relationships
            .iter()
            .filter(|r| {
                let rel = r.value();
                rel.valid_until.is_none()
                    && (rel.source_entity_id == entity_id || rel.target_entity_id == entity_id)
            })
            .filter_map(|r| {
                let rel = r.value().clone();
                let other_id = if rel.source_entity_id == entity_id {
                    rel.target_entity_id
                } else {
                    rel.source_entity_id
                };
                let other_entity = self.entities.get(&other_id)?.clone();
                Some((rel, other_entity))
            })
            .collect()
    }

    /// Simple graph traversal: find all entities within N hops.
    pub fn traverse(&self, start_entity_id: u64, max_depth: usize) -> Vec<(Entity, Vec<Relationship>)> {
        let mut visited = std::collections::HashSet::new();
        let mut result = Vec::new();
        let mut frontier = vec![(start_entity_id, 0usize)];

        while let Some((entity_id, depth)) = frontier.pop() {
            if depth > max_depth || !visited.insert(entity_id) {
                continue;
            }

            if let Some(entity) = self.entities.get(&entity_id) {
                let rels: Vec<Relationship> = self
                    .relationships
                    .iter()
                    .filter(|r| {
                        let rel = r.value();
                        rel.valid_until.is_none()
                            && (rel.source_entity_id == entity_id
                                || rel.target_entity_id == entity_id)
                    })
                    .map(|r| r.value().clone())
                    .collect();

                for rel in &rels {
                    let next_id = if rel.source_entity_id == entity_id {
                        rel.target_entity_id
                    } else {
                        rel.source_entity_id
                    };
                    if !visited.contains(&next_id) {
                        frontier.push((next_id, depth + 1));
                    }
                }

                result.push((entity.clone(), rels));
            }
        }

        result
    }

    // ========================================================================
    // Agents
    // ========================================================================

    pub fn register_agent(&self, req: RegisterAgentRequest) -> Agent {
        let now = Utc::now();
        let agent = Agent {
            agent_id: req.agent_id.clone(),
            name: req.name,
            agent_type: req.agent_type,
            capabilities: req.capabilities,
            status: AgentStatus::Online,
            last_seen: now,
            memory_count: 0,
            metadata: req.metadata,
        };

        self.agents.insert(req.agent_id.clone(), agent.clone());

        self.emit_replication(ReplicationEvent::AgentRegistered {
            agent: agent.clone(),
        });

        info!(agent_id = %req.agent_id, "Agent registered");
        agent
    }

    pub fn get_agent(&self, agent_id: &str) -> Option<Agent> {
        self.agents.get(agent_id).map(|a| a.clone())
    }

    pub fn list_agents(&self) -> Vec<Agent> {
        self.agents.iter().map(|a| a.value().clone()).collect()
    }

    pub fn heartbeat_agent(&self, agent_id: &str) {
        if let Some(mut agent) = self.agents.get_mut(agent_id) {
            agent.last_seen = Utc::now();
            agent.status = AgentStatus::Online;
        }
    }

    // ========================================================================
    // Stats
    // ========================================================================

    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "memories": self.memories.len(),
            "entities": self.entities.len(),
            "relationships": self.relationships.len(),
            "episodes": self.episodes.len(),
            "agents": self.agents.len(),
            "valid_memories": self.memories.iter().filter(|m| m.value().valid_until.is_none()).count(),
            "embeddings_indexed": self.embeddings.indexed_count(),
            "embedding_dimensions": self.embeddings.dimensions(),
            "extraction_available": self.extraction.is_available(),
            "replication_enabled": self.replication_tx.is_some(),
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HiveMindConfig;

    fn test_config() -> HiveMindConfig {
        HiveMindConfig {
            listen_addr: "0.0.0.0:8100".into(),
            rtdb_url: "ws://localhost:3001".into(),
            llm_provider: "test".into(),
            llm_api_key: None,
            llm_model: "test".into(),
            embedding_model: "none:disabled".into(),
            embedding_api_key: None,
            data_dir: "/tmp/hivemind-test".into(),
        }
    }

    #[test]
    fn test_add_and_get_memory() {
        let engine = MemoryEngine::new(test_config());
        let mem = engine.add_memory(AddMemoryRequest {
            content: "User prefers Rust".into(),
            memory_type: MemoryType::Fact,
            agent_id: Some("agent-1".into()),
            user_id: Some("ludde".into()),
            session_id: None,
            tags: vec!["preferences".into()],
            metadata: serde_json::Value::Null,
        });

        assert_eq!(mem.content, "User prefers Rust");
        assert_eq!(mem.agent_id.as_deref(), Some("agent-1"));

        let retrieved = engine.get_memory(mem.id).unwrap();
        assert_eq!(retrieved.id, mem.id);
        assert_eq!(retrieved.content, "User prefers Rust");
    }

    #[test]
    fn test_update_memory() {
        let engine = MemoryEngine::new(test_config());
        let mem = engine.add_memory(AddMemoryRequest {
            content: "User likes Python".into(),
            memory_type: MemoryType::Fact,
            agent_id: None,
            user_id: None,
            session_id: None,
            tags: vec![],
            metadata: serde_json::Value::Null,
        });

        let updated = engine
            .update_memory(
                mem.id,
                UpdateMemoryRequest {
                    content: Some("User prefers Rust over Python".into()),
                    tags: Some(vec!["preferences".into(), "languages".into()]),
                    confidence: None,
                    metadata: None,
                },
                "test-agent",
            )
            .unwrap();

        assert_eq!(updated.content, "User prefers Rust over Python");
        assert_eq!(updated.tags, vec!["preferences", "languages"]);
    }

    #[test]
    fn test_invalidate_memory() {
        let engine = MemoryEngine::new(test_config());
        let mem = engine.add_memory(AddMemoryRequest {
            content: "User works at Acme".into(),
            memory_type: MemoryType::Fact,
            agent_id: None,
            user_id: Some("ludde".into()),
            session_id: None,
            tags: vec![],
            metadata: serde_json::Value::Null,
        });

        let invalidated = engine
            .invalidate_memory(mem.id, "User changed jobs", "agent-1")
            .unwrap();
        assert!(invalidated.valid_until.is_some());

        let results = engine.search(&SearchRequest {
            query: "works at Acme".into(),
            agent_id: None,
            user_id: None,
            tags: vec![],
            limit: 10,
            include_graph: false,
        });
        assert!(results.is_empty());
    }

    #[test]
    fn test_memory_history() {
        let engine = MemoryEngine::new(test_config());
        let mem = engine.add_memory(AddMemoryRequest {
            content: "Original".into(),
            memory_type: MemoryType::Fact,
            agent_id: None,
            user_id: None,
            session_id: None,
            tags: vec![],
            metadata: serde_json::Value::Null,
        });

        engine.update_memory(
            mem.id,
            UpdateMemoryRequest {
                content: Some("Updated".into()),
                tags: None,
                confidence: None,
                metadata: None,
            },
            "test",
        );

        engine.invalidate_memory(mem.id, "outdated", "test");

        let history = engine.get_memory_history(mem.id);
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].operation, Operation::Add);
        assert_eq!(history[1].operation, Operation::Update);
        assert_eq!(history[2].operation, Operation::Invalidate);
    }

    #[test]
    fn test_search_basic() {
        let engine = MemoryEngine::new(test_config());

        engine.add_memory(AddMemoryRequest {
            content: "User prefers dark mode".into(),
            memory_type: MemoryType::Fact,
            agent_id: None,
            user_id: Some("ludde".into()),
            session_id: None,
            tags: vec!["preferences".into(), "ui".into()],
            metadata: serde_json::Value::Null,
        });

        engine.add_memory(AddMemoryRequest {
            content: "User likes Italian food".into(),
            memory_type: MemoryType::Fact,
            agent_id: None,
            user_id: Some("ludde".into()),
            session_id: None,
            tags: vec!["preferences".into(), "food".into()],
            metadata: serde_json::Value::Null,
        });

        engine.add_memory(AddMemoryRequest {
            content: "RaftTimeDB uses openraft".into(),
            memory_type: MemoryType::Semantic,
            agent_id: None,
            user_id: None,
            session_id: None,
            tags: vec!["technical".into()],
            metadata: serde_json::Value::Null,
        });

        let results = engine.search(&SearchRequest {
            query: "dark mode".into(),
            agent_id: None,
            user_id: None,
            tags: vec![],
            limit: 10,
            include_graph: false,
        });
        assert_eq!(results.len(), 1);
        assert!(results[0].memory.content.contains("dark mode"));

        let results = engine.search(&SearchRequest {
            query: "preferences".into(),
            agent_id: None,
            user_id: None,
            tags: vec!["preferences".into()],
            limit: 10,
            include_graph: false,
        });
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_filters_by_user() {
        let engine = MemoryEngine::new(test_config());

        engine.add_memory(AddMemoryRequest {
            content: "Alice prefers cats".into(),
            memory_type: MemoryType::Fact,
            agent_id: None,
            user_id: Some("alice".into()),
            session_id: None,
            tags: vec![],
            metadata: serde_json::Value::Null,
        });

        engine.add_memory(AddMemoryRequest {
            content: "Bob prefers dogs".into(),
            memory_type: MemoryType::Fact,
            agent_id: None,
            user_id: Some("bob".into()),
            session_id: None,
            tags: vec![],
            metadata: serde_json::Value::Null,
        });

        let results = engine.search(&SearchRequest {
            query: "prefers".into(),
            agent_id: None,
            user_id: Some("alice".into()),
            tags: vec![],
            limit: 10,
            include_graph: false,
        });
        assert_eq!(results.len(), 1);
        assert!(results[0].memory.content.contains("cats"));
    }

    #[test]
    fn test_add_entity_and_relationship() {
        let engine = MemoryEngine::new(test_config());

        let alice = engine.add_entity(AddEntityRequest {
            name: "Alice".into(),
            entity_type: "Person".into(),
            description: None,
            agent_id: None,
            metadata: serde_json::Value::Null,
        });

        let rafttimedb = engine.add_entity(AddEntityRequest {
            name: "RaftTimeDB".into(),
            entity_type: "Project".into(),
            description: Some("Distributed clustering for SpacetimeDB".into()),
            agent_id: None,
            metadata: serde_json::Value::Null,
        });

        let rel = engine.add_relationship(AddRelationshipRequest {
            source_entity_id: alice.id,
            target_entity_id: rafttimedb.id,
            relation_type: "maintains".into(),
            description: None,
            weight: 1.0,
            created_by: "test".into(),
            metadata: serde_json::Value::Null,
        });

        assert_eq!(rel.source_entity_id, alice.id);
        assert_eq!(rel.target_entity_id, rafttimedb.id);

        let rels = engine.get_entity_relationships(alice.id);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].1.name, "RaftTimeDB");
    }

    #[test]
    fn test_graph_traversal() {
        let engine = MemoryEngine::new(test_config());

        let a = engine.add_entity(AddEntityRequest {
            name: "A".into(),
            entity_type: "Node".into(),
            description: None,
            agent_id: None,
            metadata: serde_json::Value::Null,
        });
        let b = engine.add_entity(AddEntityRequest {
            name: "B".into(),
            entity_type: "Node".into(),
            description: None,
            agent_id: None,
            metadata: serde_json::Value::Null,
        });
        let c = engine.add_entity(AddEntityRequest {
            name: "C".into(),
            entity_type: "Node".into(),
            description: None,
            agent_id: None,
            metadata: serde_json::Value::Null,
        });

        engine.add_relationship(AddRelationshipRequest {
            source_entity_id: a.id,
            target_entity_id: b.id,
            relation_type: "connects".into(),
            description: None,
            weight: 1.0,
            created_by: "test".into(),
            metadata: serde_json::Value::Null,
        });
        engine.add_relationship(AddRelationshipRequest {
            source_entity_id: b.id,
            target_entity_id: c.id,
            relation_type: "connects".into(),
            description: None,
            weight: 1.0,
            created_by: "test".into(),
            metadata: serde_json::Value::Null,
        });

        let result = engine.traverse(a.id, 1);
        let names: Vec<&str> = result.iter().map(|(e, _)| e.name.as_str()).collect();
        assert!(names.contains(&"A"));
        assert!(names.contains(&"B"));
        assert!(!names.contains(&"C"));

        let result = engine.traverse(a.id, 2);
        let names: Vec<&str> = result.iter().map(|(e, _)| e.name.as_str()).collect();
        assert!(names.contains(&"A"));
        assert!(names.contains(&"B"));
        assert!(names.contains(&"C"));
    }

    #[test]
    fn test_find_entity_by_name() {
        let engine = MemoryEngine::new(test_config());

        engine.add_entity(AddEntityRequest {
            name: "RaftTimeDB".into(),
            entity_type: "Project".into(),
            description: None,
            agent_id: None,
            metadata: serde_json::Value::Null,
        });

        let found = engine.find_entity_by_name("rafttimedb").unwrap();
        assert_eq!(found.name, "RaftTimeDB");
        assert!(engine.find_entity_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_register_agent() {
        let engine = MemoryEngine::new(test_config());

        let agent = engine.register_agent(RegisterAgentRequest {
            agent_id: "claude-1".into(),
            name: "Claude Worker 1".into(),
            agent_type: "claude-code".into(),
            capabilities: vec!["coding".into(), "research".into()],
            metadata: serde_json::Value::Null,
        });

        assert_eq!(agent.agent_id, "claude-1");
        assert_eq!(agent.status, AgentStatus::Online);

        let agents = engine.list_agents();
        assert_eq!(agents.len(), 1);
    }

    #[test]
    fn test_stats() {
        let engine = MemoryEngine::new(test_config());

        engine.add_memory(AddMemoryRequest {
            content: "Test".into(),
            memory_type: MemoryType::Fact,
            agent_id: None,
            user_id: None,
            session_id: None,
            tags: vec![],
            metadata: serde_json::Value::Null,
        });

        engine.add_entity(AddEntityRequest {
            name: "Test".into(),
            entity_type: "Thing".into(),
            description: None,
            agent_id: None,
            metadata: serde_json::Value::Null,
        });

        let stats = engine.stats();
        assert_eq!(stats["memories"], 1);
        assert_eq!(stats["entities"], 1);
        assert_eq!(stats["valid_memories"], 1);
        assert_eq!(stats["embeddings_indexed"], 0);
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let engine = MemoryEngine::new(test_config());

        engine.add_memory(AddMemoryRequest {
            content: "Snapshot test memory".into(),
            memory_type: MemoryType::Fact,
            agent_id: Some("agent-1".into()),
            user_id: Some("user-1".into()),
            session_id: None,
            tags: vec!["test".into()],
            metadata: serde_json::Value::Null,
        });

        engine.add_entity(AddEntityRequest {
            name: "SnapshotEntity".into(),
            entity_type: "Test".into(),
            description: None,
            agent_id: None,
            metadata: serde_json::Value::Null,
        });

        let snapshot = engine.create_snapshot();
        assert_eq!(snapshot.memories.len(), 1);
        assert_eq!(snapshot.entities.len(), 1);

        // Restore into a new engine
        let mut engine2 = MemoryEngine::new(test_config());
        engine2.restore_from_snapshot(snapshot);

        assert_eq!(engine2.get_memory(1).unwrap().content, "Snapshot test memory");
        assert_eq!(engine2.find_entity_by_name("SnapshotEntity").unwrap().name, "SnapshotEntity");

        // IDs should continue past the restored state
        let new_mem = engine2.add_memory(AddMemoryRequest {
            content: "New memory after restore".into(),
            memory_type: MemoryType::Fact,
            agent_id: None,
            user_id: None,
            session_id: None,
            tags: vec![],
            metadata: serde_json::Value::Null,
        });
        assert!(new_mem.id > 1);
    }
}
