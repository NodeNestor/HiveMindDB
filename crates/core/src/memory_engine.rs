use crate::config::HiveMindConfig;
use crate::types::*;
use chrono::Utc;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::info;

/// Core memory engine — manages memories, entities, relationships, and search.
///
/// Phase 1: In-memory store with full CRUD and basic search.
/// Phase 2+: Backed by SpacetimeDB via RaftTimeDB for distributed replication.
pub struct MemoryEngine {
    config: HiveMindConfig,
    // In-memory stores (Phase 1 — will be replaced by SpacetimeDB subscriptions)
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
}

impl MemoryEngine {
    pub fn new(config: HiveMindConfig) -> Self {
        info!("Initializing memory engine");
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

        let memory = entry.clone();
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
    // Search
    // ========================================================================

    /// Basic keyword search across memories (Phase 1).
    /// Phase 2 adds vector similarity search.
    pub fn search(&self, req: &SearchRequest) -> Vec<SearchResult> {
        let query_lower = req.query.to_lowercase();

        let mut results: Vec<SearchResult> = self
            .memories
            .iter()
            .filter(|entry| {
                let m = entry.value();
                // Filter out invalidated memories
                if m.valid_until.is_some() {
                    return false;
                }
                // Filter by agent_id if specified
                if let Some(ref agent_id) = req.agent_id {
                    if m.agent_id.as_ref() != Some(agent_id) && m.agent_id.is_some() {
                        return false;
                    }
                }
                // Filter by user_id if specified
                if let Some(ref user_id) = req.user_id {
                    if m.user_id.as_ref() != Some(user_id) && m.user_id.is_some() {
                        return false;
                    }
                }
                // Filter by tags if specified
                if !req.tags.is_empty()
                    && !req.tags.iter().any(|t| m.tags.contains(t))
                {
                    return false;
                }
                // Keyword match (Phase 1 — replaced by vector search in Phase 2)
                m.content.to_lowercase().contains(&query_lower)
                    || m.tags.iter().any(|t| t.to_lowercase().contains(&query_lower))
            })
            .map(|entry| {
                let m = entry.value();
                // Simple relevance scoring based on keyword frequency
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
            embedding_model: "test".into(),
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

        // Should not appear in search results
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
        assert_eq!(history.len(), 3); // Add + Update + Invalidate
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

        // Search by tag
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

        // Depth 1: should find A and B
        let result = engine.traverse(a.id, 1);
        let names: Vec<&str> = result.iter().map(|(e, _)| e.name.as_str()).collect();
        assert!(names.contains(&"A"));
        assert!(names.contains(&"B"));
        assert!(!names.contains(&"C"));

        // Depth 2: should find A, B, and C
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
    }
}
