# HiveMindDB — Distributed AI Agent Memory System

## The Pitch

**One sentence:** A distributed, fault-tolerant hivemind memory system for AI agent swarms — replicated knowledge graphs, vector memory, and real-time memory subscriptions, all built on Raft consensus.

**The problem:** Every AI agent memory system today (mem0, Zep, Letta) is single-node. Your agent's memory lives on one machine. That machine dies, the memory is gone. Worse — when you run agent swarms (AgentCore containers, Claude Code teams, multi-agent orchestrations), each agent has its own isolated memory. There's no shared consciousness. No hivemind.

**The solution:** HiveMindDB takes the RaftTimeDB consensus engine and builds a complete agent memory platform on top. Every memory — facts, relationships, episodes, vectors — is replicated across nodes via Raft. Agents subscribe to memory channels and get real-time updates when any agent in the swarm learns something new. Kill a node, the hivemind keeps thinking.

```
Agent 1 (Claude Code)  ──┐
Agent 2 (OpenCode)     ──┤──► HiveMindDB (any node) ──► Raft consensus
Agent 3 (Aider)        ──┤         ↕              ↕              ↕
Agent 4 (custom)       ──┘    HiveMindDB      HiveMindDB      HiveMindDB
                              Node 1           Node 2           Node 3
                          (identical state) (identical state) (identical state)
```

---

## What It Replaces / Unifies

| Current Tool | What HiveMindDB replaces it with | Advantage |
|---|---|---|
| **mem0** | Built-in fact extraction + conflict resolution | Distributed, no single point of failure |
| **Zep/Graphiti** | Temporal knowledge graph with bi-temporal edges | Raft-replicated, real-time subscriptions |
| **Qdrant/Zvec** | Embedded vector store per node | Replicated vectors — search any node, same results |
| **Letta (MemGPT)** | Self-editing memory tools via MCP | Agents edit their own memory, changes replicate everywhere |
| **AgentCore agent-memory** | Drop-in MCP server replacement | Upgrade from markdown files to replicated knowledge graph |
| **AgentNetwork messages** | Shared memory channels with real-time sync | No separate message bus needed for shared state |
| **Redis (AgentCompanyEngineer)** | Raft-replicated state with persistence | No Redis dependency, built-in durability |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        HiveMindDB Node                           │
│                                                                   │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │  MCP Server      │  │  REST/WS API     │  │  SDK clients   │  │
│  │  (Claude Code,   │  │  (any HTTP       │  │  (Python, TS,  │  │
│  │   OpenCode,      │  │   client)        │  │   Rust)        │  │
│  │   Aider)         │  │                  │  │                │  │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬────────┘  │
│           │                     │                     │           │
│  ┌────────▼─────────────────────▼─────────────────────▼────────┐ │
│  │                    Memory Engine                              │ │
│  │                                                              │ │
│  │  ┌──────────┐  ┌──────────────┐  ┌────────────┐            │ │
│  │  │ Semantic  │  │  Knowledge   │  │  Episodic  │            │ │
│  │  │ Memory    │  │  Graph       │  │  Memory    │            │ │
│  │  │ (vectors) │  │  (entities + │  │ (sessions, │            │ │
│  │  │           │  │   relations) │  │  events)   │            │ │
│  │  └──────────┘  └──────────────┘  └────────────┘            │ │
│  │                                                              │ │
│  │  ┌──────────────────────────────────────────────────┐       │ │
│  │  │  Extraction Pipeline (LLM-powered)               │       │ │
│  │  │  conversations → facts → entities → relationships │       │ │
│  │  └──────────────────────────────────────────────────┘       │ │
│  │                                                              │ │
│  │  ┌──────────────────────────────────────────────────┐       │ │
│  │  │  Conflict Resolution                              │       │ │
│  │  │  ADD / UPDATE / INVALIDATE / NOOP                 │       │ │
│  │  │  (temporal — old facts aren't deleted, just aged)  │       │ │
│  │  └──────────────────────────────────────────────────┘       │ │
│  └──────────────────────────┬───────────────────────────────────┘ │
│                             │                                     │
│  ┌──────────────────────────▼───────────────────────────────────┐ │
│  │                  SpacetimeDB Module                           │ │
│  │                                                              │ │
│  │  Tables:                    Reducers:                        │ │
│  │   memories                   add_memory()                    │ │
│  │   entities                   extract_from_conversation()     │ │
│  │   relationships              search_semantic()               │ │
│  │   episodes                   search_graph()                  │ │
│  │   embeddings                 update_memory()                 │ │
│  │   channels                   invalidate_memory()             │ │
│  │   channel_subscriptions      subscribe_channel()             │ │
│  │   agents                     agent_register()                │ │
│  │   memory_history             share_to_channel()              │ │
│  │                                                              │ │
│  └──────────────────────────┬───────────────────────────────────┘ │
│                             │                                     │
│  ┌──────────────────────────▼───────────────────────────────────┐ │
│  │              RaftTimeDB Consensus Layer                       │ │
│  │  (multi-shard: agents shard, graph shard, vector shard)      │ │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
          ↕ Raft replication ↕
    ┌──────────────┐    ┌──────────────┐
    │  HiveMindDB  │    │  HiveMindDB  │
    │  Node 2      │    │  Node 3      │
    └──────────────┘    └──────────────┘
```

---

## Data Model

### Core Tables (SpacetimeDB module)

```rust
// === MEMORIES ===

#[spacetimedb::table(name = memories, public)]
pub struct Memory {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub content: String,              // The actual memory text
    pub memory_type: MemoryType,      // Fact, Episodic, Procedural, Semantic
    pub agent_id: Option<String>,     // Owning agent (None = shared)
    pub user_id: Option<String>,      // Associated user
    pub session_id: Option<String>,   // Associated session
    pub confidence: f32,              // 0.0-1.0, decays over time
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub valid_from: Timestamp,        // Bi-temporal: when fact became true
    pub valid_until: Option<Timestamp>, // Bi-temporal: when fact stopped being true (None = still valid)
    pub source: String,               // What produced this memory (conversation, extraction, manual)
    pub metadata: String,             // JSON blob for extensible metadata
}

#[spacetimedb::table(name = memory_history, public)]
pub struct MemoryHistory {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub memory_id: u64,
    pub operation: Operation,         // Add, Update, Invalidate, Merge
    pub old_content: Option<String>,
    pub new_content: String,
    pub reason: String,               // Why the change happened
    pub changed_by: String,           // Agent that made the change
    pub timestamp: Timestamp,
}

// === KNOWLEDGE GRAPH ===

#[spacetimedb::table(name = entities, public)]
pub struct Entity {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub name: String,                 // "Alice", "RaftTimeDB", "Rust"
    pub entity_type: String,          // "Person", "Project", "Technology", "Concept"
    pub description: Option<String>,
    pub agent_id: Option<String>,     // Discoverer
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub metadata: String,             // JSON
}

#[spacetimedb::table(name = relationships, public)]
pub struct Relationship {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub source_entity_id: u64,
    pub target_entity_id: u64,
    pub relation_type: String,        // "works_on", "prefers", "knows", "depends_on"
    pub description: Option<String>,
    pub weight: f32,                  // Strength of relationship
    pub valid_from: Timestamp,        // Bi-temporal
    pub valid_until: Option<Timestamp>,
    pub created_by: String,
    pub metadata: String,
}

// === VECTOR EMBEDDINGS ===

#[spacetimedb::table(name = embeddings, public)]
pub struct Embedding {
    #[primary_key]
    pub memory_id: u64,               // 1:1 with Memory
    pub vector: Vec<f32>,             // 1536-dim (OpenAI) or 1024-dim (configurable)
    pub model: String,                // "text-embedding-3-small", "nomic-embed-text", etc.
}

// === EPISODES (session-level memory) ===

#[spacetimedb::table(name = episodes, public)]
pub struct Episode {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub agent_id: String,
    pub user_id: Option<String>,
    pub session_id: String,
    pub summary: String,              // LLM-generated session summary
    pub key_decisions: String,        // JSON array of key decisions made
    pub tools_used: String,           // JSON array of tools invoked
    pub outcome: String,              // What was accomplished
    pub started_at: Timestamp,
    pub ended_at: Timestamp,
    pub metadata: String,
}

// === CHANNELS (hivemind sharing) ===

#[spacetimedb::table(name = channels, public)]
pub struct Channel {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub name: String,                 // "project:rafttimedb", "team:backend", "global"
    pub description: Option<String>,
    pub channel_type: ChannelType,    // Public, Private, Agent, User
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
    pub shared_by: String,            // Agent that shared it
    pub shared_at: Timestamp,
}

#[spacetimedb::table(name = channel_subscriptions, public)]
pub struct ChannelSubscription {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub channel_id: u64,
    pub agent_id: String,
    pub subscribed_at: Timestamp,
}

// === AGENTS (registry) ===

#[spacetimedb::table(name = agents, public)]
pub struct Agent {
    #[primary_key]
    pub agent_id: String,             // Unique agent identifier
    pub name: String,                 // Human-readable name
    pub agent_type: String,           // "claude-code", "opencode", "aider", "custom"
    pub capabilities: String,         // JSON array of capabilities
    pub status: AgentStatus,          // Online, Offline, Busy
    pub last_seen: Timestamp,
    pub memory_count: u64,            // How many memories this agent has
    pub metadata: String,
}
```

### Sharding Strategy

| Shard | Contains | Why separate |
|---|---|---|
| **shard 0** | agents, channels, subscriptions, config | Low write volume, high read — cluster metadata |
| **shard 1** | memories, memory_history, episodes | High write volume — main memory pipeline |
| **shard 2** | entities, relationships | Graph mutations — independent write path |
| **shard 3** | embeddings | Large payloads — vector storage separated from text |

Agents writing memories don't block graph updates. Vector indexing doesn't block fact extraction. Each shard has its own Raft leader — true parallel write throughput.

---

## Agent-Facing API

### MCP Server (Primary Interface)

This is how Claude Code, OpenCode, Aider, and any MCP-compatible agent talks to HiveMindDB. Drop-in replacement for the current `agent-memory` MCP server in AgentCore.

```
MCP Tools:
  ─────────── Core Memory ───────────
  memory_add          Store a memory (auto-extracts entities + embeddings)
  memory_search       Semantic search across all memories
  memory_recall       Get memories by topic/tag/agent/user
  memory_update       Update an existing memory
  memory_forget       Temporally invalidate a memory (doesn't delete — marks as expired)
  memory_history      Full audit trail for a memory

  ─────────── Conversations ───────────
  memory_extract      Run extraction pipeline on a conversation (bulk add)
  memory_summarize    Generate episode summary for current session

  ─────────── Knowledge Graph ───────────
  graph_add_entity    Create or update an entity
  graph_add_relation  Create a relationship between entities
  graph_query         Natural language graph query ("what does Alice work on?")
  graph_traverse      Multi-hop traversal from an entity
  graph_visualize     Return graph structure as DOT/JSON for rendering

  ─────────── Hivemind Channels ───────────
  channel_create      Create a shared memory channel
  channel_subscribe   Subscribe to a channel (get real-time updates)
  channel_share       Share a memory to a channel
  channel_feed        Get recent memories from subscribed channels

  ─────────── Agent Identity ───────────
  agent_register      Register this agent with the hivemind
  agent_status        Check status of other agents
  agent_whoami        Get this agent's memory stats and identity
```

**Example usage in Claude Code:**

```
> Remember that the user prefers Rust over Python for new projects.

Agent calls: memory_add(
  content: "User prefers Rust over Python for new projects",
  user_id: "ludde",
  memory_type: "fact",
  metadata: { category: "preferences", topic: "programming_languages" }
)

→ HiveMindDB extracts entities: [User:ludde, Language:Rust, Language:Python]
→ Creates relationships: [ludde --prefers--> Rust], [ludde --avoids--> Python]
→ Generates embedding, stores vector
→ Replicates via Raft to all nodes
→ Pushes to "user:ludde" channel
→ All other agents subscribed to that channel get notified
```

```
> What does this user like?

Agent calls: memory_search(
  query: "user preferences and likes",
  user_id: "ludde",
  limit: 10
)

→ Semantic search across memory vectors
→ Graph traversal from entity "ludde" (outgoing "prefers" edges)
→ Merged, ranked results returned
→ Agent sees: "Prefers Rust over Python", "Prefers Italian food", etc.
```

### REST API

For non-MCP agents, scripts, dashboards, and external integrations.

```
# Memory CRUD
POST   /api/v1/memories                    Add memory
GET    /api/v1/memories/:id                Get memory
PUT    /api/v1/memories/:id                Update memory
DELETE /api/v1/memories/:id                Invalidate memory
GET    /api/v1/memories/:id/history        Audit trail

# Search
POST   /api/v1/search                     Semantic search
POST   /api/v1/search/graph               Graph-aware search
POST   /api/v1/search/hybrid              Combined vector + graph + keyword

# Extraction
POST   /api/v1/extract                    Extract memories from conversation
POST   /api/v1/summarize                  Summarize a session into an episode

# Knowledge Graph
POST   /api/v1/entities                   Create entity
GET    /api/v1/entities/:id               Get entity + relationships
POST   /api/v1/relationships              Create relationship
POST   /api/v1/graph/query                Natural language graph query
POST   /api/v1/graph/traverse             Multi-hop traversal

# Channels
POST   /api/v1/channels                   Create channel
GET    /api/v1/channels/:id/feed          Get channel feed
POST   /api/v1/channels/:id/share         Share memory to channel
WS     /api/v1/channels/:id/subscribe     WebSocket subscription (real-time)

# Agents
POST   /api/v1/agents/register            Register agent
GET    /api/v1/agents                     List all known agents
GET    /api/v1/agents/:id/memories        Get agent's memories

# Admin
GET    /api/v1/status                     Cluster status
GET    /api/v1/metrics                    Prometheus metrics
GET    /health                            Health check
```

### WebSocket Subscriptions

Agents connect via WebSocket and subscribe to channels. When any agent in the swarm creates a memory in that channel, all subscribers get a real-time push.

```
WS /api/v1/ws

→ Client sends: { "type": "subscribe", "channels": ["global", "project:rafttimedb", "user:ludde"] }
← Server pushes: { "type": "memory_added", "channel": "global", "memory": { ... } }
← Server pushes: { "type": "entity_updated", "channel": "project:rafttimedb", "entity": { ... } }
← Server pushes: { "type": "memory_invalidated", "channel": "user:ludde", "memory_id": 42, "reason": "..." }
```

This is how the hivemind works — Agent 1 learns something, Agent 2 knows it 50ms later.

### Python SDK

```python
from hiveminddb import HiveMind

hive = HiveMind("ws://localhost:3001")  # Connect to any node

# Store a memory
hive.add("User prefers dark mode", user_id="ludde", tags=["preferences", "ui"])

# Search
results = hive.search("what does the user prefer?", user_id="ludde")

# Extract from conversation
hive.extract([
    {"role": "user", "content": "I always use neovim, never VS Code"},
    {"role": "assistant", "content": "Noted, you prefer neovim."}
], user_id="ludde")

# Knowledge graph
hive.graph.add_entity("RaftTimeDB", type="Project", description="Distributed clustering for SpacetimeDB")
hive.graph.add_relation("ludde", "maintains", "RaftTimeDB")
hive.graph.query("What projects does ludde maintain?")

# Subscribe to channel updates (async)
async for memory in hive.subscribe("project:rafttimedb"):
    print(f"New memory: {memory.content}")
```

### TypeScript SDK

```typescript
import { HiveMind } from 'hiveminddb';

const hive = new HiveMind('ws://localhost:3001');

await hive.add('User prefers TypeScript for frontend', { userId: 'ludde' });

const results = await hive.search('frontend preferences', { userId: 'ludde' });

// Real-time channel subscription
hive.subscribe('project:rafttimedb', (memory) => {
  console.log(`New memory: ${memory.content}`);
});
```

---

## Integration with AgentCore

### Drop-in MCP Replacement

Add to AgentCore's `mcp-tools/library.json`:

```json
{
  "hivemind": {
    "name": "HiveMind Memory",
    "description": "Distributed AI agent memory — replicated knowledge graph, vector search, hivemind channels",
    "command": "npx",
    "args": ["hiveminddb-mcp", "--url", "${HIVEMIND_URL}"],
    "builtIn": false,
    "category": "memory",
    "default": false,
    "requiredEnv": ["HIVEMIND_URL"]
  }
}
```

Add to `.env`:
```bash
HIVEMIND_URL=ws://hivemind-node-1:3001
MEMORY_PROVIDER=hivemind   # Replaces local/mem0/qdrant
```

### Docker Compose (Swarm with HiveMindDB)

```yaml
# docker-compose.hivemind.yml
version: "3.8"

services:
  # === HiveMindDB Cluster (3 nodes) ===
  hivemind-1:
    image: hiveminddb:latest
    environment:
      RTDB_NODE_ID: 1
      RTDB_RAFT_ADDR: "0.0.0.0:4001"
      RTDB_LISTEN_ADDR: "0.0.0.0:3001"
      RTDB_STDB_URL: "ws://stdb-1:3000"
      RTDB_PEERS: "2=hivemind-2:4001,3=hivemind-3:4001"
      HIVEMIND_LLM_PROVIDER: "anthropic"         # For extraction pipeline
      HIVEMIND_EMBEDDING_MODEL: "text-embedding-3-small"
    ports:
      - "3001:3001"   # Agent WebSocket
      - "4001:4001"   # Raft management

  hivemind-2:
    image: hiveminddb:latest
    environment:
      RTDB_NODE_ID: 2
      RTDB_RAFT_ADDR: "0.0.0.0:4001"
      RTDB_LISTEN_ADDR: "0.0.0.0:3001"
      RTDB_STDB_URL: "ws://stdb-2:3000"
      RTDB_PEERS: "1=hivemind-1:4001,3=hivemind-3:4001"

  hivemind-3:
    image: hiveminddb:latest
    environment:
      RTDB_NODE_ID: 3
      RTDB_RAFT_ADDR: "0.0.0.0:4001"
      RTDB_LISTEN_ADDR: "0.0.0.0:3001"
      RTDB_STDB_URL: "ws://stdb-3:3000"
      RTDB_PEERS: "1=hivemind-1:4001,2=hivemind-2:4001"

  # === SpacetimeDB Instances ===
  stdb-1:
    image: clockworklabs/spacetime:latest
    command: ["spacetimedb-standalone", "start", "--listen-addr", "0.0.0.0:3000"]
  stdb-2:
    image: clockworklabs/spacetime:latest
    command: ["spacetimedb-standalone", "start", "--listen-addr", "0.0.0.0:3000"]
  stdb-3:
    image: clockworklabs/spacetime:latest
    command: ["spacetimedb-standalone", "start", "--listen-addr", "0.0.0.0:3000"]

  # === Agent Swarm ===
  agent-lead:
    image: agentcore:ubuntu
    environment:
      AGENT_TYPE: claude
      AGENT_ID: lead
      AGENT_ROLE: lead
      HIVEMIND_URL: "ws://hivemind-1:3001"
      MEMORY_PROVIDER: hivemind
    depends_on: [hivemind-1]

  agent-worker-1:
    image: agentcore:minimal
    environment:
      AGENT_TYPE: claude
      AGENT_ID: worker-1
      AGENT_ROLE: backend
      HIVEMIND_URL: "ws://hivemind-2:3001"
      MEMORY_PROVIDER: hivemind
    depends_on: [hivemind-2]

  agent-worker-2:
    image: agentcore:minimal
    environment:
      AGENT_TYPE: opencode
      AGENT_ID: worker-2
      AGENT_ROLE: frontend
      HIVEMIND_URL: "ws://hivemind-3:3001"
      MEMORY_PROVIDER: hivemind
    depends_on: [hivemind-3]
```

Each agent connects to a different HiveMindDB node (load distributed), but they all share the same replicated memory. Agent-lead learns something → worker-1 and worker-2 know it within milliseconds.

### Integration with AgentNetwork

HiveMindDB doesn't replace AgentNetwork's real-time chat — it augments it with persistent, searchable, structured memory. AgentNetwork handles transient communication (chat, DMs, file sharing). HiveMindDB handles persistent knowledge (facts, relationships, preferences, learned patterns).

The AgentNetwork MCP plugin could be extended to auto-extract memories from agent chat messages and push them into HiveMindDB.

---

## Repo Structure

```
hiveminddb/
├── Cargo.toml                          # Workspace root
├── README.md
├── CLAUDE.md
├── LICENSE                             # Apache 2.0
│
├── crates/
│   ├── core/                           # HiveMindDB server binary
│   │   └── src/
│   │       ├── main.rs                 # Entry point — extends RaftTimeDB
│   │       ├── api.rs                  # REST + WebSocket API layer
│   │       ├── memory_engine.rs        # Memory CRUD, search, conflict resolution
│   │       ├── extraction.rs           # LLM-powered fact/entity/relation extraction
│   │       ├── graph.rs                # Knowledge graph queries (traversal, natural language)
│   │       ├── vector.rs               # Embedding generation + vector similarity search
│   │       ├── channels.rs             # Hivemind channel pub/sub
│   │       ├── temporal.rs             # Bi-temporal logic, decay, validity windows
│   │       └── config.rs              # HiveMindDB-specific config
│   │
│   ├── mcp-server/                     # MCP server (Node.js/TypeScript)
│   │   ├── package.json
│   │   └── src/
│   │       ├── index.ts                # MCP protocol handler
│   │       ├── tools.ts                # Tool definitions (memory_add, graph_query, etc.)
│   │       └── client.ts               # HiveMindDB WebSocket client
│   │
│   ├── sdk-python/                     # Python SDK
│   │   ├── pyproject.toml
│   │   └── hiveminddb/
│   │       ├── __init__.py
│   │       ├── client.py               # HiveMind class
│   │       ├── graph.py                # Graph query builder
│   │       └── types.py                # Memory, Entity, Relationship types
│   │
│   ├── sdk-typescript/                 # TypeScript SDK
│   │   ├── package.json
│   │   └── src/
│   │       ├── index.ts
│   │       ├── client.ts
│   │       └── types.ts
│   │
│   └── cli/                            # CLI tool
│       └── src/
│           └── main.rs                 # hmdb CLI: status, search, graph, channels
│
├── module/                             # SpacetimeDB WASM module
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                      # Module entry point
│       ├── tables.rs                   # All table definitions
│       ├── reducers/
│       │   ├── memory.rs               # add, update, invalidate, search
│       │   ├── graph.rs                # entity CRUD, relationship CRUD
│       │   ├── channels.rs             # channel create, subscribe, share
│       │   ├── episodes.rs             # session summarization
│       │   └── agents.rs               # agent registration, status
│       └── indexes.rs                  # BTree + vector indexes
│
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile                  # Multi-stage build (core + module)
│   │   ├── docker-compose.yml          # Standalone 3-node cluster
│   │   └── docker-compose.swarm.yml    # Full swarm: HiveMindDB + AgentCore agents
│   └── agentcore/
│       ├── library-patch.json          # HiveMindDB entry for AgentCore's library.json
│       └── env.example                 # AgentCore env vars for HiveMindDB
│
└── tests/
    ├── unit/                           # Rust unit tests
    ├── integration/                    # Multi-node integration tests
    └── e2e/
        └── swarm_test.sh              # Full swarm E2E: agents + hivemind + extraction
```

---

## How It Builds on RaftTimeDB

HiveMindDB is **not a fork** of RaftTimeDB. It's a separate project that depends on RaftTimeDB as a library/binary:

```toml
# crates/core/Cargo.toml
[dependencies]
rafttimedb = { git = "https://github.com/NodeNestor/RaftimeDB.git" }
# OR: uses RaftTimeDB binary + SpacetimeDB module approach
```

**Two possible build strategies:**

### Option A: SpacetimeDB Module (Simpler)

RaftTimeDB runs unchanged. HiveMindDB is a SpacetimeDB WASM module that gets published to each node. The memory engine (extraction, graph queries, vector search) runs as a sidecar service that calls reducers through the WebSocket proxy.

```
Agent → MCP Server → HiveMindDB API (sidecar) → RaftTimeDB (ws://localhost:3001) → SpacetimeDB
```

**Pros:** RaftTimeDB stays generic. SpacetimeDB handles all storage. Module is just WASM.
**Cons:** Vector search in WASM is limited. Extraction pipeline needs external LLM calls.

### Option B: Extended Binary (More Powerful)

HiveMindDB extends the RaftTimeDB binary with additional HTTP/WS endpoints and an embedded vector index (like Zvec). The SpacetimeDB module handles structured data (entities, relationships, episodes), while the sidecar handles embeddings and LLM calls.

```
Agent → HiveMindDB API → [vector search (embedded)] + [SpacetimeDB (via RaftTimeDB)]
```

**Pros:** Full control over vector search. No WASM limitations. Single binary.
**Cons:** More complex build. Tighter coupling.

**Recommendation: Start with Option A** (module + sidecar). It's simpler, keeps concerns separated, and we can always optimize later. The sidecar pattern also means the extraction pipeline can use whatever LLM provider the user configures.

---

## Memory Pipeline (How Extraction Works)

```
┌────────────────────────────────────────────────────────────────────┐
│                    Memory Extraction Pipeline                       │
│                                                                    │
│  1. Agent sends conversation to memory_extract()                   │
│     ┌─────────────────────────────────────┐                       │
│     │ "I always use neovim for Rust and   │                       │
│     │  VS Code for TypeScript projects"   │                       │
│     └────────────────┬────────────────────┘                       │
│                      ▼                                             │
│  2. LLM Fact Extraction (configurable provider)                    │
│     ┌─────────────────────────────────────┐                       │
│     │ Facts:                              │                       │
│     │  - "User uses neovim for Rust"      │                       │
│     │  - "User uses VS Code for TS"       │                       │
│     └────────────────┬────────────────────┘                       │
│                      ▼                                             │
│  3. Entity + Relationship Extraction                               │
│     ┌─────────────────────────────────────┐                       │
│     │ Entities: [User, Neovim, VS Code,   │                       │
│     │           Rust, TypeScript]          │                       │
│     │ Relations:                           │                       │
│     │  User --uses_for_rust--> Neovim     │                       │
│     │  User --uses_for_ts--> VS Code      │                       │
│     └────────────────┬────────────────────┘                       │
│                      ▼                                             │
│  4. Conflict Resolution (against existing memories)                │
│     ┌─────────────────────────────────────┐                       │
│     │ Existing: "User uses VS Code"       │                       │
│     │ New: "User uses neovim for Rust"    │                       │
│     │ Decision: UPDATE (more specific)    │                       │
│     │                                     │                       │
│     │ Old "User uses VS Code" → narrowed  │                       │
│     │ to "User uses VS Code for TS"       │                       │
│     └────────────────┬────────────────────┘                       │
│                      ▼                                             │
│  5. Embedding Generation                                           │
│     ┌─────────────────────────────────────┐                       │
│     │ Each fact → 1536-dim vector         │                       │
│     │ (text-embedding-3-small or local)   │                       │
│     └────────────────┬────────────────────┘                       │
│                      ▼                                             │
│  6. Raft Commit (replicate to all nodes)                           │
│     ┌─────────────────────────────────────┐                       │
│     │ CallReducer: add_memory(...)        │                       │
│     │ CallReducer: add_entity(...)        │                       │
│     │ CallReducer: add_relationship(...)  │                       │
│     │ → Raft consensus → all nodes        │                       │
│     └────────────────┬────────────────────┘                       │
│                      ▼                                             │
│  7. Channel Broadcast                                              │
│     ┌─────────────────────────────────────┐                       │
│     │ Push to: "user:ludde", "global"     │                       │
│     │ All subscribed agents notified      │                       │
│     └─────────────────────────────────────┘                       │
└────────────────────────────────────────────────────────────────────┘
```

---

## What Makes This Different from Just Using mem0

| | mem0 | HiveMindDB |
|---|---|---|
| **Fault tolerance** | Single process | 3+ node Raft cluster, survives node failures |
| **Multi-agent** | Scoped isolation only | Real-time hivemind channels — agents share consciousness |
| **Knowledge graph** | Optional Neo4j addon | Built-in, Raft-replicated, bi-temporal |
| **Real-time updates** | Poll-based | WebSocket subscriptions — learn in milliseconds |
| **Agent framework** | Python SDK | MCP server (works with Claude Code, OpenCode, Aider natively) |
| **Self-hosting** | Requires Qdrant + Neo4j + LLM | Single binary + SpacetimeDB (or Docker Compose) |
| **Consistency** | Eventually consistent | Raft linearizable consistency — all nodes identical |
| **Storage** | External DBs | SpacetimeDB (deterministic WASM) — one source of truth |
| **Agent identity** | user_id/agent_id labels | Full agent registry with capabilities, status, memory stats |
| **History** | Mutation log | Full bi-temporal history — query "what did we know last Tuesday?" |

---

## CLI Tool

```bash
# Cluster
hmdb status                                    # Cluster health + memory stats
hmdb init --nodes 1=node1:4001 ...             # Bootstrap

# Memory
hmdb search "what does the user prefer?"       # Semantic search
hmdb add "User prefers dark mode" --user ludde # Manual memory add
hmdb history 42                                 # Audit trail for memory #42
hmdb forget 42 --reason "outdated preference"  # Temporally invalidate

# Graph
hmdb graph query "what projects does ludde maintain?"
hmdb graph entity "RaftTimeDB"                 # Show entity + all relationships
hmdb graph traverse "ludde" --depth 3          # 3-hop traversal
hmdb graph export --format dot > graph.dot     # Export for visualization

# Channels
hmdb channels list                              # All channels
hmdb channels feed "project:rafttimedb"        # Recent memories in channel
hmdb channels subscribe "global"               # Live stream

# Agents
hmdb agents list                                # All registered agents
hmdb agents memories "worker-1"                # Agent's memories
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] New repo scaffold (workspace, crates, module directory)
- [ ] SpacetimeDB module with core tables (memories, entities, relationships, embeddings)
- [ ] Basic reducers (add_memory, search, add_entity, add_relationship)
- [ ] MCP server with memory_add, memory_search, memory_recall
- [ ] Docker Compose with RaftTimeDB + SpacetimeDB + HiveMindDB sidecar
- [ ] Verify Raft replication of memories across 3 nodes

### Phase 2: Extraction Pipeline (Week 3-4)
- [ ] LLM-powered fact extraction (configurable provider: Anthropic, OpenAI, Ollama)
- [ ] Entity + relationship extraction from conversations
- [ ] Conflict resolution (ADD/UPDATE/INVALIDATE/NOOP)
- [ ] Embedding generation (OpenAI, Ollama/nomic-embed-text, or local)
- [ ] Vector similarity search in sidecar
- [ ] memory_extract MCP tool

### Phase 3: Knowledge Graph (Week 5-6)
- [ ] Natural language graph queries
- [ ] Multi-hop graph traversal
- [ ] Bi-temporal validity tracking on relationships
- [ ] Graph-enhanced search (vector + graph combined)
- [ ] graph_query, graph_traverse, graph_visualize MCP tools

### Phase 4: Hivemind Channels (Week 7-8)
- [ ] Channel creation and subscription
- [ ] Real-time WebSocket push for new memories
- [ ] Channel-scoped search
- [ ] Auto-sharing rules (e.g., "all facts about project X go to channel project:X")
- [ ] channel_create, channel_subscribe, channel_share, channel_feed MCP tools

### Phase 5: Agent Identity + AgentCore Integration (Week 9-10)
- [ ] Agent registry (register, heartbeat, capabilities)
- [ ] AgentCore library.json integration
- [ ] Docker Compose swarm template (HiveMindDB + AgentCore agents)
- [ ] Python SDK (pip install hiveminddb)
- [ ] TypeScript SDK (npm install hiveminddb)
- [ ] CLI tool (hmdb)

### Phase 6: Advanced Features (Week 11+)
- [ ] Memory decay (confidence decreases over time, old memories fade)
- [ ] Episodic memory (session summaries, "what happened last time?")
- [ ] Procedural memory (how an agent solved a problem — replayable steps)
- [ ] Community detection (cluster related entities automatically)
- [ ] Memory importance ranking (not all memories are equal)
- [ ] Cross-agent memory merging (when two agents learn the same thing independently)

---

## Quick Start (What It Looks Like When Done)

### For a single agent (Claude Code)

```bash
# Start HiveMindDB (single node for dev)
docker compose -f deploy/docker/docker-compose.yml up -d

# Add MCP server to Claude Code
claude mcp add hivemind npx hiveminddb-mcp --url ws://localhost:3001

# Done. Claude Code now has persistent, searchable memory.
```

### For an agent swarm (AgentCore)

```bash
# Start everything
docker compose -f deploy/docker/docker-compose.swarm.yml up -d

# Bootstrap the cluster
hmdb init --nodes 1=hivemind-1:4001 2=hivemind-2:4001 3=hivemind-3:4001

# Agents auto-connect via HIVEMIND_URL env var. They share memory immediately.
```

### For any HTTP client

```bash
# Store a memory
curl -X POST http://localhost:8080/api/v1/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "The deploy pipeline uses GitHub Actions", "agent_id": "lead", "tags": ["devops"]}'

# Search
curl -X POST http://localhost:8080/api/v1/search \
  -d '{"query": "how do we deploy?", "limit": 5}'
```

---

## Open Questions

1. **Embedding provider**: Ship with local embeddings (e.g., candle + ONNX model) or require external API? Local = zero dependencies but larger binary. External = smaller but needs API key.

2. **LLM for extraction**: Same question. The extraction pipeline needs an LLM. Options:
   - Use the agent's own LLM (route through CodeGate)
   - Require ANTHROPIC_API_KEY / OPENAI_API_KEY
   - Ship with a small local model (heavy)
   - Make it pluggable (recommended)

3. **Vector search backend**: Embedded (Zvec-style, in the sidecar) vs SpacetimeDB table scan vs external (Qdrant)? Embedded is simplest for users.

4. **Repo name**: `hiveminddb`? `raftmind`? `swarm-memory`? `hive`?
