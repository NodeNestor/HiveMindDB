<div align="center">

# HiveMindDB

### Shared memory for AI agent swarms.

Distributed, fault-tolerant memory system for AI agents — knowledge graphs, semantic search, LLM extraction, real-time hivemind channels, all replicated via Raft consensus.

[![CI](https://github.com/NodeNestor/HiveMindDB/actions/workflows/ci.yml/badge.svg)](https://github.com/NodeNestor/HiveMindDB/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85+-orange.svg)](https://www.rust-lang.org)

[Quick Start](#quick-start) | [Features](#features) | [Architecture](#architecture) | [API](#api) | [Contributing](CONTRIBUTING.md)

</div>

---

Your AI agents forget everything between sessions. When you run agent swarms, each agent has its own isolated memory. There's no shared consciousness.

**HiveMindDB fixes that.** It gives your agents persistent, replicated, shared memory — built on [RaftTimeDB](https://github.com/NodeNestor/RaftimeDB) for fault tolerance.

```
Agent 1 learns something ──► HiveMindDB ──► Raft consensus
                                                   ↓
Agent 2 knows it instantly ◄── real-time push ◄── all nodes
Agent 3 knows it instantly ◄── real-time push ◄── identical
```

## Quick Start

### Option 1: Docker Compose (easiest)

Spins up the full stack locally — SpacetimeDB + RaftTimeDB + HiveMindDB sidecars.

```bash
git clone https://github.com/NodeNestor/HiveMindDB.git
cd HiveMindDB/deploy/docker
docker compose up -d
```

Connect your agent to `http://localhost:8100`.

### Option 2: Pre-built Binary

Download from [Releases](https://github.com/NodeNestor/HiveMindDB/releases) for your platform:

| Platform | Binary |
|----------|--------|
| Linux x86_64 | `hiveminddb-x86_64-unknown-linux-gnu` |
| Windows x86_64 | `hiveminddb-x86_64-pc-windows-msvc.exe` |
| macOS ARM | `hiveminddb-aarch64-apple-darwin` |

```bash
hiveminddb --listen-addr 0.0.0.0:8100
```

### Option 3: Build from Source

```bash
cargo install --path crates/core    # hiveminddb server
cargo install --path crates/cli     # hmdb CLI
```

### With Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add hivemind -- npx hiveminddb-mcp --url http://localhost:8100

# Done. Claude Code now has persistent, searchable memory.
```

### With AgentCore

```bash
# Add to your .env
HIVEMINDDB_URL=http://hivemind-1:8100
MEMORY_PROVIDER=hiveminddb
```

### With CodeGate Proxy

```bash
# Use CodeGate as the LLM provider for extraction
HIVEMIND_LLM_PROVIDER=codegate  # Uses http://localhost:9212/v1
# Or point at your CodeGate instance directly:
HIVEMIND_LLM_PROVIDER=http://codegate:9212/v1
```

### What Your Agent Can Do

```
> Remember that the user prefers Rust over Python for new projects.
✓ Stored memory #1 under topic "preferences"

> What does the user prefer?
Found 1 result:
  #1 [score: 0.95] User prefers Rust over Python for new projects
     tags: preferences

> Extract knowledge from this conversation
Added 3 memories:
  #2: User is building RaftTimeDB
  #3: User prefers dark mode
  #4: User works with SpacetimeDB
Added 2 entities:
  RaftTimeDB (Project)
  SpacetimeDB (Technology)
Added 1 relationship:
  RaftTimeDB --uses--> SpacetimeDB

> Who maintains RaftTimeDB?
Entity: ludde (Person)
  --maintains--> RaftTimeDB (Project)
  --prefers--> Rust (Language)
```

## Features

| Feature | Description |
|---------|-------------|
| **Persistent Memory** | Facts, preferences, and knowledge survive across sessions |
| **Knowledge Graph** | Entities + typed relationships with graph traversal |
| **Hybrid Search** | Keyword + vector similarity search (OpenAI/Ollama/CodeGate embeddings) |
| **LLM Extraction** | Automatically extract facts, entities, and relationships from conversations |
| **Bi-Temporal** | Old facts are invalidated, not deleted — query "what did we know last Tuesday?" |
| **Hivemind Channels** | Agents subscribe to channels, get real-time WebSocket updates |
| **Conflict Resolution** | LLM determines ADD/UPDATE/NOOP for new facts vs existing knowledge |
| **Full Audit Trail** | Every memory change is recorded — who changed what, when, and why |
| **Snapshot Persistence** | Periodic JSON snapshots to disk, auto-restore on restart |
| **Raft Replication** | Optional RaftTimeDB replication for multi-node fault tolerance |
| **MCP Native** | Drop-in MCP server for Claude Code, OpenCode, Aider (20 tools) |
| **AgentCore Compatible** | Same `remember`/`recall`/`forget`/`search` interface |
| **CodeGate Support** | Use your CodeGate proxy for LLM and embedding calls |
| **REST + WebSocket API** | Works with any HTTP client or agent framework |
| **Graceful Shutdown** | Ctrl+C saves final snapshot, drains connections cleanly |

## Architecture

```
┌─────────────────────────────────────────────┐
│            Your AI Agent Swarm              │
│  Claude Code  ·  OpenCode  ·  Aider  ·  …  │
└──────────────────┬──────────────────────────┘
                   │ MCP / REST / WebSocket
┌──────────────────▼──────────────────────────┐
│           HiveMindDB Sidecar                │
│  Memory Engine · Knowledge Graph · Channels │
│  LLM Extraction · Vector Embeddings         │
│  Snapshot Persistence · Replication Client   │
└──────────────────┬──────────────────────────┘
                   │ WebSocket (replication)
┌──────────────────▼──────────────────────────┐
│        RaftTimeDB (Raft Consensus)          │
│  Multi-shard · Leader forwarding · TLS      │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│      SpacetimeDB (Deterministic Storage)    │
│  WASM module · Tables · Reducers            │
└─────────────────────────────────────────────┘
```

## CLI

```bash
hmdb status                                    # Cluster stats + embedding/extraction info
hmdb add "User prefers Rust" --user ludde     # Add a memory
hmdb search "what does the user prefer?"      # Hybrid search
hmdb extract "User said they prefer Rust"     # LLM extraction
hmdb extract --file conversation.json          # Extract from conversation file
hmdb entity "RaftTimeDB"                       # Entity + relationships
hmdb traverse 1 --depth 3                      # Graph traversal
hmdb history 42                                # Audit trail
hmdb forget 42 --reason "outdated"            # Invalidate
hmdb channels                                  # List channels
hmdb agents                                    # List agents
```

## MCP Tools (20 tools)

### AgentCore-Compatible (drop-in replacement)

| Tool | Description |
|------|-------------|
| `remember` | Store memory under a topic |
| `recall` | Recall all memories for a topic |
| `forget` | Invalidate all memories for a topic |
| `search` | Hybrid search (keyword + vector) |
| `list_topics` | List topics with counts |

### Extended HiveMindDB Tools

| Tool | Description |
|------|-------------|
| `memory_add` | Add memory with full metadata |
| `memory_search` | Hybrid search with filters |
| `memory_history` | Full audit trail |
| `extract` | LLM knowledge extraction from conversation |
| `graph_add_entity` | Add knowledge graph entity |
| `graph_add_relation` | Create entity relationship |
| `graph_query` | Find entity + relationships |
| `graph_traverse` | Graph traversal from entity |
| `channel_create` | Create hivemind channel |
| `channel_share` | Share memory to channel |
| `channel_list` | List all channels |
| `agent_register` | Register agent in hivemind |
| `agent_status` | List agents + status |
| `hivemind_status` | Full cluster status |

## API

Full REST API at `http://localhost:8100/api/v1/`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/memories` | POST/GET | Add/list memories |
| `/memories/:id` | GET/PUT/DELETE | Get, update, invalidate |
| `/memories/:id/history` | GET | Audit trail |
| `/search` | POST | Hybrid search (keyword + vector) |
| `/extract` | POST | LLM knowledge extraction |
| `/entities` | POST | Add entity |
| `/entities/:id` | GET | Get entity |
| `/entities/find` | POST | Find by name |
| `/entities/:id/relationships` | GET | Entity relationships |
| `/relationships` | POST | Add relationship |
| `/graph/traverse` | POST | Graph traversal |
| `/channels` | POST/GET | Create/list channels |
| `/channels/:id/share` | POST | Share memory to channel |
| `/agents/register` | POST | Register agent |
| `/agents` | GET | List agents |
| `/agents/:id/heartbeat` | POST | Agent heartbeat |
| `/status` | GET | Cluster stats |

### Request Body Examples

**POST /api/v1/relationships**
```json
{
  "source_entity_id": 1,
  "target_entity_id": 2,
  "relation_type": "uses",
  "created_by": "agent-1"
}
```
Required: `source_entity_id`, `target_entity_id`, `relation_type`, `created_by`

**POST /api/v1/channels**
```json
{
  "name": "general",
  "created_by": "agent-1",
  "description": "General discussion channel",
  "channel_type": "broadcast"
}
```
Required: `name`, `created_by`. Optional: `description`, `channel_type`

**POST /api/v1/agents/register**
```json
{
  "agent_id": "agent-1",
  "name": "Claude Code",
  "agent_type": "claude",
  "capabilities": ["code", "memory"],
  "metadata": {}
}
```
Required: `agent_id`, `name`, `agent_type`. Optional: `capabilities`, `metadata`

WebSocket at `ws://localhost:8100/ws` for real-time channel subscriptions.

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `HIVEMIND_LISTEN_ADDR` | `0.0.0.0:8100` | API address |
| `HIVEMIND_RTDB_URL` | `ws://127.0.0.1:3001` | RaftTimeDB URL |
| `HIVEMIND_LLM_PROVIDER` | `anthropic` | LLM provider (openai/anthropic/ollama/codegate/URL) |
| `HIVEMIND_LLM_API_KEY` | - | LLM API key |
| `HIVEMIND_LLM_MODEL` | `claude-sonnet-4-20250514` | LLM model |
| `HIVEMIND_EMBEDDING_MODEL` | `openai:text-embedding-3-small` | Embedding model |
| `HIVEMIND_EMBEDDING_API_KEY` | - | Embedding API key |
| `HIVEMIND_DATA_DIR` | `./data` | Snapshot directory |
| `HIVEMIND_SNAPSHOT_INTERVAL` | `60` | Snapshot interval (seconds) |
| `HIVEMIND_ENABLE_REPLICATION` | `false` | Enable Raft replication |

## Building from Source

```bash
cargo build                    # Build core + CLI
cargo test                     # Run all tests
cd crates/mcp-server && npm install  # Install MCP server deps
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[Apache 2.0](LICENSE)
