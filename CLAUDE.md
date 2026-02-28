# HiveMindDB

## What This Is

HiveMindDB is a **distributed AI agent memory system** built on top of RaftTimeDB. It gives AI agent swarms (Claude Code, OpenCode, Aider, custom agents) persistent, replicated, shared memory — knowledge graphs, semantic search, temporal facts, real-time hivemind channels, and LLM-powered knowledge extraction.

Think of it as: mem0 + Graphiti + Graffiti channels, all Raft-replicated so the hivemind survives node failures.

## Architecture

```
Agents (Claude Code / OpenCode / Aider)
    ↓ MCP tools or REST API or WebSocket
HiveMindDB Sidecar (memory engine + API)
    ↓ WebSocket (replication)
RaftTimeDB (Raft consensus)
    ↓
SpacetimeDB (persistent storage, deterministic WASM)
```

## Project Structure

```
crates/
  core/src/           # HiveMindDB server binary
    main.rs           # Entry point: starts HTTP API + WebSocket + snapshots + replication
    api.rs            # REST + WebSocket API (axum) — 20 endpoints
    memory_engine.rs  # Memory CRUD, hybrid search, graph, conflict resolution, extraction
    channels.rs       # Hivemind channel pub/sub (broadcast)
    extraction.rs     # LLM-powered fact extraction (OpenAI/Anthropic/Ollama/CodeGate)
    embeddings.rs     # Vector embeddings + cosine similarity search
    persistence.rs    # Snapshot-to-disk + RaftTimeDB replication client
    websocket.rs      # WebSocket server for real-time client subscriptions
    types.rs          # All data types
    config.rs         # Configuration
  mcp-server/         # MCP server (Node.js) for Claude Code / OpenCode / Aider
    src/index.js      # MCP protocol handler — 20 tools (AgentCore-compatible)
  cli/src/            # hmdb CLI tool
    main.rs           # Commands: status, add, search, extract, entity, traverse, etc.
module/src/           # SpacetimeDB WASM module
  lib.rs              # Tables + reducers (Raft-replicated state)
deploy/
  docker/             # Docker Compose for dev cluster
  agentcore/          # AgentCore integration patches
```

## Development

```bash
cargo build                    # Build core + CLI
cargo test                     # Run all 44 tests
cd crates/mcp-server && npm install  # Install MCP server deps
```

## Key Design Decisions

- **In-memory + snapshot**: Core engine uses DashMap for concurrent in-memory storage, with periodic JSON snapshots to disk for restart recovery.
- **Hybrid search**: Combines keyword matching with vector similarity. Local embeddings enabled by default via `fastembed` (ONNX Runtime, CPU-only, 22M param model). Also supports external APIs (OpenAI, Ollama, CodeGate). Falls back to keyword-only when embeddings are disabled.
- **LLM extraction**: Sends conversation text to an LLM (OpenAI/Anthropic/Ollama/CodeGate) to extract facts, entities, and relationships. Handles conflict resolution (ADD/UPDATE/NOOP).
- **Provider-agnostic**: LLM and embedding calls work with any OpenAI-compatible API, including CodeGate proxy, Ollama local models, and custom URLs.
- **MCP-first**: Primary interface is MCP tools (drop-in compatible with AgentCore's agent-memory).
- **AgentCore-compatible**: Implements remember/recall/forget/search/list_topics matching AgentCore's existing interface.
- **Bi-temporal**: Memories have valid_from/valid_until — old facts are invalidated, never deleted.
- **Channel pub/sub**: Agents subscribe to channels and get real-time pushes via broadcast::channel + WebSocket.
- **Replication**: Optional RaftTimeDB replication client forwards writes through Raft consensus for multi-node sync.
- **Graceful shutdown**: Ctrl+C triggers final snapshot save, connection drain, and clean exit.

## Code Conventions

- Rust 2024 edition
- tokio async runtime
- axum for HTTP/WS API
- dashmap for concurrent in-memory stores
- serde for serialization
- chrono for timestamps
- tracing for structured logging
- reqwest for LLM/embedding API calls

## Configuration

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--listen-addr` | `HIVEMIND_LISTEN_ADDR` | `0.0.0.0:8100` | REST + WebSocket address |
| `--rtdb-url` | `HIVEMIND_RTDB_URL` | `ws://127.0.0.1:3001` | RaftTimeDB WebSocket URL |
| `--llm-provider` | `HIVEMIND_LLM_PROVIDER` | `anthropic` | LLM provider (openai/anthropic/ollama/codegate/URL) |
| `--llm-api-key` | `HIVEMIND_LLM_API_KEY` | none | LLM API key |
| `--llm-model` | `HIVEMIND_LLM_MODEL` | `claude-sonnet-4-20250514` | LLM model for extraction |
| `--embedding-model` | `HIVEMIND_EMBEDDING_MODEL` | `local:all-MiniLM-L6-v2` | Embedding model (provider:model) |
| `--embedding-api-key` | `HIVEMIND_EMBEDDING_API_KEY` | none | Embedding API key |
| `--data-dir` | `HIVEMIND_DATA_DIR` | `./data` | Snapshot directory |
| `--snapshot-interval` | `HIVEMIND_SNAPSHOT_INTERVAL` | `60` | Snapshot interval (seconds, 0=disable) |
| `--enable-replication` | `HIVEMIND_ENABLE_REPLICATION` | false | Enable RaftTimeDB replication |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/memories` | POST | Add memory |
| `/api/v1/memories` | GET | List memories (with filters) |
| `/api/v1/memories/:id` | GET/PUT/DELETE | Get, update, invalidate |
| `/api/v1/memories/:id/history` | GET | Audit trail |
| `/api/v1/search` | POST | Hybrid search (keyword + vector) |
| `/api/v1/extract` | POST | LLM knowledge extraction |
| `/api/v1/entities` | POST | Add entity |
| `/api/v1/entities/:id` | GET | Get entity |
| `/api/v1/entities/find` | POST | Find entity by name |
| `/api/v1/entities/:id/relationships` | GET | Entity relationships |
| `/api/v1/relationships` | POST | Add relationship |
| `/api/v1/graph/traverse` | POST | Graph traversal |
| `/api/v1/channels` | POST/GET | Create/list channels |
| `/api/v1/channels/:id/share` | POST | Share memory to channel |
| `/api/v1/agents/register` | POST | Register agent |
| `/api/v1/agents` | GET | List agents |
| `/api/v1/agents/:id/heartbeat` | POST | Agent heartbeat |
| `/ws` | GET (upgrade) | WebSocket real-time subscriptions |
| `/api/v1/status` | GET | Cluster stats |
| `/health` | GET | Health check |
