# HiveMindDB

## What This Is

HiveMindDB is a **distributed AI agent memory system** built on top of RaftTimeDB. It gives AI agent swarms (Claude Code, OpenCode, Aider, custom agents) persistent, replicated, shared memory — knowledge graphs, semantic search, temporal facts, and real-time hivemind channels.

Think of it as: mem0 + Graphiti + Graffiti channels, all Raft-replicated so the hivemind survives node failures.

## Architecture

```
Agents (Claude Code / OpenCode / Aider)
    ↓ MCP tools or REST API
HiveMindDB Sidecar (memory engine + API)
    ↓ WebSocket
RaftTimeDB (Raft consensus)
    ↓
SpacetimeDB (persistent storage, deterministic WASM)
```

## Project Structure

```
crates/
  core/src/           # HiveMindDB server binary
    main.rs           # Entry point
    api.rs            # REST + WebSocket API (axum)
    memory_engine.rs  # Memory CRUD, search, graph, conflict resolution
    channels.rs       # Hivemind channel pub/sub
    types.rs          # All data types
    config.rs         # Configuration
  mcp-server/         # MCP server (Node.js) for Claude Code / OpenCode / Aider
    src/index.js      # MCP protocol handler + tool definitions
  cli/src/            # hmdb CLI tool
    main.rs           # Commands: status, add, search, entity, traverse, etc.
module/src/           # SpacetimeDB WASM module
  lib.rs              # Tables + reducers (Raft-replicated state)
deploy/
  docker/             # Docker Compose for dev cluster
  agentcore/          # AgentCore integration patches
```

## Development

```bash
cargo build                    # Build core + CLI
cargo test                     # Run all tests
cd crates/mcp-server && npm install  # Install MCP server deps
```

## Key Design Decisions

- **In-memory Phase 1**: Core engine uses DashMap for concurrent in-memory storage. Phase 2 replaces with SpacetimeDB subscriptions via RaftTimeDB.
- **MCP-first**: Primary interface is MCP tools (drop-in compatible with AgentCore's agent-memory).
- **AgentCore-compatible**: Implements remember/recall/forget/search/list_topics matching AgentCore's existing interface.
- **Bi-temporal**: Memories have valid_from/valid_until — old facts are invalidated, never deleted.
- **Channel pub/sub**: Agents subscribe to channels and get real-time pushes via broadcast::channel.

## Code Conventions

- Rust 2024 edition
- tokio async runtime
- axum for HTTP/WS API
- dashmap for concurrent in-memory stores
- serde for serialization
- chrono for timestamps
- tracing for structured logging
