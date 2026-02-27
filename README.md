<div align="center">

# HiveMindDB

### Shared memory for AI agent swarms.

Distributed, fault-tolerant memory system for AI agents — knowledge graphs, semantic search, and real-time hivemind channels, all replicated via Raft consensus.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85+-orange.svg)](https://www.rust-lang.org)

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

### With Claude Code (one command)

```bash
# Start HiveMindDB
docker compose -f deploy/docker/docker-compose.yml up -d

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

### What Your Agent Can Do

```
> Remember that the user prefers Rust over Python for new projects.
✓ Stored memory #1 under topic "preferences"

> What does the user prefer?
Found 1 result:
  #1 [score: 0.95] User prefers Rust over Python for new projects
     tags: preferences

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
| **Bi-Temporal** | Old facts are invalidated, not deleted — query "what did we know last Tuesday?" |
| **Hivemind Channels** | Agents subscribe to channels, get real-time updates when any agent learns something |
| **Conflict Resolution** | New facts that contradict old ones trigger temporal invalidation |
| **Full Audit Trail** | Every memory change is recorded — who changed what, when, and why |
| **MCP Native** | Drop-in MCP server for Claude Code, OpenCode, Aider |
| **AgentCore Compatible** | Same `remember`/`recall`/`forget`/`search` interface |
| **Fault Tolerant** | Built on RaftTimeDB — kill a node, the hivemind keeps thinking |
| **REST + WebSocket API** | Works with any HTTP client or agent framework |

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
└──────────────────┬──────────────────────────┘
                   │ WebSocket
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
hmdb status                                    # Cluster stats
hmdb add "User prefers Rust" --user ludde     # Add a memory
hmdb search "what does the user prefer?"      # Search
hmdb entity "RaftTimeDB"                       # Entity + relationships
hmdb traverse 1 --depth 3                      # Graph traversal
hmdb history 42                                # Audit trail
hmdb forget 42 --reason "outdated"            # Invalidate
hmdb channels                                  # List channels
hmdb agents                                    # List agents
```

## API

Full REST API at `http://localhost:8100/api/v1/`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/memories` | POST | Add memory |
| `/memories/:id` | GET/PUT/DELETE | Get, update, invalidate |
| `/memories/:id/history` | GET | Audit trail |
| `/search` | POST | Semantic search |
| `/entities` | POST | Add entity |
| `/entities/:id/relationships` | GET | Entity relationships |
| `/graph/traverse` | POST | Graph traversal |
| `/channels` | POST/GET | Create/list channels |
| `/channels/:id/share` | POST | Share memory to channel |
| `/agents/register` | POST | Register agent |
| `/agents` | GET | List agents |
| `/status` | GET | Cluster stats |

## License

[Apache 2.0](LICENSE)
