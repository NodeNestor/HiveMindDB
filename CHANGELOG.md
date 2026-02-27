# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Core memory engine with in-memory DashMap stores and atomic ID counters
- Knowledge graph with entities, typed relationships, and BFS graph traversal
- Bi-temporal data model — memories have `valid_from`/`valid_until`, invalidated not deleted
- Hybrid search combining keyword matching and vector similarity (70% vector / 30% keyword)
- LLM-powered knowledge extraction pipeline with conflict resolution (ADD/UPDATE/NOOP)
- Provider-agnostic LLM support: OpenAI, Anthropic, Ollama, CodeGate, custom URLs
- Vector embedding engine with cosine similarity search (OpenAI-compatible API)
- Hivemind channels — pub/sub system for real-time memory sharing between agents
- WebSocket server for real-time channel subscriptions
- Snapshot persistence — periodic JSON snapshots to disk with atomic writes
- RaftTimeDB replication client — forwards writes through Raft consensus
- REST API with 20 endpoints (axum)
- MCP server with 20 tools — AgentCore-compatible (`remember`/`recall`/`forget`/`search`/`list_topics`)
- CLI tool (`hmdb`) for cluster management and memory operations
- SpacetimeDB WASM module with 8 tables and 6 reducers
- Docker Compose setup for local 3-node cluster (SpacetimeDB + RaftTimeDB + HiveMindDB)
- AgentCore integration (library.json entry + env var)
- CodeGate proxy support for both LLM and embedding calls
- Full audit trail — every memory change recorded with who, what, when, why
- Graceful shutdown with final snapshot save and connection drain
- GitHub Actions CI for Linux, Windows, macOS
