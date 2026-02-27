# Contributing to HiveMindDB

Thanks for your interest in contributing! HiveMindDB is early-stage and there's a lot to build.

## Getting Started

### Prerequisites

- **Rust** (latest stable): https://rustup.rs
- **Node.js** (for MCP server): https://nodejs.org
- **Docker** (for running the full stack): https://docker.com

### Building

```bash
git clone https://github.com/NodeNestor/HiveMindDB.git
cd HiveMindDB
cargo build
```

### Running the Tests

```bash
cargo test
```

### Running Locally (Docker Compose)

```bash
cd deploy/docker
docker compose up
```

This starts a 3-node cluster with SpacetimeDB + RaftTimeDB + HiveMindDB sidecars.

## Project Structure

```
crates/
  core/             # HiveMindDB server binary
    src/
      main.rs           # Entry point
      api.rs            # REST + WebSocket API (axum)
      memory_engine.rs  # Memory CRUD, search, graph, extraction
      channels.rs       # Pub/sub channels
      extraction.rs     # LLM extraction pipeline
      embeddings.rs     # Vector embeddings + similarity search
      persistence.rs    # Snapshots + RaftTimeDB replication
      websocket.rs      # WebSocket real-time subscriptions
      types.rs          # Data types
      config.rs         # Configuration
  cli/              # CLI tool (hmdb)
  mcp-server/       # MCP server for Claude Code / OpenCode / Aider
module/             # SpacetimeDB WASM module
deploy/
  docker/           # Docker Compose for local dev
  agentcore/        # AgentCore integration
```

## What Needs Work

Check the [GitHub Issues](https://github.com/NodeNestor/HiveMindDB/issues) for current priorities.

### Core

- Integration tests with live LLM/embedding APIs
- Docker-based E2E test suite
- Batch embedding indexing on snapshot restore
- Memory deduplication and merging strategies
- Graph query language (Cypher-like)

### Integrations

- More MCP tool coverage
- Langchain / LlamaIndex memory adapters
- OpenTelemetry tracing
- Kubernetes Helm chart

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Ensure `cargo test` and `cargo clippy` pass
4. Write a clear PR description explaining what and why
5. Submit!

## Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy` and fix warnings
- Keep functions focused and small
- Add comments for non-obvious logic (but don't over-comment)
- Error handling: use `anyhow` for applications, `thiserror` for libraries

## Communication

- **GitHub Issues**: Bug reports, feature requests, questions
- **Pull Requests**: Code contributions
- **Discussions**: Architecture decisions, design proposals

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
