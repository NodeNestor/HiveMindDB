# HiveMindDB Docker build
# Multi-stage: build Rust binaries, then create minimal runtime image
#
# Uses newer glibc base for ONNX Runtime compatibility (local embeddings)

FROM rust:latest AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock* ./
COPY crates/ crates/
COPY module/ module/

# Build release binaries (includes local-embeddings by default)
RUN cargo build --release --bin hiveminddb --bin hmdb

# Runtime image — trixie has glibc 2.38+ needed by ONNX Runtime
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/hiveminddb /usr/local/bin/
COPY --from=builder /build/target/release/hmdb /usr/local/bin/

RUN useradd -m hivemind && mkdir -p /data && chown hivemind:hivemind /data
USER hivemind
WORKDIR /home/hivemind

EXPOSE 8100

ENTRYPOINT ["hiveminddb"]
