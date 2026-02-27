# HiveMindDB Docker build
# Multi-stage: build Rust binaries, then create minimal runtime image

FROM rust:1.85-bookworm AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock* ./
COPY crates/ crates/
COPY module/ module/

# Build release binaries
RUN cargo build --release --bin hiveminddb --bin hmdb

# Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/hiveminddb /usr/local/bin/
COPY --from=builder /build/target/release/hmdb /usr/local/bin/

RUN useradd -m hivemind
USER hivemind
WORKDIR /home/hivemind

EXPOSE 8100

ENTRYPOINT ["hiveminddb"]
