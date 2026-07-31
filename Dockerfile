FROM debian:bookworm-slim AS zig-download

ARG ZIG_VERSION=0.16.0
ARG ZIG_MINISIG_PK=RWSGOq2NVecA2UPNdBUZykf1CCb147pkmdtYxgb3Ti+JO/wCYvhbAb/U

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        minisign \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/zig && \
    curl -L "https://ziglang.org/download/${ZIG_VERSION}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz" -o /tmp/zig.tar.xz && \
    curl -L "https://ziglang.org/download/${ZIG_VERSION}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz.minisig" -o /tmp/zig.tar.xz.minisig && \
    minisign -Vm /tmp/zig.tar.xz -P "$ZIG_MINISIG_PK" && \
    tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && \
    rm /tmp/zig.tar.xz /tmp/zig.tar.xz.minisig

FROM debian:bookworm-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=zig-download /opt/zig /opt/zig

ENV PATH=/opt/zig:${PATH}

WORKDIR /opt/ggufy
COPY build.zig build.zig.zon build_ggml.zig ./
COPY src ./src
COPY vendor ./vendor
RUN zig build cli --release=fast

FROM debian:bookworm-slim AS cli-runtime

WORKDIR /app
COPY --from=builder /opt/ggufy/zig-out/bin/ggufy /usr/local/bin/ggufy

ENTRYPOINT ["ggufy"]
CMD ["--help"]

FROM ghcr.io/astral-sh/uv:0.9.18 AS uv

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    GGUFY_DATA_ROOT=/data \
    GGUFY_BINARY=/usr/local/bin/ggufy \
    GGUFY_MAX_CONCURRENT_JOBS=1

COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=builder /opt/ggufy/zig-out/bin/ggufy /usr/local/bin/ggufy

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY api ./api
RUN uv sync --frozen --no-dev

RUN groupadd --system --gid 10001 ggufy \
    && useradd --system --uid 10001 --gid ggufy --home-dir /app ggufy \
    && mkdir -p /data/input /data/output /data/tmp \
    && chown -R ggufy:ggufy /app /data

USER 10001:10001

EXPOSE 8000
VOLUME ["/data"]

CMD ["/app/.venv/bin/uvicorn", "ggufy_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
