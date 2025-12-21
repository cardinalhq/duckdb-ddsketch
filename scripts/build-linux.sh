#!/bin/bash
set -euo pipefail

# Build duckdb-ddsketch for Linux platforms using Docker
# Usage: ./scripts/build-linux.sh [amd64|arm64|all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/dist"

PLATFORM="${1:-all}"

build_platform() {
    local arch="$1"
    local docker_platform

    case "$arch" in
        amd64)
            docker_platform="linux/amd64"
            ;;
        arm64)
            docker_platform="linux/arm64"
            ;;
        *)
            echo "Unknown architecture: $arch"
            exit 1
            ;;
    esac

    echo "Building for linux/${arch}..."
    mkdir -p "${OUTPUT_DIR}/linux-${arch}"

    # Build in Docker container
    docker run --rm \
        --platform "${docker_platform}" \
        -v "${PROJECT_DIR}:/build" \
        -w /build \
        rust:1.84-bookworm \
        bash -c "cargo build --release && cp target/release/libddsketch.so /build/dist/linux-${arch}/"

    echo "Built: ${OUTPUT_DIR}/linux-${arch}/libddsketch.so"
}

case "$PLATFORM" in
    all)
        build_platform arm64   # Native first (faster)
        build_platform amd64   # Emulated
        ;;
    amd64|arm64)
        build_platform "$PLATFORM"
        ;;
    *)
        echo "Usage: $0 [amd64|arm64|all]"
        echo "Builds DuckDB DDSketch extension for Linux platforms"
        exit 1
        ;;
esac

echo ""
echo "Build complete. Output:"
ls -lh "${OUTPUT_DIR}"/*/*.so 2>/dev/null || true
