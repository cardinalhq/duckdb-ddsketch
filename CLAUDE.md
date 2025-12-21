# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build and package extension (recommended)
make release

# Extension output: build/release/ddsketch.duckdb_extension

# Run tests
cargo test

# Test extension in DuckDB
duckdb -unsigned -c "LOAD 'build/release/ddsketch.duckdb_extension';"
```

## Critical: DuckDB API Version

**DuckDB v1.4.x uses C API version v1.2.0, NOT v1.4.3.**

The Makefile `TARGET_DUCKDB_VERSION` must be set to `v1.2.0`. If you get errors like:
- "Built for DuckDB C API version 'vX.X.X', but we can only load extensions built for DuckDB C API 'v1.2.0'"

Check that `TARGET_DUCKDB_VERSION=v1.2.0` in the Makefile.

## Architecture

This is a Rust DuckDB extension providing DDSketch quantile functions.

### Source Files

- **`src/lib.rs`** - DuckDB extension entry point, registers all functions:
  - Table function: `ddsketch_create(relative_accuracy)`
  - Scalar functions: `ddsketch_add`, `ddsketch_merge`, `ddsketch_quantile`, `ddsketch_count`, `ddsketch_sum`, `ddsketch_min`, `ddsketch_max`, `ddsketch_avg`, `ddsketch_stats`
  - Aggregate function: `ddsketch_agg` (merges BLOBs)

- **`src/datadog_encoding.rs`** - DataDog DDSketch wire format implementation:
  - `DataDogSketch` struct with encode/decode
  - Compatible with DataDog Agent's serialization (github.com/DataDog/sketches-go v1.4.7)
  - Wire format: flag bytes, index mapping (gamma/offset), bin stores with delta-encoded indices

### Key Implementation Details

- Sketches are serialized as BLOB using DataDog wire format
- Quantile calculation uses Go's `KeyAtRank` algorithm (uses `>` not `>=` for rank comparison)
- Sketches can only merge if they have the same relative accuracy (gamma value)
- Extension uses `duckdb` crate v1.4.3 with `vtab-loadable` and `vscalar` features

### Test Vectors

The `datadog_encoding.rs` contains compatibility tests with Go-generated hex strings to verify encoding/decoding matches the DataDog library exactly.
