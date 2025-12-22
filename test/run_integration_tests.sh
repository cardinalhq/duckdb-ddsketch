#!/bin/bash
# Run DuckDB integration tests for the ddsketch extension

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
EXTENSION_PATH="${PROJ_DIR}/build/release/ddsketch.duckdb_extension"

if [ ! -f "$EXTENSION_PATH" ]; then
    echo "ERROR: Extension not found at $EXTENSION_PATH"
    echo "Run 'make release' first"
    exit 1
fi

echo "=== Running DDSketch Integration Tests ==="
echo "Extension: $EXTENSION_PATH"
echo

# Run the SQL tests and capture output
OUTPUT=$(duckdb -unsigned -cmd "LOAD '$EXTENSION_PATH';" < "${SCRIPT_DIR}/integration_test.sql" 2>&1)
EXIT_CODE=$?

echo "$OUTPUT"
echo

# Check for any false values in the output (test failures)
if echo "$OUTPUT" | grep -q "false"; then
    echo "=== FAILED: Some tests returned false ==="
    exit 1
fi

# Check for errors
if echo "$OUTPUT" | grep -qi "error"; then
    echo "=== FAILED: Errors detected ==="
    exit 1
fi

if [ $EXIT_CODE -ne 0 ]; then
    echo "=== FAILED: DuckDB exited with code $EXIT_CODE ==="
    exit 1
fi

echo "=== All integration tests passed ==="
