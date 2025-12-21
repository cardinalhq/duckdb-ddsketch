# DDSketch Extension for DuckDB

A DuckDB extension providing DDSketch quantile sketches with relative-error guarantees. DDSketches allow you to compute approximate quantiles (percentiles) over large datasets while using bounded memory, and can be merged across time periods or data partitions.

## Installation

```sql
LOAD 'ddsketch.duckdb_extension';
```

For unsigned extensions, start DuckDB with:
```bash
duckdb -unsigned
```

## Quick Start

```sql
-- Create a sketch with 1% relative accuracy
CREATE TABLE metrics AS SELECT sketch FROM ddsketch_create(0.01);

-- Add values
UPDATE metrics SET sketch = ddsketch_add(sketch, 100.0);
UPDATE metrics SET sketch = ddsketch_add(sketch, 200.0);
UPDATE metrics SET sketch = ddsketch_add(sketch, 300.0);

-- Query statistics
SELECT
    ddsketch_count(sketch) as count,
    ddsketch_avg(sketch) as avg,
    ddsketch_quantile(sketch, 0.50) as p50,
    ddsketch_quantile(sketch, 0.95) as p95,
    ddsketch_quantile(sketch, 0.99) as p99
FROM metrics;
```

## Functions

### Creating Sketches

| Function | Description |
|----------|-------------|
| `ddsketch_create(relative_accuracy)` | Creates a new empty sketch. Returns a table with one row containing the serialized sketch. |

**Parameters:**
- `relative_accuracy` (DOUBLE): The maximum relative error for quantile estimates. Default: 0.01 (1%). Lower values = more accurate but larger sketches.

**Example:**
```sql
-- Create a sketch with 0.1% relative accuracy
SELECT sketch FROM ddsketch_create(0.001);
```

### Adding Values

| Function | Description |
|----------|-------------|
| `ddsketch_add(sketch, value)` | Adds a value to a sketch, returns the updated sketch. |

**Parameters:**
- `sketch` (BLOB): A serialized DDSketch
- `value` (DOUBLE): The value to add

**Example:**
```sql
UPDATE my_table SET sketch = ddsketch_add(sketch, measurement_value);
```

### Merging Sketches

| Function | Description |
|----------|-------------|
| `ddsketch_merge(sketch1, sketch2)` | Merges two sketches into one, returns the combined sketch. |

**Parameters:**
- `sketch1` (BLOB): First serialized DDSketch
- `sketch2` (BLOB): Second serialized DDSketch

**Requirements:** Both sketches must have the same relative accuracy configuration.

**Example:**
```sql
SELECT ddsketch_merge(sketch_a, sketch_b) as merged FROM my_table;
```

### Querying Statistics

| Function | Returns | Description |
|----------|---------|-------------|
| `ddsketch_count(sketch)` | BIGINT | Number of values added to the sketch |
| `ddsketch_sum(sketch)` | DOUBLE | Sum of all values |
| `ddsketch_avg(sketch)` | DOUBLE | Average of all values |
| `ddsketch_min(sketch)` | DOUBLE | Minimum value |
| `ddsketch_max(sketch)` | DOUBLE | Maximum value |
| `ddsketch_quantile(sketch, q)` | DOUBLE | Value at quantile q (0.0 to 1.0) |

**Example:**
```sql
SELECT
    ddsketch_count(sketch) as n,
    ddsketch_min(sketch) as min,
    ddsketch_quantile(sketch, 0.25) as p25,
    ddsketch_quantile(sketch, 0.50) as p50,
    ddsketch_quantile(sketch, 0.75) as p75,
    ddsketch_quantile(sketch, 0.90) as p90,
    ddsketch_quantile(sketch, 0.95) as p95,
    ddsketch_quantile(sketch, 0.99) as p99,
    ddsketch_max(sketch) as max
FROM metrics;
```

## Common Use Cases

### Aggregating Pre-Computed Sketches

When you have serialized sketches stored in a table (e.g., pre-aggregated per time bucket or partition), you can merge them to compute quantiles across the combined dataset.

```sql
-- Table structure: each row has a serialized sketch for a time period
CREATE TABLE hourly_latency_sketches (
    hour TIMESTAMP,
    service VARCHAR,
    latency_sketch BLOB  -- serialized DDSketch
);

-- Merge all sketches for a service over a day
WITH merged AS (
    SELECT
        service,
        -- Use a recursive merge or aggregate
        latency_sketch as sketch
    FROM hourly_latency_sketches
    WHERE hour >= '2024-01-01' AND hour < '2024-01-02'
      AND service = 'api-gateway'
)
SELECT
    ddsketch_quantile(sketch, 0.50) as p50_latency,
    ddsketch_quantile(sketch, 0.95) as p95_latency,
    ddsketch_quantile(sketch, 0.99) as p99_latency
FROM merged;
```

### Merging Multiple Sketch Columns

If you have multiple sketch columns representing different dimensions that need to be combined:

```sql
-- Merge sketches from multiple regions into a global view
SELECT
    timestamp,
    ddsketch_merge(
        ddsketch_merge(us_east_sketch, us_west_sketch),
        ddsketch_merge(eu_west_sketch, ap_south_sketch)
    ) as global_sketch
FROM regional_metrics;

-- Then query the merged result
SELECT
    timestamp,
    ddsketch_quantile(global_sketch, 0.99) as global_p99
FROM (
    SELECT
        timestamp,
        ddsketch_merge(
            ddsketch_merge(us_east_sketch, us_west_sketch),
            ddsketch_merge(eu_west_sketch, ap_south_sketch)
        ) as global_sketch
    FROM regional_metrics
);
```

### Folding Sketches Across Rows

To merge all sketches in a column into a single result, use a recursive approach or window functions:

```sql
-- Using a subquery chain for small datasets
WITH RECURSIVE merged_sketches AS (
    -- Base case: first row
    SELECT
        sketch,
        ROW_NUMBER() OVER (ORDER BY hour) as rn
    FROM hourly_latency_sketches
    WHERE service = 'api-gateway'
),
sketch_fold AS (
    -- Start with first sketch
    SELECT sketch, rn FROM merged_sketches WHERE rn = 1
    UNION ALL
    -- Merge each subsequent sketch
    SELECT
        ddsketch_merge(f.sketch, m.sketch),
        m.rn
    FROM sketch_fold f
    JOIN merged_sketches m ON m.rn = f.rn + 1
)
SELECT
    ddsketch_count(sketch) as total_count,
    ddsketch_quantile(sketch, 0.95) as p95
FROM sketch_fold
ORDER BY rn DESC
LIMIT 1;
```

### Writing Merged Sketches Back to Storage

```sql
-- Aggregate hourly sketches into daily sketches
INSERT INTO daily_latency_sketches (day, service, latency_sketch)
WITH hourly AS (
    SELECT
        DATE_TRUNC('day', hour) as day,
        service,
        latency_sketch
    FROM hourly_latency_sketches
    WHERE hour >= '2024-01-01' AND hour < '2024-01-02'
)
-- For each service, merge all hourly sketches
SELECT
    day,
    service,
    -- This requires iterative merging; simplified here
    latency_sketch
FROM hourly
GROUP BY day, service;
```

## Serialization Format

Sketches are serialized using the **DataDog wire format** and stored as BLOB. This format is:
- **Compatible**: Matches DataDog Agent's DDSketch serialization exactly
- **Compact**: Typically 100-500 bytes per sketch depending on data distribution
- **Binary**: Stored as BLOB for efficient storage in Parquet, database columns, etc.

```sql
-- Store sketch in a table
CREATE TABLE sketch_store (
    id INTEGER,
    sketch BLOB
);

INSERT INTO sketch_store
SELECT 1, sketch FROM ddsketch_create(0.01);

-- Retrieve and query later
SELECT ddsketch_count(sketch) FROM sketch_store WHERE id = 1;
```

## Limitations

### Accuracy
- Quantile estimates have relative error bounded by the configured `relative_accuracy`
- A relative accuracy of 0.01 means the true quantile is within 1% of the reported value
- Min and max values are tracked exactly
- Count and sum are exact

### Merge Constraints
- **Sketches must have identical configuration** (same relative accuracy) to merge
- Attempting to merge incompatible sketches returns an empty string
- No automatic error on merge failure; check for empty results

### Performance Considerations
- Each sketch operation deserializes and re-serializes the sketch
- For bulk operations, consider batching or using aggregate functions (when available)
- Sketch size grows logarithmically with the range of values, not linearly with count

### Value Constraints
- Values must be positive (DDSketch limitation)
- Zero and negative values may produce unexpected results
- NaN and infinity values should be filtered before adding

### Current Limitations
- No aggregate function for folding sketches (requires manual merging)
- No direct way to add multiple values in a single call
- Table function `ddsketch_create` returns a table; use subquery or INSERT to capture

## Configuration Guidelines

| Use Case | Recommended Accuracy | Sketch Size |
|----------|---------------------|-------------|
| Rough estimates | 0.05 (5%) | ~500 bytes |
| Standard monitoring | 0.01 (1%) | ~1-2 KB |
| Precise SLO tracking | 0.005 (0.5%) | ~2-3 KB |
| High-precision analysis | 0.001 (0.1%) | ~3-5 KB |

## Error Handling

Functions return sensible defaults on error:
- Invalid sketch input → returns `NaN` for numeric functions, `0` for count, empty string for sketch functions
- Empty sketch → `NaN` for quantiles, `0` for count/sum

Check for valid results:
```sql
SELECT
    CASE WHEN ddsketch_count(sketch) > 0
         THEN ddsketch_quantile(sketch, 0.95)
         ELSE NULL
    END as p95
FROM my_table;
```

## Platform Support

| Platform | Architecture | Status |
|----------|--------------|--------|
| macOS | arm64 (Apple Silicon) | ✅ |
| macOS | amd64 (Intel) | ✅ |
| Linux | arm64 | ✅ |
| Linux | amd64 | ✅ |

## Building from Source

### Prerequisites

- Rust 1.84+ (`rustup update stable`)
- Python 3 (for extension packaging)
- Docker (for Linux cross-compilation on macOS)

### Build Commands

```bash
# Build the shared library
cargo build --release

# Package as DuckDB extension (macOS arm64 example)
python3 extension-ci-tools/scripts/append_extension_metadata.py \
  --library-file target/release/libddsketch.dylib \
  --extension-name ddsketch \
  --duckdb-platform osx_arm64 \
  --duckdb-version v1.4.3 \
  --extension-version v0.1.0 \
  --out-file ddsketch.duckdb_extension
```

### Platform Strings

| Platform | Library | Platform String |
|----------|---------|-----------------|
| macOS arm64 | `libddsketch.dylib` | `osx_arm64` |
| macOS amd64 | `libddsketch.dylib` | `osx_amd64` |
| Linux arm64 | `libddsketch.so` | `linux_arm64` |
| Linux amd64 | `libddsketch.so` | `linux_amd64` |

### Cross-Compile for Linux (from macOS)

```bash
# Build for Linux arm64 (native on ARM Mac, fast)
./scripts/build-linux.sh arm64

# Build for Linux amd64 (requires x86 machine or CI)
./scripts/build-linux.sh amd64

# Output in dist/linux-{arch}/libddsketch.so
```

### Run Tests

```bash
cargo test
```

## License

Apache-2.0
