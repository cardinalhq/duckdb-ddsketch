# Feature Gap Analysis

## Remaining Gap: Struct-Returning Aggregate

### Current State

The aggregate function `ddsketch_agg` returns a merged sketch as BLOB. To extract percentiles, you must call `ddsketch_quantile` multiple times, each deserializing the sketch.

### Workaround (CTE)

```sql
WITH merged AS (
    SELECT
        metric_name,
        ddsketch_agg(chq_sketch) as sketch
    FROM read_parquet('*.parquet')
    GROUP BY metric_name
)
SELECT
    metric_name,
    ddsketch_quantile(sketch, 0.50) as p50,
    ddsketch_quantile(sketch, 0.95) as p95,
    ddsketch_quantile(sketch, 0.99) as p99,
    sketch as chq_sketch
FROM merged
```

### Ideal Solution

Single aggregate returning struct with all values computed in one pass:

```sql
SELECT
    metric_name,
    ddsketch_stats_agg(chq_sketch) as stats
FROM read_parquet('*.parquet')
GROUP BY metric_name
```

Returns:
```
STRUCT(
    sketch BLOB,
    p25 DOUBLE,
    p50 DOUBLE,
    p75 DOUBLE,
    p90 DOUBLE,
    p95 DOUBLE,
    p99 DOUBLE
)
```

This avoids deserializing the merged sketch multiple times for each percentile extraction.
