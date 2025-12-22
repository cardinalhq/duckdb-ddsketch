-- DDSketch DuckDB Integration Tests
-- This file tests all extension functions with expected results

-- ============================================================================
-- Test 1: Create sketch and verify empty state
-- ============================================================================
SELECT 'Test 1: Create empty sketch' as test;
SELECT
    ddsketch_count(sketch) = 0 as count_ok,
    ddsketch_sum(sketch) IS NULL OR ddsketch_sum(sketch) = 0 as sum_ok
FROM ddsketch_create(0.01);

-- ============================================================================
-- Test 2: Add values and verify count/sum
-- ============================================================================
SELECT 'Test 2: Add values' as test;
WITH sketch AS (
    SELECT ddsketch_add(ddsketch_add(ddsketch_add(
        (SELECT sketch FROM ddsketch_create(0.01)),
        10.0), 20.0), 30.0) as s
)
SELECT
    ddsketch_count(s) = 3 as count_ok,
    -- Sum is approximate (within 5% of expected)
    abs(ddsketch_sum(s) - 60.0) / 60.0 < 0.05 as sum_ok,
    abs(ddsketch_avg(s) - 20.0) / 20.0 < 0.05 as avg_ok
FROM sketch;

-- ============================================================================
-- Test 3: Min/Max tracking
-- ============================================================================
SELECT 'Test 3: Min/Max' as test;
WITH sketch AS (
    SELECT ddsketch_add(ddsketch_add(ddsketch_add(
        (SELECT sketch FROM ddsketch_create(0.01)),
        5.0), 50.0), 500.0) as s
)
SELECT
    -- Min/max are approximate (within 5% of expected)
    abs(ddsketch_min(s) - 5.0) / 5.0 < 0.05 as min_ok,
    abs(ddsketch_max(s) - 500.0) / 500.0 < 0.05 as max_ok
FROM sketch;

-- ============================================================================
-- Test 4: Quantile estimation
-- ============================================================================
SELECT 'Test 4: Quantile estimation' as test;
WITH sketch AS (
    SELECT ddsketch_add(ddsketch_add(ddsketch_add(ddsketch_add(ddsketch_add(
        ddsketch_add(ddsketch_add(ddsketch_add(ddsketch_add(ddsketch_add(
            (SELECT sketch FROM ddsketch_create(0.01)),
            10.0), 20.0), 30.0), 40.0), 50.0),
            60.0), 70.0), 80.0), 90.0), 100.0) as s
)
SELECT
    -- p50 should be around 50-60
    ddsketch_quantile(s, 0.50) BETWEEN 40.0 AND 70.0 as p50_ok,
    -- p90 should be around 90
    ddsketch_quantile(s, 0.90) BETWEEN 80.0 AND 100.0 as p90_ok
FROM sketch;

-- ============================================================================
-- Test 5: Merge sketches
-- ============================================================================
SELECT 'Test 5: Merge sketches' as test;
WITH
    s1 AS (SELECT ddsketch_add((SELECT sketch FROM ddsketch_create(0.01)), 10.0) as sketch),
    s2 AS (SELECT ddsketch_add((SELECT sketch FROM ddsketch_create(0.01)), 20.0) as sketch),
    merged AS (SELECT ddsketch_merge(s1.sketch, s2.sketch) as sketch FROM s1, s2)
SELECT
    ddsketch_count(sketch) = 2 as count_ok
FROM merged;

-- ============================================================================
-- Test 6: Aggregate function ddsketch_agg
-- ============================================================================
SELECT 'Test 6: ddsketch_agg aggregate' as test;
WITH data AS (
    SELECT ddsketch_add((SELECT sketch FROM ddsketch_create(0.01)), 10.0) as sketch
    UNION ALL
    SELECT ddsketch_add((SELECT sketch FROM ddsketch_create(0.01)), 20.0)
    UNION ALL
    SELECT ddsketch_add((SELECT sketch FROM ddsketch_create(0.01)), 30.0)
)
SELECT
    ddsketch_count(ddsketch_agg(sketch)) = 3 as count_ok
FROM data;

-- ============================================================================
-- Test 7: ddsketch_stats_agg with all fields
-- ============================================================================
SELECT 'Test 7: ddsketch_stats_agg struct' as test;
WITH data AS (
    SELECT ddsketch_add((SELECT sketch FROM ddsketch_create(0.01)), 10.0) as sketch
    UNION ALL
    SELECT ddsketch_add((SELECT sketch FROM ddsketch_create(0.01)), 20.0)
    UNION ALL
    SELECT ddsketch_add((SELECT sketch FROM ddsketch_create(0.01)), 30.0)
),
stats AS (SELECT ddsketch_stats_agg(sketch) as s FROM data)
SELECT
    s.count = 3 as count_ok,
    -- Sum/avg are approximate (within 5%)
    abs(s.sum - 60.0) / 60.0 < 0.05 as sum_ok,
    abs(s.avg - 20.0) / 20.0 < 0.05 as avg_ok,
    s.min > 0 as min_ok,
    s.max > 0 as max_ok,
    s.p50 > 0 as p50_ok,
    s.p95 > 0 as p95_ok,
    s.p99 > 0 as p99_ok,
    ddsketch_count(s.sketch) = 3 as sketch_usable
FROM stats;

-- ============================================================================
-- Test 8: NULL handling
-- ============================================================================
SELECT 'Test 8: NULL handling' as test;
SELECT
    ddsketch_add(NULL, 1.0) IS NULL as null_sketch_ok,
    ddsketch_quantile((SELECT sketch FROM ddsketch_create(0.01)), NULL) IS NULL as null_quantile_ok;

-- ============================================================================
-- Test 9: ddsketch_stats scalar function
-- ============================================================================
SELECT 'Test 9: ddsketch_stats scalar' as test;
WITH sketch AS (
    SELECT ddsketch_add(ddsketch_add(ddsketch_add(
        (SELECT sketch FROM ddsketch_create(0.01)),
        10.0), 20.0), 30.0) as s
)
SELECT
    (ddsketch_stats(s)).count = 3 as count_ok,
    (ddsketch_stats(s)).sum > 0 as sum_ok
FROM sketch;

-- ============================================================================
-- Summary
-- ============================================================================
SELECT 'All integration tests completed' as status;
