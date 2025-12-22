// Copyright 2025-2026 CardinalHQ, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

extern crate duckdb;
extern crate libduckdb_sys;

mod datadog_encoding;

use duckdb::{
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId, Inserter},
    vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab},
    vtab::arrow::WritableVector,
    vscalar::{ScalarFunctionSignature, VScalar},
    Connection, Result,
};
use libduckdb_sys as ffi;
use datadog_encoding::DataDogSketch;
use std::{
    error::Error,
    ffi::CString,
};

// ============================================================================
// Serialization helpers for DDSketch (DataDog format)
// ============================================================================

/// Serialize a DDSketch to DataDog wire format bytes
/// Returns None on encoding failure instead of panicking
fn serialize_sketch(sketch: &DataDogSketch) -> Option<Vec<u8>> {
    sketch.encode().ok()
}

/// Deserialize a DDSketch from DataDog wire format bytes
fn deserialize_sketch(bytes: &[u8]) -> std::result::Result<DataDogSketch, Box<dyn Error>> {
    DataDogSketch::decode(bytes)
}

// ============================================================================
// ddsketch_create: Create a new empty DDSketch (Table Function)
// ============================================================================

#[repr(C)]
struct CreateBindData {
    relative_accuracy: f64,
}

#[repr(C)]
struct CreateInitData {
    done: std::sync::atomic::AtomicBool,
}

struct CreateSketchVTab;

impl VTab for CreateSketchVTab {
    type InitData = CreateInitData;
    type BindData = CreateBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        bind.add_result_column("sketch", LogicalTypeHandle::from(LogicalTypeId::Blob));

        // Optional relative_accuracy parameter, default 0.01 (1%)
        let relative_accuracy = if bind.get_parameter_count() > 0 {
            bind.get_parameter(0).to_string().parse::<f64>().unwrap_or(0.01)
        } else {
            0.01
        };

        Ok(CreateBindData { relative_accuracy })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(CreateInitData {
            done: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();
        let bind_data = func.get_bind_data();

        if init_data.done.swap(true, std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
        } else {
            let sketch = DataDogSketch::new(bind_data.relative_accuracy);
            match serialize_sketch(&sketch) {
                Some(bytes) => {
                    let vector = output.flat_vector(0);
                    vector.insert(0, bytes.as_slice());
                    output.set_len(1);
                }
                None => {
                    return Err("Failed to serialize sketch".into());
                }
            }
        }
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Double)])
    }
}

// ============================================================================
// Scalar Functions for DDSketch operations
// ============================================================================

// Helper to check if a row is valid (not NULL) in a given column
unsafe fn is_row_valid(input: &DataChunkHandle, col: usize, row: usize) -> bool {
    let vector = input.flat_vector(col);
    !vector.row_is_null(row as u64)
}

// Helper to get blob data from input at index (caller must check validity first)
unsafe fn get_blob_from_input(input: &DataChunkHandle, col: usize, row: usize) -> Vec<u8> {
    let values = input.flat_vector(col);
    let strings = values.as_slice_with_len::<ffi::duckdb_string_t>(input.len());
    let string_t_ptr = &strings[row] as *const ffi::duckdb_string_t as *mut ffi::duckdb_string_t;
    let len = ffi::duckdb_string_t_length(*string_t_ptr) as usize;

    if len == 0 {
        return Vec::new();
    }

    let data_ptr = ffi::duckdb_string_t_data(string_t_ptr);
    std::slice::from_raw_parts(data_ptr as *const u8, len).to_vec()
}

// Helper to get double from input (caller must check validity first)
unsafe fn get_double_from_input(input: &DataChunkHandle, col: usize, row: usize) -> f64 {
    let values = input.flat_vector(col);
    let doubles = values.as_slice_with_len::<f64>(input.len());
    doubles[row]
}

// Helper to write blob data to output vector
unsafe fn set_blob_in_output(output: &mut dyn WritableVector, row: usize, data: &[u8]) {
    let vector = output.flat_vector();
    vector.insert(row, data);
}

// Helper to set a row as NULL in output
unsafe fn set_null_in_output(output: &mut dyn WritableVector, row: usize) {
    let mut vector = output.flat_vector();
    vector.set_null(row);
}

// ============================================================================
// ddsketch_add: Add a value to a sketch (Scalar Function)
// ============================================================================

struct AddScalar;

impl VScalar for AddScalar {
    type State = ();

    unsafe fn invoke(
        _state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        for row in 0..input.len() {
            // Check if either input is NULL
            if !is_row_valid(input, 0, row) || !is_row_valid(input, 1, row) {
                set_null_in_output(output, row);
                continue;
            }

            let sketch_bytes = get_blob_from_input(input, 0, row);
            let value = get_double_from_input(input, 1, row);

            match deserialize_sketch(&sketch_bytes) {
                Ok(mut sketch) => {
                    sketch.add(value);
                    match serialize_sketch(&sketch) {
                        Some(bytes) => set_blob_in_output(output, row, &bytes),
                        None => set_null_in_output(output, row),
                    }
                }
                Err(_) => {
                    set_null_in_output(output, row);
                }
            }
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![
                LogicalTypeId::Blob.into(),   // sketch
                LogicalTypeId::Double.into(), // value
            ],
            LogicalTypeId::Blob.into(),
        )]
    }
}

// ============================================================================
// ddsketch_merge: Merge two sketches (Scalar Function)
// ============================================================================

struct MergeScalar;

impl VScalar for MergeScalar {
    type State = ();

    unsafe fn invoke(
        _state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        for row in 0..input.len() {
            // Check if either input is NULL
            if !is_row_valid(input, 0, row) || !is_row_valid(input, 1, row) {
                set_null_in_output(output, row);
                continue;
            }

            let sketch1_bytes = get_blob_from_input(input, 0, row);
            let sketch2_bytes = get_blob_from_input(input, 1, row);

            match (deserialize_sketch(&sketch1_bytes), deserialize_sketch(&sketch2_bytes)) {
                (Ok(mut sketch1), Ok(sketch2)) => {
                    if sketch1.merge(&sketch2).is_ok() {
                        match serialize_sketch(&sketch1) {
                            Some(bytes) => set_blob_in_output(output, row, &bytes),
                            None => set_null_in_output(output, row),
                        }
                    } else {
                        set_null_in_output(output, row);
                    }
                }
                _ => {
                    set_null_in_output(output, row);
                }
            }
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![
                LogicalTypeId::Blob.into(), // sketch1
                LogicalTypeId::Blob.into(), // sketch2
            ],
            LogicalTypeId::Blob.into(),
        )]
    }
}

// ============================================================================
// ddsketch_quantile: Get quantile value (Scalar Function)
// ============================================================================

struct QuantileScalar;

impl VScalar for QuantileScalar {
    type State = ();

    unsafe fn invoke(
        _state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        let mut out_vec = output.flat_vector();

        for row in 0..input.len() {
            // Check if either input is NULL
            if !is_row_valid(input, 0, row) || !is_row_valid(input, 1, row) {
                out_vec.set_null(row);
                continue;
            }

            let sketch_bytes = get_blob_from_input(input, 0, row);
            let quantile = get_double_from_input(input, 1, row);

            match deserialize_sketch(&sketch_bytes) {
                Ok(sketch) => {
                    match sketch.quantile(quantile) {
                        Some(val) => out_vec.as_mut_slice::<f64>()[row] = val,
                        None => out_vec.set_null(row),
                    }
                }
                Err(_) => {
                    out_vec.set_null(row);
                }
            }
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![
                LogicalTypeId::Blob.into(),   // sketch
                LogicalTypeId::Double.into(), // quantile (0.0-1.0)
            ],
            LogicalTypeId::Double.into(),
        )]
    }
}

// ============================================================================
// ddsketch_count: Get count (Scalar Function)
// ============================================================================

struct CountScalar;

impl VScalar for CountScalar {
    type State = ();

    unsafe fn invoke(
        _state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        let mut out_vec = output.flat_vector();

        for row in 0..input.len() {
            // Check if input is NULL
            if !is_row_valid(input, 0, row) {
                out_vec.set_null(row);
                continue;
            }

            let sketch_bytes = get_blob_from_input(input, 0, row);

            match deserialize_sketch(&sketch_bytes) {
                Ok(sketch) => {
                    out_vec.as_mut_slice::<i64>()[row] = sketch.count() as i64;
                }
                Err(_) => {
                    out_vec.set_null(row);
                }
            }
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeId::Blob.into()],
            LogicalTypeId::Bigint.into(),
        )]
    }
}

// ============================================================================
// ddsketch_min: Get min value (Scalar Function)
// ============================================================================

struct MinScalar;

impl VScalar for MinScalar {
    type State = ();

    unsafe fn invoke(
        _state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        let mut out_vec = output.flat_vector();

        for row in 0..input.len() {
            // Check if input is NULL
            if !is_row_valid(input, 0, row) {
                out_vec.set_null(row);
                continue;
            }

            let sketch_bytes = get_blob_from_input(input, 0, row);

            match deserialize_sketch(&sketch_bytes) {
                Ok(sketch) => {
                    match sketch.min() {
                        Some(val) => out_vec.as_mut_slice::<f64>()[row] = val,
                        None => out_vec.set_null(row),
                    }
                }
                Err(_) => {
                    out_vec.set_null(row);
                }
            }
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeId::Blob.into()],
            LogicalTypeId::Double.into(),
        )]
    }
}

// ============================================================================
// ddsketch_max: Get max value (Scalar Function)
// ============================================================================

struct MaxScalar;

impl VScalar for MaxScalar {
    type State = ();

    unsafe fn invoke(
        _state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        let mut out_vec = output.flat_vector();

        for row in 0..input.len() {
            // Check if input is NULL
            if !is_row_valid(input, 0, row) {
                out_vec.set_null(row);
                continue;
            }

            let sketch_bytes = get_blob_from_input(input, 0, row);

            match deserialize_sketch(&sketch_bytes) {
                Ok(sketch) => {
                    match sketch.max() {
                        Some(val) => out_vec.as_mut_slice::<f64>()[row] = val,
                        None => out_vec.set_null(row),
                    }
                }
                Err(_) => {
                    out_vec.set_null(row);
                }
            }
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeId::Blob.into()],
            LogicalTypeId::Double.into(),
        )]
    }
}

// ============================================================================
// ddsketch_sum: Get sum value (Scalar Function)
// ============================================================================

struct SumScalar;

impl VScalar for SumScalar {
    type State = ();

    unsafe fn invoke(
        _state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        let mut out_vec = output.flat_vector();

        for row in 0..input.len() {
            // Check if input is NULL
            if !is_row_valid(input, 0, row) {
                out_vec.set_null(row);
                continue;
            }

            let sketch_bytes = get_blob_from_input(input, 0, row);

            match deserialize_sketch(&sketch_bytes) {
                Ok(sketch) => {
                    match sketch.sum() {
                        Some(val) => out_vec.as_mut_slice::<f64>()[row] = val,
                        None => out_vec.set_null(row),
                    }
                }
                Err(_) => {
                    out_vec.set_null(row);
                }
            }
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeId::Blob.into()],
            LogicalTypeId::Double.into(),
        )]
    }
}

// ============================================================================
// ddsketch_avg: Get average value (Scalar Function)
// ============================================================================

struct AvgScalar;

impl VScalar for AvgScalar {
    type State = ();

    unsafe fn invoke(
        _state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        let mut out_vec = output.flat_vector();

        for row in 0..input.len() {
            // Check if input is NULL
            if !is_row_valid(input, 0, row) {
                out_vec.set_null(row);
                continue;
            }

            let sketch_bytes = get_blob_from_input(input, 0, row);

            match deserialize_sketch(&sketch_bytes) {
                Ok(sketch) => {
                    let count = sketch.count();
                    if count > 0 {
                        out_vec.as_mut_slice::<f64>()[row] = sketch.sum().unwrap_or(0.0) / count as f64;
                    } else {
                        out_vec.set_null(row);
                    }
                }
                Err(_) => {
                    out_vec.set_null(row);
                }
            }
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeId::Blob.into()],
            LogicalTypeId::Double.into(),
        )]
    }
}

// ============================================================================
// ddsketch_stats: Get all stats as a struct (Scalar Function)
// ============================================================================

struct StatsScalar;

impl VScalar for StatsScalar {
    type State = ();

    unsafe fn invoke(
        _state: &Self::State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        let mut struct_vec = output.struct_vector();
        let row_count = input.len();

        // Get child vectors for each field
        let mut count_vec = struct_vec.child(0, row_count);
        let mut sum_vec = struct_vec.child(1, row_count);
        let mut min_vec = struct_vec.child(2, row_count);
        let mut max_vec = struct_vec.child(3, row_count);
        let mut avg_vec = struct_vec.child(4, row_count);

        for row in 0..row_count {
            // Check if input is NULL
            if !is_row_valid(input, 0, row) {
                // Set all struct fields as NULL for this row
                struct_vec.set_null(row);
                continue;
            }

            let sketch_bytes = get_blob_from_input(input, 0, row);

            match deserialize_sketch(&sketch_bytes) {
                Ok(sketch) => {
                    let count = sketch.count();
                    let sum = sketch.sum().unwrap_or(f64::NAN);
                    count_vec.as_mut_slice::<i64>()[row] = count as i64;
                    sum_vec.as_mut_slice::<f64>()[row] = sum;
                    min_vec.as_mut_slice::<f64>()[row] = sketch.min().unwrap_or(f64::NAN);
                    max_vec.as_mut_slice::<f64>()[row] = sketch.max().unwrap_or(f64::NAN);
                    avg_vec.as_mut_slice::<f64>()[row] = if count > 0 { sum / count as f64 } else { f64::NAN };
                }
                Err(_) => {
                    struct_vec.set_null(row);
                }
            }
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        // Return type is a struct with 5 fields
        let struct_type = LogicalTypeHandle::struct_type(&[
            ("count", LogicalTypeId::Bigint.into()),
            ("sum", LogicalTypeId::Double.into()),
            ("min", LogicalTypeId::Double.into()),
            ("max", LogicalTypeId::Double.into()),
            ("avg", LogicalTypeId::Double.into()),
        ]);

        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeId::Blob.into()],
            struct_type,
        )]
    }
}

// ============================================================================
// ddsketch_agg: Aggregate function to merge sketches (C API)
// ============================================================================

/// State for the ddsketch_agg aggregate function
/// We store a pointer to a heap-allocated Option<DataDogSketch>
#[repr(C)]
struct SketchAggState {
    sketch: *mut Option<DataDogSketch>,
}

impl SketchAggState {
    fn new() -> Self {
        let sketch = Box::new(None);
        SketchAggState {
            sketch: Box::into_raw(sketch),
        }
    }

    unsafe fn get_sketch(&self) -> &Option<DataDogSketch> {
        &*self.sketch
    }

    unsafe fn get_sketch_mut(&mut self) -> &mut Option<DataDogSketch> {
        &mut *self.sketch
    }

    unsafe fn drop_inner(&mut self) {
        if !self.sketch.is_null() {
            drop(Box::from_raw(self.sketch));
            self.sketch = std::ptr::null_mut();
        }
    }
}

/// Returns the size of the aggregate state
unsafe extern "C" fn sketch_agg_state_size(_info: ffi::duckdb_function_info) -> ffi::idx_t {
    std::mem::size_of::<SketchAggState>() as ffi::idx_t
}

/// Initializes the aggregate state
unsafe extern "C" fn sketch_agg_init(
    _info: ffi::duckdb_function_info,
    state: ffi::duckdb_aggregate_state,
) {
    let state_ptr = state as *mut SketchAggState;
    std::ptr::write(state_ptr, SketchAggState::new());
}

/// Destroys the aggregate states
unsafe extern "C" fn sketch_agg_destroy(
    states: *mut ffi::duckdb_aggregate_state,
    count: ffi::idx_t,
) {
    for i in 0..count as usize {
        let state_ptr = *states.add(i) as *mut SketchAggState;
        if !state_ptr.is_null() {
            (*state_ptr).drop_inner();
        }
    }
}

/// Updates aggregate states with new input values
unsafe extern "C" fn sketch_agg_update(
    _info: ffi::duckdb_function_info,
    input: ffi::duckdb_data_chunk,
    states: *mut ffi::duckdb_aggregate_state,
) {
    let row_count = ffi::duckdb_data_chunk_get_size(input);
    let vector = ffi::duckdb_data_chunk_get_vector(input, 0);
    let data = ffi::duckdb_vector_get_data(vector) as *const ffi::duckdb_string_t;
    let validity = ffi::duckdb_vector_get_validity(vector);

    for row in 0..row_count as usize {
        // Check validity
        let is_valid = if validity.is_null() {
            true
        } else {
            let entry_idx = row / 64;
            let bit_idx = row % 64;
            ((*validity.add(entry_idx)) & (1u64 << bit_idx)) != 0
        };

        if !is_valid {
            continue;
        }

        let state_ptr = *states.add(row) as *mut SketchAggState;
        let state = &mut *state_ptr;

        // Get the blob data using helper functions
        let string_t_ptr = data.add(row) as *mut ffi::duckdb_string_t;
        let blob_len = ffi::duckdb_string_t_length(*string_t_ptr) as usize;

        if blob_len == 0 {
            continue;
        }

        let blob_ptr = ffi::duckdb_string_t_data(string_t_ptr);
        let slice = std::slice::from_raw_parts(blob_ptr as *const u8, blob_len);

        if let Ok(input_sketch) = deserialize_sketch(slice) {
            let current = state.get_sketch_mut();
            match current {
                Some(existing) => {
                    let _ = existing.merge(&input_sketch);
                }
                None => {
                    *current = Some(input_sketch);
                }
            }
        }
    }
}

/// Combines two aggregate states
unsafe extern "C" fn sketch_agg_combine(
    _info: ffi::duckdb_function_info,
    source: *mut ffi::duckdb_aggregate_state,
    target: *mut ffi::duckdb_aggregate_state,
    count: ffi::idx_t,
) {
    for i in 0..count as usize {
        let src_ptr = *source.add(i) as *mut SketchAggState;
        let tgt_ptr = *target.add(i) as *mut SketchAggState;

        let src_state = &*src_ptr;
        let tgt_state = &mut *tgt_ptr;

        if let Some(src_sketch) = src_state.get_sketch() {
            let target_sketch = tgt_state.get_sketch_mut();
            match target_sketch {
                Some(existing) => {
                    let _ = existing.merge(src_sketch);
                }
                None => {
                    *target_sketch = Some(src_sketch.clone());
                }
            }
        }
    }
}

/// Finalizes aggregate states into a result vector
unsafe extern "C" fn sketch_agg_finalize(
    _info: ffi::duckdb_function_info,
    source: *mut ffi::duckdb_aggregate_state,
    result: ffi::duckdb_vector,
    count: ffi::idx_t,
    _offset: ffi::idx_t,
) {
    let validity = ffi::duckdb_vector_get_validity(result);

    for i in 0..count as usize {
        let state_ptr = *source.add(i) as *mut SketchAggState;
        let state = &*state_ptr;

        match state.get_sketch() {
            Some(sketch) => {
                match serialize_sketch(sketch) {
                    Some(bytes) => {
                        ffi::duckdb_vector_assign_string_element_len(
                            result,
                            i as ffi::idx_t,
                            bytes.as_ptr() as *const std::ffi::c_char,
                            bytes.len() as ffi::idx_t,
                        );
                    }
                    None => {
                        // Set NULL on encoding failure
                        ffi::duckdb_validity_set_row_invalid(validity, i as ffi::idx_t);
                    }
                }
            }
            None => {
                // Return NULL for empty aggregation
                ffi::duckdb_validity_set_row_invalid(validity, i as ffi::idx_t);
            }
        }
    }
}

// ============================================================================
// ddsketch_stats_agg: Aggregate returning struct with sketch + percentiles
// ============================================================================

/// Finalizes aggregate states into a struct result vector with sketch, stats, and percentiles
unsafe extern "C" fn sketch_stats_agg_finalize(
    _info: ffi::duckdb_function_info,
    source: *mut ffi::duckdb_aggregate_state,
    result: ffi::duckdb_vector,
    count: ffi::idx_t,
    _offset: ffi::idx_t,
) {
    // Get child vectors from the struct vector
    // Struct order: sketch, count, sum, avg, min, max, p25, p50, p75, p90, p95, p99
    let sketch_vec = ffi::duckdb_struct_vector_get_child(result, 0);
    let count_vec = ffi::duckdb_struct_vector_get_child(result, 1);
    let sum_vec = ffi::duckdb_struct_vector_get_child(result, 2);
    let avg_vec = ffi::duckdb_struct_vector_get_child(result, 3);
    let min_vec = ffi::duckdb_struct_vector_get_child(result, 4);
    let max_vec = ffi::duckdb_struct_vector_get_child(result, 5);
    let p25_vec = ffi::duckdb_struct_vector_get_child(result, 6);
    let p50_vec = ffi::duckdb_struct_vector_get_child(result, 7);
    let p75_vec = ffi::duckdb_struct_vector_get_child(result, 8);
    let p90_vec = ffi::duckdb_struct_vector_get_child(result, 9);
    let p95_vec = ffi::duckdb_struct_vector_get_child(result, 10);
    let p99_vec = ffi::duckdb_struct_vector_get_child(result, 11);

    let count_data = ffi::duckdb_vector_get_data(count_vec) as *mut i64;
    let sum_data = ffi::duckdb_vector_get_data(sum_vec) as *mut f64;
    let avg_data = ffi::duckdb_vector_get_data(avg_vec) as *mut f64;
    let min_data = ffi::duckdb_vector_get_data(min_vec) as *mut f64;
    let max_data = ffi::duckdb_vector_get_data(max_vec) as *mut f64;
    let p25_data = ffi::duckdb_vector_get_data(p25_vec) as *mut f64;
    let p50_data = ffi::duckdb_vector_get_data(p50_vec) as *mut f64;
    let p75_data = ffi::duckdb_vector_get_data(p75_vec) as *mut f64;
    let p90_data = ffi::duckdb_vector_get_data(p90_vec) as *mut f64;
    let p95_data = ffi::duckdb_vector_get_data(p95_vec) as *mut f64;
    let p99_data = ffi::duckdb_vector_get_data(p99_vec) as *mut f64;

    for i in 0..count as usize {
        let state_ptr = *source.add(i) as *mut SketchAggState;
        let state = &*state_ptr;

        match state.get_sketch() {
            Some(sketch) => {
                // Serialize sketch
                match serialize_sketch(sketch) {
                    Some(bytes) => {
                        ffi::duckdb_vector_assign_string_element_len(
                            sketch_vec,
                            i as ffi::idx_t,
                            bytes.as_ptr() as *const std::ffi::c_char,
                            bytes.len() as ffi::idx_t,
                        );
                    }
                    None => {
                        // Set entire struct row as NULL on encoding failure
                        ffi::duckdb_vector_ensure_validity_writable(result);
                        let validity = ffi::duckdb_vector_get_validity(result);
                        ffi::duckdb_validity_set_row_invalid(validity, i as ffi::idx_t);
                        continue;
                    }
                }

                // Compute all stats in one pass (sketch already in memory)
                let cnt = sketch.count();
                let sum = sketch.sum().unwrap_or(f64::NAN);
                *count_data.add(i) = cnt as i64;
                *sum_data.add(i) = sum;
                *avg_data.add(i) = if cnt > 0 { sum / cnt as f64 } else { f64::NAN };
                *min_data.add(i) = sketch.min().unwrap_or(f64::NAN);
                *max_data.add(i) = sketch.max().unwrap_or(f64::NAN);

                // Compute all percentiles
                *p25_data.add(i) = sketch.quantile(0.25).unwrap_or(f64::NAN);
                *p50_data.add(i) = sketch.quantile(0.50).unwrap_or(f64::NAN);
                *p75_data.add(i) = sketch.quantile(0.75).unwrap_or(f64::NAN);
                *p90_data.add(i) = sketch.quantile(0.90).unwrap_or(f64::NAN);
                *p95_data.add(i) = sketch.quantile(0.95).unwrap_or(f64::NAN);
                *p99_data.add(i) = sketch.quantile(0.99).unwrap_or(f64::NAN);
            }
            None => {
                // Return NULL for empty aggregation
                ffi::duckdb_vector_ensure_validity_writable(result);
                let validity = ffi::duckdb_vector_get_validity(result);
                ffi::duckdb_validity_set_row_invalid(validity, i as ffi::idx_t);
            }
        }
    }
}

/// Create a struct logical type for stats_agg return value
unsafe fn create_stats_agg_return_type() -> ffi::duckdb_logical_type {
    // Create child types
    let blob_type = ffi::duckdb_create_logical_type(ffi::DUCKDB_TYPE_DUCKDB_TYPE_BLOB);
    let bigint_type = ffi::duckdb_create_logical_type(ffi::DUCKDB_TYPE_DUCKDB_TYPE_BIGINT);
    let double_type = ffi::duckdb_create_logical_type(ffi::DUCKDB_TYPE_DUCKDB_TYPE_DOUBLE);

    // Field names: sketch, count, sum, avg, min, max, p25, p50, p75, p90, p95, p99
    let names = [
        CString::new("sketch").unwrap(),
        CString::new("count").unwrap(),
        CString::new("sum").unwrap(),
        CString::new("avg").unwrap(),
        CString::new("min").unwrap(),
        CString::new("max").unwrap(),
        CString::new("p25").unwrap(),
        CString::new("p50").unwrap(),
        CString::new("p75").unwrap(),
        CString::new("p90").unwrap(),
        CString::new("p95").unwrap(),
        CString::new("p99").unwrap(),
    ];
    let name_ptrs: Vec<*const std::ffi::c_char> = names.iter().map(|n| n.as_ptr()).collect();

    // Field types
    let types = [
        blob_type,    // sketch
        bigint_type,  // count
        double_type,  // sum
        double_type,  // avg
        double_type,  // min
        double_type,  // max
        double_type,  // p25
        double_type,  // p50
        double_type,  // p75
        double_type,  // p90
        double_type,  // p95
        double_type,  // p99
    ];

    let struct_type = ffi::duckdb_create_struct_type(
        types.as_ptr() as *mut ffi::duckdb_logical_type,
        name_ptrs.as_ptr() as *mut *const std::ffi::c_char,
        12,
    );

    // Cleanup child types
    ffi::duckdb_destroy_logical_type(&mut { blob_type });
    ffi::duckdb_destroy_logical_type(&mut { bigint_type });
    ffi::duckdb_destroy_logical_type(&mut { double_type });

    struct_type
}

/// Register the ddsketch_stats_agg aggregate function
unsafe fn register_sketch_stats_agg(raw_con: ffi::duckdb_connection) -> Result<(), Box<dyn Error>> {
    let agg_func = ffi::duckdb_create_aggregate_function();

    let name = CString::new("ddsketch_stats_agg").unwrap();
    ffi::duckdb_aggregate_function_set_name(agg_func, name.as_ptr());

    // Add parameter (BLOB for serialized sketch)
    let blob_type = ffi::duckdb_create_logical_type(ffi::DUCKDB_TYPE_DUCKDB_TYPE_BLOB);
    ffi::duckdb_aggregate_function_add_parameter(agg_func, blob_type);
    ffi::duckdb_destroy_logical_type(&mut { blob_type });

    // Set return type (STRUCT with sketch and percentiles)
    let return_type = create_stats_agg_return_type();
    ffi::duckdb_aggregate_function_set_return_type(agg_func, return_type);
    ffi::duckdb_destroy_logical_type(&mut { return_type });

    // Set functions - reuse state/init/update/combine from ddsketch_agg
    ffi::duckdb_aggregate_function_set_functions(
        agg_func,
        Some(sketch_agg_state_size),
        Some(sketch_agg_init),
        Some(sketch_agg_update),
        Some(sketch_agg_combine),
        Some(sketch_stats_agg_finalize), // Different finalize
    );

    ffi::duckdb_aggregate_function_set_destructor(agg_func, Some(sketch_agg_destroy));
    ffi::duckdb_aggregate_function_set_special_handling(agg_func);

    let result = ffi::duckdb_register_aggregate_function(raw_con, agg_func);
    ffi::duckdb_destroy_aggregate_function(&mut { agg_func });

    if result != ffi::duckdb_state_DuckDBSuccess {
        return Err("Failed to register ddsketch_stats_agg".into());
    }

    Ok(())
}

/// Register the ddsketch_agg aggregate function using raw connection pointer
unsafe fn register_sketch_agg(raw_con: ffi::duckdb_connection) -> Result<(), Box<dyn Error>> {
    // Create the aggregate function
    let agg_func = ffi::duckdb_create_aggregate_function();

    // Set the name
    let name = CString::new("ddsketch_agg").unwrap();
    ffi::duckdb_aggregate_function_set_name(agg_func, name.as_ptr());

    // Add parameter (BLOB for serialized sketch)
    let blob_type = ffi::duckdb_create_logical_type(ffi::DUCKDB_TYPE_DUCKDB_TYPE_BLOB);
    ffi::duckdb_aggregate_function_add_parameter(agg_func, blob_type);
    ffi::duckdb_destroy_logical_type(&mut { blob_type });

    // Set return type (BLOB for serialized sketch)
    let return_type = ffi::duckdb_create_logical_type(ffi::DUCKDB_TYPE_DUCKDB_TYPE_BLOB);
    ffi::duckdb_aggregate_function_set_return_type(agg_func, return_type);
    ffi::duckdb_destroy_logical_type(&mut { return_type });

    // Set the functions
    ffi::duckdb_aggregate_function_set_functions(
        agg_func,
        Some(sketch_agg_state_size),
        Some(sketch_agg_init),
        Some(sketch_agg_update),
        Some(sketch_agg_combine),
        Some(sketch_agg_finalize),
    );

    // Set destructor
    ffi::duckdb_aggregate_function_set_destructor(agg_func, Some(sketch_agg_destroy));

    // Handle NULLs specially (we want to skip them)
    ffi::duckdb_aggregate_function_set_special_handling(agg_func);

    // Register using the raw connection pointer directly
    let result = ffi::duckdb_register_aggregate_function(raw_con, agg_func);

    // Clean up
    ffi::duckdb_destroy_aggregate_function(&mut { agg_func });

    if result != ffi::duckdb_state_DuckDBSuccess {
        return Err("Failed to register ddsketch_agg".into());
    }

    Ok(())
}

// ============================================================================
// Extension entry point
// ============================================================================

/// Minimum DuckDB C API version required (v1.2.0 is the stable C API)
const MINIMUM_DUCKDB_VERSION: &str = "v1.2.0";

/// Internal entrypoint with error handling
unsafe fn extension_entrypoint_internal(
    info: ffi::duckdb_extension_info,
    access: *const ffi::duckdb_extension_access,
) -> std::result::Result<bool, Box<dyn Error>> {
    // Initialize the API struct
    let have_api_struct = ffi::duckdb_rs_extension_api_init(info, access, MINIMUM_DUCKDB_VERSION)
        .map_err(|e| format!("Failed to init API: {:?}", e))?;

    if !have_api_struct {
        return Ok(false);
    }

    // Get the database from DuckDB
    let db: ffi::duckdb_database = *(*access).get_database.unwrap()(info);

    // Create a connection for aggregate function registration
    let mut raw_con: ffi::duckdb_connection = std::ptr::null_mut();
    if ffi::duckdb_connect(db, &mut raw_con) != ffi::duckdb_state_DuckDBSuccess {
        return Err("Failed to create connection for aggregate registration".into());
    }

    // Register aggregate functions using the raw connection
    register_sketch_agg(raw_con)?;
    register_sketch_stats_agg(raw_con)?;

    // Disconnect the temporary connection
    ffi::duckdb_disconnect(&mut raw_con);

    // Now create the high-level Connection for other registrations
    let con = Connection::open_from_raw(db.cast())?;

    // Register table function for creating sketches
    con.register_table_function::<CreateSketchVTab>("ddsketch_create")
        .expect("Failed to register ddsketch_create");

    // Register scalar functions
    con.register_scalar_function::<AddScalar>("ddsketch_add")
        .expect("Failed to register ddsketch_add");

    con.register_scalar_function::<MergeScalar>("ddsketch_merge")
        .expect("Failed to register ddsketch_merge");

    con.register_scalar_function::<QuantileScalar>("ddsketch_quantile")
        .expect("Failed to register ddsketch_quantile");

    con.register_scalar_function::<CountScalar>("ddsketch_count")
        .expect("Failed to register ddsketch_count");

    con.register_scalar_function::<MinScalar>("ddsketch_min")
        .expect("Failed to register ddsketch_min");

    con.register_scalar_function::<MaxScalar>("ddsketch_max")
        .expect("Failed to register ddsketch_max");

    con.register_scalar_function::<SumScalar>("ddsketch_sum")
        .expect("Failed to register ddsketch_sum");

    con.register_scalar_function::<AvgScalar>("ddsketch_avg")
        .expect("Failed to register ddsketch_avg");

    con.register_scalar_function::<StatsScalar>("ddsketch_stats")
        .expect("Failed to register ddsketch_stats");

    Ok(true)
}

/// Entrypoint called by DuckDB
#[no_mangle]
pub unsafe extern "C" fn ddsketch_init_c_api(
    info: ffi::duckdb_extension_info,
    access: *const ffi::duckdb_extension_access,
) -> bool {
    match extension_entrypoint_internal(info, access) {
        Ok(result) => result,
        Err(e) => {
            // Log error if possible (extension loading failed)
            eprintln!("DDSketch extension init failed: {}", e);
            false
        }
    }
}
