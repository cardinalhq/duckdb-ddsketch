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
fn serialize_sketch(sketch: &DataDogSketch) -> Vec<u8> {
    sketch.encode().expect("Failed to serialize sketch")
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
            let bytes = serialize_sketch(&sketch);

            // Use Inserter<&[u8]> to assign blob data
            let vector = output.flat_vector(0);
            vector.insert(0, bytes.as_slice());
            output.set_len(1);
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

// Helper to get blob data from input at index
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
            let sketch_bytes = get_blob_from_input(input, 0, row);
            let value = get_double_from_input(input, 1, row);

            match deserialize_sketch(&sketch_bytes) {
                Ok(mut sketch) => {
                    sketch.add(value);
                    let bytes = serialize_sketch(&sketch);
                    set_blob_in_output(output, row, &bytes);
                }
                Err(_) => {
                    set_blob_in_output(output, row, &[]);
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
            let sketch1_bytes = get_blob_from_input(input, 0, row);
            let sketch2_bytes = get_blob_from_input(input, 1, row);

            match (deserialize_sketch(&sketch1_bytes), deserialize_sketch(&sketch2_bytes)) {
                (Ok(mut sketch1), Ok(sketch2)) => {
                    if sketch1.merge(&sketch2).is_ok() {
                        let bytes = serialize_sketch(&sketch1);
                        set_blob_in_output(output, row, &bytes);
                    } else {
                        set_blob_in_output(output, row, &[]);
                    }
                }
                _ => {
                    set_blob_in_output(output, row, &[]);
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
        let mut output = output.flat_vector();
        let output_slice = output.as_mut_slice::<f64>();

        for row in 0..input.len() {
            let sketch_bytes = get_blob_from_input(input, 0, row);
            let quantile = get_double_from_input(input, 1, row);

            match deserialize_sketch(&sketch_bytes) {
                Ok(sketch) => {
                    output_slice[row] = sketch.quantile(quantile).unwrap_or(f64::NAN);
                }
                Err(_) => {
                    output_slice[row] = f64::NAN;
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
        let mut output = output.flat_vector();
        let output_slice = output.as_mut_slice::<i64>();

        for row in 0..input.len() {
            let sketch_bytes = get_blob_from_input(input, 0, row);

            match deserialize_sketch(&sketch_bytes) {
                Ok(sketch) => {
                    output_slice[row] = sketch.count() as i64;
                }
                Err(_) => {
                    output_slice[row] = 0;
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
        let mut output = output.flat_vector();
        let output_slice = output.as_mut_slice::<f64>();

        for row in 0..input.len() {
            let sketch_bytes = get_blob_from_input(input, 0, row);

            match deserialize_sketch(&sketch_bytes) {
                Ok(sketch) => {
                    output_slice[row] = sketch.min().unwrap_or(f64::NAN);
                }
                Err(_) => {
                    output_slice[row] = f64::NAN;
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
        let mut output = output.flat_vector();
        let output_slice = output.as_mut_slice::<f64>();

        for row in 0..input.len() {
            let sketch_bytes = get_blob_from_input(input, 0, row);

            match deserialize_sketch(&sketch_bytes) {
                Ok(sketch) => {
                    output_slice[row] = sketch.max().unwrap_or(f64::NAN);
                }
                Err(_) => {
                    output_slice[row] = f64::NAN;
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
        let mut output = output.flat_vector();
        let output_slice = output.as_mut_slice::<f64>();

        for row in 0..input.len() {
            let sketch_bytes = get_blob_from_input(input, 0, row);

            match deserialize_sketch(&sketch_bytes) {
                Ok(sketch) => {
                    output_slice[row] = sketch.sum().unwrap_or(f64::NAN);
                }
                Err(_) => {
                    output_slice[row] = f64::NAN;
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
        let mut output = output.flat_vector();
        let output_slice = output.as_mut_slice::<f64>();

        for row in 0..input.len() {
            let sketch_bytes = get_blob_from_input(input, 0, row);

            match deserialize_sketch(&sketch_bytes) {
                Ok(sketch) => {
                    let count = sketch.count();
                    output_slice[row] = if count > 0 {
                        sketch.sum().unwrap_or(0.0) / count as f64
                    } else {
                        f64::NAN
                    };
                }
                Err(_) => {
                    output_slice[row] = f64::NAN;
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
    for i in 0..count as usize {
        let state_ptr = *source.add(i) as *mut SketchAggState;
        let state = &*state_ptr;

        match state.get_sketch() {
            Some(sketch) => {
                let bytes = serialize_sketch(sketch);
                ffi::duckdb_vector_assign_string_element_len(
                    result,
                    i as ffi::idx_t,
                    bytes.as_ptr() as *const std::ffi::c_char,
                    bytes.len() as ffi::idx_t,
                );
            }
            None => {
                // Return NULL for empty aggregation
                ffi::duckdb_vector_assign_string_element_len(
                    result,
                    i as ffi::idx_t,
                    std::ptr::null(),
                    0,
                );
            }
        }
    }
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

    // Register aggregate function using the raw connection
    register_sketch_agg(raw_con)?;

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
