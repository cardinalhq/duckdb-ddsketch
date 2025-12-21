extern crate duckdb;
extern crate duckdb_loadable_macros;
extern crate libduckdb_sys;

use duckdb::{
    core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId},
    vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab},
    vtab::arrow::WritableVector,
    vscalar::{ScalarFunctionSignature, VScalar},
    types::DuckString,
    Connection, Result,
};
use duckdb_loadable_macros::duckdb_entrypoint_c_api;
use libduckdb_sys as ffi;
use sketches_ddsketch::{Config, DDSketch};
use std::{
    error::Error,
    ffi::CString,
};

// ============================================================================
// Serialization helpers for DDSketch
// ============================================================================

/// Serialize a DDSketch to a base64-encoded string
fn serialize_sketch(sketch: &DDSketch) -> String {
    let bytes = bincode::serialize(sketch).expect("Failed to serialize sketch");
    base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes)
}

/// Deserialize a DDSketch from a base64-encoded string
fn deserialize_sketch(encoded: &str) -> std::result::Result<DDSketch, Box<dyn Error>> {
    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, encoded)?;
    let sketch: DDSketch = bincode::deserialize(&bytes)?;
    Ok(sketch)
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
        bind.add_result_column("sketch", LogicalTypeHandle::from(LogicalTypeId::Varchar));

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
            let config = Config::new(bind_data.relative_accuracy, 2048, 1e-9);
            let sketch = DDSketch::new(config);
            let encoded = serialize_sketch(&sketch);

            let vector = output.flat_vector(0);
            let result = CString::new(encoded)?;
            vector.insert(0, result);
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

// Helper to get string from input at index
unsafe fn get_string_from_input(input: &DataChunkHandle, col: usize, row: usize) -> String {
    use ffi::duckdb_string_t;

    let values = input.flat_vector(col);
    let strings = values.as_slice_with_len::<duckdb_string_t>(input.len());
    DuckString::new(&mut { strings[row] }).as_str().to_string()
}

unsafe fn get_double_from_input(input: &DataChunkHandle, col: usize, row: usize) -> f64 {
    let values = input.flat_vector(col);
    let doubles = values.as_slice_with_len::<f64>(input.len());
    doubles[row]
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
        let output = output.flat_vector();

        for row in 0..input.len() {
            let sketch_encoded = get_string_from_input(input, 0, row);
            let value = get_double_from_input(input, 1, row);

            match deserialize_sketch(&sketch_encoded) {
                Ok(mut sketch) => {
                    sketch.add(value);
                    let encoded = serialize_sketch(&sketch);
                    output.insert(row, encoded.as_str());
                }
                Err(_) => {
                    output.insert(row, "");
                }
            }
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![
                LogicalTypeId::Varchar.into(),  // sketch
                LogicalTypeId::Double.into(),   // value
            ],
            LogicalTypeId::Varchar.into(),
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
        let output = output.flat_vector();

        for row in 0..input.len() {
            let sketch1_encoded = get_string_from_input(input, 0, row);
            let sketch2_encoded = get_string_from_input(input, 1, row);

            match (deserialize_sketch(&sketch1_encoded), deserialize_sketch(&sketch2_encoded)) {
                (Ok(mut sketch1), Ok(sketch2)) => {
                    if sketch1.merge(&sketch2).is_ok() {
                        let encoded = serialize_sketch(&sketch1);
                        output.insert(row, encoded.as_str());
                    } else {
                        output.insert(row, "");
                    }
                }
                _ => {
                    output.insert(row, "");
                }
            }
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![
                LogicalTypeId::Varchar.into(),  // sketch1
                LogicalTypeId::Varchar.into(),  // sketch2
            ],
            LogicalTypeId::Varchar.into(),
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
            let sketch_encoded = get_string_from_input(input, 0, row);
            let quantile = get_double_from_input(input, 1, row);

            match deserialize_sketch(&sketch_encoded) {
                Ok(sketch) => {
                    output_slice[row] = sketch.quantile(quantile)
                        .ok()
                        .flatten()
                        .unwrap_or(f64::NAN);
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
                LogicalTypeId::Varchar.into(),  // sketch
                LogicalTypeId::Double.into(),   // quantile (0.0-1.0)
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
            let sketch_encoded = get_string_from_input(input, 0, row);

            match deserialize_sketch(&sketch_encoded) {
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
            vec![LogicalTypeId::Varchar.into()],
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
            let sketch_encoded = get_string_from_input(input, 0, row);

            match deserialize_sketch(&sketch_encoded) {
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
            vec![LogicalTypeId::Varchar.into()],
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
            let sketch_encoded = get_string_from_input(input, 0, row);

            match deserialize_sketch(&sketch_encoded) {
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
            vec![LogicalTypeId::Varchar.into()],
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
            let sketch_encoded = get_string_from_input(input, 0, row);

            match deserialize_sketch(&sketch_encoded) {
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
            vec![LogicalTypeId::Varchar.into()],
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
            let sketch_encoded = get_string_from_input(input, 0, row);

            match deserialize_sketch(&sketch_encoded) {
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
            vec![LogicalTypeId::Varchar.into()],
            LogicalTypeId::Double.into(),
        )]
    }
}

// ============================================================================
// Extension entry point
// ============================================================================

#[duckdb_entrypoint_c_api()]
pub unsafe fn extension_entrypoint(con: Connection) -> Result<(), Box<dyn Error>> {
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

    Ok(())
}
