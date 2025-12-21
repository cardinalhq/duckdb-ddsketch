extern crate duckdb;
extern crate duckdb_loadable_macros;
extern crate libduckdb_sys;

use duckdb::{
    core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId},
    vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab},
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
// ddsketch_create: Create a new empty DDSketch
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
// ddsketch_add: Add values to a sketch
// ============================================================================

#[repr(C)]
struct AddBindData {
    sketch_encoded: String,
    values: Vec<f64>,
}

#[repr(C)]
struct AddInitData {
    done: std::sync::atomic::AtomicBool,
}

struct AddToSketchVTab;

impl VTab for AddToSketchVTab {
    type InitData = AddInitData;
    type BindData = AddBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        bind.add_result_column("sketch", LogicalTypeHandle::from(LogicalTypeId::Varchar));

        let sketch_encoded = bind.get_parameter(0).to_string();
        let value = bind.get_parameter(1).to_string().parse::<f64>().unwrap_or(0.0);

        Ok(AddBindData {
            sketch_encoded,
            values: vec![value],
        })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(AddInitData {
            done: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();
        let bind_data = func.get_bind_data();

        if init_data.done.swap(true, std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
        } else {
            let mut sketch = deserialize_sketch(&bind_data.sketch_encoded)?;
            for value in &bind_data.values {
                sketch.add(*value);
            }
            let encoded = serialize_sketch(&sketch);

            let vector = output.flat_vector(0);
            let result = CString::new(encoded)?;
            vector.insert(0, result);
            output.set_len(1);
        }
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),  // sketch
            LogicalTypeHandle::from(LogicalTypeId::Double),   // value
        ])
    }
}

// ============================================================================
// ddsketch_merge: Merge two sketches
// ============================================================================

#[repr(C)]
struct MergeBindData {
    sketch1_encoded: String,
    sketch2_encoded: String,
}

#[repr(C)]
struct MergeInitData {
    done: std::sync::atomic::AtomicBool,
}

struct MergeSketchVTab;

impl VTab for MergeSketchVTab {
    type InitData = MergeInitData;
    type BindData = MergeBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        bind.add_result_column("sketch", LogicalTypeHandle::from(LogicalTypeId::Varchar));

        let sketch1_encoded = bind.get_parameter(0).to_string();
        let sketch2_encoded = bind.get_parameter(1).to_string();

        Ok(MergeBindData { sketch1_encoded, sketch2_encoded })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(MergeInitData {
            done: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();
        let bind_data = func.get_bind_data();

        if init_data.done.swap(true, std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
        } else {
            let mut sketch1 = deserialize_sketch(&bind_data.sketch1_encoded)?;
            let sketch2 = deserialize_sketch(&bind_data.sketch2_encoded)?;
            sketch1.merge(&sketch2)?;
            let encoded = serialize_sketch(&sketch1);

            let vector = output.flat_vector(0);
            let result = CString::new(encoded)?;
            vector.insert(0, result);
            output.set_len(1);
        }
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),  // sketch1
            LogicalTypeHandle::from(LogicalTypeId::Varchar),  // sketch2
        ])
    }
}

// ============================================================================
// ddsketch_quantile: Get a quantile value from a sketch
// ============================================================================

#[repr(C)]
struct QuantileBindData {
    sketch_encoded: String,
    quantile: f64,
}

#[repr(C)]
struct QuantileInitData {
    done: std::sync::atomic::AtomicBool,
}

struct QuantileVTab;

impl VTab for QuantileVTab {
    type InitData = QuantileInitData;
    type BindData = QuantileBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        bind.add_result_column("value", LogicalTypeHandle::from(LogicalTypeId::Double));

        let sketch_encoded = bind.get_parameter(0).to_string();
        let quantile = bind.get_parameter(1).to_string().parse::<f64>().unwrap_or(0.5);

        Ok(QuantileBindData { sketch_encoded, quantile })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(QuantileInitData {
            done: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();
        let bind_data = func.get_bind_data();

        if init_data.done.swap(true, std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
        } else {
            let sketch = deserialize_sketch(&bind_data.sketch_encoded)?;
            let value = sketch.quantile(bind_data.quantile)?.unwrap_or(f64::NAN);

            let mut vector = output.flat_vector(0);
            vector.as_mut_slice::<f64>()[0] = value;
            output.set_len(1);
        }
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),  // sketch
            LogicalTypeHandle::from(LogicalTypeId::Double),   // quantile (0.0-1.0)
        ])
    }
}

// ============================================================================
// ddsketch_count: Get count from a sketch
// ============================================================================

#[repr(C)]
struct CountBindData {
    sketch_encoded: String,
}

#[repr(C)]
struct CountInitData {
    done: std::sync::atomic::AtomicBool,
}

struct CountVTab;

impl VTab for CountVTab {
    type InitData = CountInitData;
    type BindData = CountBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        bind.add_result_column("count", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        let sketch_encoded = bind.get_parameter(0).to_string();
        Ok(CountBindData { sketch_encoded })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(CountInitData {
            done: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();
        let bind_data = func.get_bind_data();

        if init_data.done.swap(true, std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
        } else {
            let sketch = deserialize_sketch(&bind_data.sketch_encoded)?;
            let count = sketch.count() as i64;

            let mut vector = output.flat_vector(0);
            vector.as_mut_slice::<i64>()[0] = count;
            output.set_len(1);
        }
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }
}

// ============================================================================
// ddsketch_min: Get min from a sketch
// ============================================================================

#[repr(C)]
struct MinBindData {
    sketch_encoded: String,
}

#[repr(C)]
struct MinInitData {
    done: std::sync::atomic::AtomicBool,
}

struct MinVTab;

impl VTab for MinVTab {
    type InitData = MinInitData;
    type BindData = MinBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        bind.add_result_column("min", LogicalTypeHandle::from(LogicalTypeId::Double));
        let sketch_encoded = bind.get_parameter(0).to_string();
        Ok(MinBindData { sketch_encoded })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(MinInitData {
            done: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();
        let bind_data = func.get_bind_data();

        if init_data.done.swap(true, std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
        } else {
            let sketch = deserialize_sketch(&bind_data.sketch_encoded)?;
            let value = sketch.min().unwrap_or(f64::NAN);

            let mut vector = output.flat_vector(0);
            vector.as_mut_slice::<f64>()[0] = value;
            output.set_len(1);
        }
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }
}

// ============================================================================
// ddsketch_max: Get max from a sketch
// ============================================================================

#[repr(C)]
struct MaxBindData {
    sketch_encoded: String,
}

#[repr(C)]
struct MaxInitData {
    done: std::sync::atomic::AtomicBool,
}

struct MaxVTab;

impl VTab for MaxVTab {
    type InitData = MaxInitData;
    type BindData = MaxBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        bind.add_result_column("max", LogicalTypeHandle::from(LogicalTypeId::Double));
        let sketch_encoded = bind.get_parameter(0).to_string();
        Ok(MaxBindData { sketch_encoded })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(MaxInitData {
            done: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();
        let bind_data = func.get_bind_data();

        if init_data.done.swap(true, std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
        } else {
            let sketch = deserialize_sketch(&bind_data.sketch_encoded)?;
            let value = sketch.max().unwrap_or(f64::NAN);

            let mut vector = output.flat_vector(0);
            vector.as_mut_slice::<f64>()[0] = value;
            output.set_len(1);
        }
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }
}

// ============================================================================
// ddsketch_sum: Get sum from a sketch
// ============================================================================

#[repr(C)]
struct SumBindData {
    sketch_encoded: String,
}

#[repr(C)]
struct SumInitData {
    done: std::sync::atomic::AtomicBool,
}

struct SumVTab;

impl VTab for SumVTab {
    type InitData = SumInitData;
    type BindData = SumBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        bind.add_result_column("sum", LogicalTypeHandle::from(LogicalTypeId::Double));
        let sketch_encoded = bind.get_parameter(0).to_string();
        Ok(SumBindData { sketch_encoded })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(SumInitData {
            done: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();
        let bind_data = func.get_bind_data();

        if init_data.done.swap(true, std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
        } else {
            let sketch = deserialize_sketch(&bind_data.sketch_encoded)?;
            let value = sketch.sum().unwrap_or(f64::NAN);

            let mut vector = output.flat_vector(0);
            vector.as_mut_slice::<f64>()[0] = value;
            output.set_len(1);
        }
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }
}

// ============================================================================
// ddsketch_avg: Get average from a sketch
// ============================================================================

#[repr(C)]
struct AvgBindData {
    sketch_encoded: String,
}

#[repr(C)]
struct AvgInitData {
    done: std::sync::atomic::AtomicBool,
}

struct AvgVTab;

impl VTab for AvgVTab {
    type InitData = AvgInitData;
    type BindData = AvgBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        bind.add_result_column("avg", LogicalTypeHandle::from(LogicalTypeId::Double));
        let sketch_encoded = bind.get_parameter(0).to_string();
        Ok(AvgBindData { sketch_encoded })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(AvgInitData {
            done: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();
        let bind_data = func.get_bind_data();

        if init_data.done.swap(true, std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
        } else {
            let sketch = deserialize_sketch(&bind_data.sketch_encoded)?;
            let count = sketch.count();
            let value = if count > 0 {
                sketch.sum().unwrap_or(0.0) / count as f64
            } else {
                f64::NAN
            };

            let mut vector = output.flat_vector(0);
            vector.as_mut_slice::<f64>()[0] = value;
            output.set_len(1);
        }
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }
}

// ============================================================================
// Extension entry point
// ============================================================================

const EXTENSION_NAME: &str = "ddsketch";

#[duckdb_entrypoint_c_api()]
pub unsafe fn extension_entrypoint(con: Connection) -> Result<(), Box<dyn Error>> {
    // Register table functions
    con.register_table_function::<CreateSketchVTab>("ddsketch_create")
        .expect("Failed to register ddsketch_create");

    con.register_table_function::<AddToSketchVTab>("ddsketch_add")
        .expect("Failed to register ddsketch_add");

    con.register_table_function::<MergeSketchVTab>("ddsketch_merge")
        .expect("Failed to register ddsketch_merge");

    con.register_table_function::<QuantileVTab>("ddsketch_quantile")
        .expect("Failed to register ddsketch_quantile");

    con.register_table_function::<CountVTab>("ddsketch_count")
        .expect("Failed to register ddsketch_count");

    con.register_table_function::<MinVTab>("ddsketch_min")
        .expect("Failed to register ddsketch_min");

    con.register_table_function::<MaxVTab>("ddsketch_max")
        .expect("Failed to register ddsketch_max");

    con.register_table_function::<SumVTab>("ddsketch_sum")
        .expect("Failed to register ddsketch_sum");

    con.register_table_function::<AvgVTab>("ddsketch_avg")
        .expect("Failed to register ddsketch_avg");

    Ok(())
}
