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

// DataDog DDSketch wire format encoding/decoding
// Based on github.com/DataDog/sketches-go v1.4.7
//
// Wire format:
// - Flag byte(s): 2 LSB = type, 6 MSB = subflag
// - Index mapping: gamma, indexOffset
// - Stores: positive, negative, zero count
// - Each store uses bin encoding with delta-encoded indices

use std::error::Error;
use std::io::{Read, Write};

// ============================================================================
// Flag Types and Subflags (from flag.go)
// ============================================================================

/// Flag type is encoded in the 2 least significant bits
/// From Go: flagTypeSketchFeatures=0b00, FlagTypePositiveStore=0b01, FlagTypeIndexMapping=0b10, FlagTypeNegativeStore=0b11
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum FlagType {
    SketchFeatures = 0b00,
    PositiveStore = 0b01,
    IndexMapping = 0b10,
    NegativeStore = 0b11,
}

impl FlagType {
    fn from_byte(b: u8) -> Self {
        match b & 0b11 {
            0b00 => FlagType::SketchFeatures,
            0b01 => FlagType::PositiveStore,
            0b10 => FlagType::IndexMapping,
            0b11 => FlagType::NegativeStore,
            _ => unreachable!(),
        }
    }
}

/// Sketch feature subflags (from Go flag.go)
/// FlagZeroCountVarFloat = newSubFlag(1)
/// FlagSum = newSubFlag(0x21)
/// FlagCount = newSubFlag(0x28)
/// FlagMin = newSubFlag(0x22)
/// FlagMax = newSubFlag(0x23)
#[repr(u8)]
#[allow(dead_code)]
pub enum SketchFeatureSubflag {
    ZeroCount = 1,
    Sum = 0x21,     // 33
    Min = 0x22,     // 34
    Max = 0x23,     // 35
    Count = 0x28,   // 40
}

/// Index mapping subflags (all variants needed for decoding compatibility)
#[repr(u8)]
#[allow(dead_code)]
pub enum IndexMappingSubflag {
    LogarithmicMapping = 0,
    LinearMapping = 1,
    CubicallyInterpolatedMapping = 2,
}

/// Store bin encoding subflags (all variants needed for decoding compatibility)
#[repr(u8)]
#[allow(dead_code)]
pub enum BinEncodingSubflag {
    IndexDeltasAndCounts = 1,
    IndexDeltas = 2,
    ContiguousCounts = 3,
}

/// Build a flag byte from type and subflag
fn make_flag(flag_type: FlagType, subflag: u8) -> u8 {
    (subflag << 2) | (flag_type as u8)
}

/// Extract subflag from flag byte
fn get_subflag(flag: u8) -> u8 {
    flag >> 2
}

// ============================================================================
// Varint Encoding (from encoding.go)
// ============================================================================

/// Encode unsigned varint (7 bits per byte, MSB = continuation bit)
pub fn encode_uvarint64<W: Write>(w: &mut W, mut value: u64) -> std::io::Result<()> {
    while value >= 0x80 {
        w.write_all(&[(value as u8) | 0x80])?;
        value >>= 7;
    }
    w.write_all(&[value as u8])
}

/// Decode unsigned varint
pub fn decode_uvarint64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut result: u64 = 0;
    let mut shift = 0;
    let mut buf = [0u8; 1];

    loop {
        r.read_exact(&mut buf)?;
        let byte = buf[0];
        result |= ((byte & 0x7F) as u64) << shift;
        if byte < 0x80 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "varint overflow",
            ));
        }
    }

    Ok(result)
}

/// Encode signed varint using zig-zag encoding
pub fn encode_varint64<W: Write>(w: &mut W, value: i64) -> std::io::Result<()> {
    // Zig-zag: (value << 1) ^ (value >> 63)
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    encode_uvarint64(w, zigzag)
}

/// Decode signed varint (zig-zag)
pub fn decode_varint64<R: Read>(r: &mut R) -> std::io::Result<i64> {
    let zigzag = decode_uvarint64(r)?;
    // Reverse zig-zag: (zigzag >> 1) ^ (-(zigzag & 1))
    Ok(((zigzag >> 1) as i64) ^ (-((zigzag & 1) as i64)))
}

const VARFLOAT64_ROTATE: u32 = 6;
const MAX_VAR_LEN_64: usize = 9;

/// Encode float64 using varfloat encoding (DataDog format)
/// This is optimized for non-negative integer values.
/// The encoding:
/// 1. Shifts the float (+1) to make small integers have trailing zeros
/// 2. Subtracts Float64bits(1) to normalize
/// 3. Rotates left by 6 bits
/// 4. Encodes MSB-first with 7 bits per byte
pub fn encode_varfloat64<W: Write>(w: &mut W, v: f64) -> std::io::Result<()> {
    let float_bits_1 = 1.0f64.to_bits();
    let mut x = ((v + 1.0).to_bits().wrapping_sub(float_bits_1)).rotate_left(VARFLOAT64_ROTATE);

    for _ in 0..MAX_VAR_LEN_64 - 1 {
        let n = (x >> (64 - 7)) as u8;
        x <<= 7;
        if x == 0 {
            w.write_all(&[n])?;
            return Ok(());
        }
        w.write_all(&[n | 0x80])?;
    }
    let n = (x >> (8 * 7)) as u8;
    w.write_all(&[n])?;
    Ok(())
}

/// Decode varfloat64 (DataDog format)
pub fn decode_varfloat64<R: Read>(r: &mut R) -> std::io::Result<f64> {
    let mut x: u64 = 0;
    let mut s: u32 = 64 - 7;
    let mut buf = [0u8; 1];

    for i in 0..MAX_VAR_LEN_64 {
        r.read_exact(&mut buf)?;
        let n = buf[0];

        if i == MAX_VAR_LEN_64 - 1 {
            x |= n as u64;
            break;
        }

        if n < 0x80 {
            x |= (n as u64) << s;
            break;
        }

        x |= ((n & 0x7F) as u64) << s;
        s = s.saturating_sub(7);
    }

    let float_bits_1 = 1.0f64.to_bits();
    let bits = x.rotate_right(VARFLOAT64_ROTATE).wrapping_add(float_bits_1);
    Ok(f64::from_bits(bits) - 1.0)
}

/// Encode float64 as little-endian 8 bytes
pub fn encode_float64_le<W: Write>(w: &mut W, value: f64) -> std::io::Result<()> {
    w.write_all(&value.to_le_bytes())
}

/// Decode float64 from little-endian 8 bytes
pub fn decode_float64_le<R: Read>(r: &mut R) -> std::io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

// ============================================================================
// DDSketch Structure for DataDog format
// ============================================================================

/// Represents a DDSketch in DataDog-compatible format
#[derive(Debug, Clone)]
pub struct DataDogSketch {
    /// Gamma for logarithmic mapping (derived from relative accuracy)
    pub gamma: f64,
    /// Index offset for the mapping
    pub index_offset: f64,
    /// Positive value bins: (index, count)
    pub positive_bins: Vec<(i32, f64)>,
    /// Negative value bins: (index, count)
    pub negative_bins: Vec<(i32, f64)>,
    /// Count of zero values
    pub zero_count: f64,
    /// Sum of all values
    pub sum: f64,
    /// Total count
    pub count: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
}

impl Default for DataDogSketch {
    fn default() -> Self {
        DataDogSketch {
            // gamma for 1% relative accuracy
            gamma: 1.0 + 2.0 * 0.01 / (1.0 - 0.01),
            index_offset: 0.0,
            positive_bins: Vec::new(),
            negative_bins: Vec::new(),
            zero_count: 0.0,
            sum: 0.0,
            count: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
}

impl DataDogSketch {
    /// Create new sketch with given relative accuracy
    pub fn new(relative_accuracy: f64) -> Self {
        DataDogSketch {
            gamma: 1.0 + 2.0 * relative_accuracy / (1.0 - relative_accuracy),
            index_offset: 0.0,
            ..Default::default()
        }
    }

    /// Get the total count
    pub fn count(&self) -> usize {
        self.count as usize
    }

    /// Get the sum (returns None if empty)
    pub fn sum(&self) -> Option<f64> {
        if self.count > 0.0 {
            Some(self.sum)
        } else {
            None
        }
    }

    /// Get the minimum value (returns None if empty)
    pub fn min(&self) -> Option<f64> {
        if self.count > 0.0 && self.min.is_finite() {
            Some(self.min)
        } else {
            None
        }
    }

    /// Get the maximum value (returns None if empty)
    pub fn max(&self) -> Option<f64> {
        if self.count > 0.0 && self.max.is_finite() {
            Some(self.max)
        } else {
            None
        }
    }

    /// Encode the sketch to bytes in DataDog format
    pub fn encode(&self) -> Result<Vec<u8>, Box<dyn Error>> {
        let mut buf = Vec::new();
        self.encode_to(&mut buf)?;
        Ok(buf)
    }

    /// Encode to a writer
    pub fn encode_to<W: Write>(&self, w: &mut W) -> Result<(), Box<dyn Error>> {
        // 1. Write index mapping
        self.encode_index_mapping(w)?;

        // 2. Write positive store
        if !self.positive_bins.is_empty() {
            self.encode_store(w, FlagType::PositiveStore, &self.positive_bins)?;
        }

        // 3. Write negative store
        if !self.negative_bins.is_empty() {
            self.encode_store(w, FlagType::NegativeStore, &self.negative_bins)?;
        }

        // 4. Write zero count if present
        if self.zero_count > 0.0 {
            let flag = make_flag(FlagType::SketchFeatures, SketchFeatureSubflag::ZeroCount as u8);
            w.write_all(&[flag])?;
            encode_varfloat64(w, self.zero_count)?;
        }

        // Note: We intentionally do NOT write Sum/Count/Min/Max feature flags.
        // Go's decoder has a bug where it expects 8 bytes for FlagCount but
        // the spec says varfloat64. Go never writes these flags itself - it
        // computes stats from bins on decode. For compatibility, we do the same.

        Ok(())
    }

    fn encode_index_mapping<W: Write>(&self, w: &mut W) -> Result<(), Box<dyn Error>> {
        let flag = make_flag(FlagType::IndexMapping, IndexMappingSubflag::LogarithmicMapping as u8);
        w.write_all(&[flag])?;
        // Index mapping uses float64LE, not varfloat64
        encode_float64_le(w, self.gamma)?;
        encode_float64_le(w, self.index_offset)?;
        Ok(())
    }

    fn encode_store<W: Write>(
        &self,
        w: &mut W,
        flag_type: FlagType,
        bins: &[(i32, f64)],
    ) -> Result<(), Box<dyn Error>> {
        if bins.is_empty() {
            return Ok(());
        }

        // Use IndexDeltasAndCounts encoding (most general)
        let flag = make_flag(flag_type, BinEncodingSubflag::IndexDeltasAndCounts as u8);
        w.write_all(&[flag])?;

        // Write number of bins
        encode_uvarint64(w, bins.len() as u64)?;

        // Write delta-encoded indices and counts
        let mut prev_index: i32 = 0;
        for (index, count) in bins {
            let delta = *index - prev_index;
            encode_varint64(w, delta as i64)?;
            encode_varfloat64(w, *count)?;
            prev_index = *index;
        }

        Ok(())
    }

    /// Decode a sketch from bytes
    pub fn decode(data: &[u8]) -> Result<Self, Box<dyn Error>> {
        let mut cursor = std::io::Cursor::new(data);
        Self::decode_from(&mut cursor)
    }

    /// Decode from a reader
    pub fn decode_from<R: Read>(r: &mut R) -> Result<Self, Box<dyn Error>> {
        let mut sketch = DataDogSketch::default();
        let mut buf = [0u8; 1];
        let mut has_explicit_count = false;
        let mut has_explicit_sum = false;

        loop {
            // Try to read a flag byte
            match r.read_exact(&mut buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }

            let flag = buf[0];
            let flag_type = FlagType::from_byte(flag);
            let subflag = get_subflag(flag);

            match flag_type {
                FlagType::IndexMapping => {
                    sketch.decode_index_mapping(r, subflag)?;
                }
                FlagType::PositiveStore => {
                    sketch.positive_bins = sketch.decode_store(r, subflag)?;
                }
                FlagType::NegativeStore => {
                    sketch.negative_bins = sketch.decode_store(r, subflag)?;
                }
                FlagType::SketchFeatures => {
                    if subflag == 0x28 {
                        has_explicit_count = true;
                    }
                    if subflag == 0x21 {
                        has_explicit_sum = true;
                    }
                    sketch.decode_feature(r, subflag)?;
                }
            }
        }

        // If count/sum weren't explicitly encoded, compute from bins
        // This is how Go's DDSketch.GetCount() and GetSum() work
        if !has_explicit_count {
            sketch.count = sketch.compute_count_from_bins();
        }
        if !has_explicit_sum {
            sketch.sum = sketch.compute_sum_from_bins();
        }
        // Compute min/max from bins if not explicitly set
        if !sketch.min.is_finite() || !sketch.max.is_finite() {
            sketch.compute_min_max_from_bins();
        }

        Ok(sketch)
    }

    /// Compute total count from bins (how Go's GetCount works)
    fn compute_count_from_bins(&self) -> f64 {
        let pos_count: f64 = self.positive_bins.iter().map(|(_, c)| c).sum();
        let neg_count: f64 = self.negative_bins.iter().map(|(_, c)| c).sum();
        pos_count + neg_count + self.zero_count
    }

    /// Compute sum from bins using bin midpoints (how Go's GetSum works)
    fn compute_sum_from_bins(&self) -> f64 {
        let mut sum = 0.0;
        for (index, count) in &self.positive_bins {
            sum += self.bin_to_value(*index) * count;
        }
        for (index, count) in &self.negative_bins {
            sum -= self.bin_to_value(*index) * count;
        }
        // zero count contributes 0 to sum
        sum
    }

    /// Compute min/max from bins
    fn compute_min_max_from_bins(&mut self) {
        // For min: check negative bins (highest index = closest to 0), then zero, then positive (lowest index)
        // For max: check positive bins (highest index), then zero, then negative (lowest index = most negative)

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for (index, count) in &self.negative_bins {
            if *count > 0.0 {
                let value = -self.bin_to_value(*index);
                if value < min { min = value; }
                if value > max { max = value; }
            }
        }

        if self.zero_count > 0.0 {
            if 0.0 < min { min = 0.0; }
            if 0.0 > max { max = 0.0; }
        }

        for (index, count) in &self.positive_bins {
            if *count > 0.0 {
                let value = self.bin_to_value(*index);
                if value < min { min = value; }
                if value > max { max = value; }
            }
        }

        if min.is_finite() { self.min = min; }
        if max.is_finite() { self.max = max; }
    }

    fn decode_index_mapping<R: Read>(&mut self, r: &mut R, subflag: u8) -> Result<(), Box<dyn Error>> {
        // All index mapping types use float64LE for gamma and indexOffset
        match subflag {
            0 | 1 | 2 | 3 | 4 => {
                // 0=Logarithmic, 1=Linear, 2=Quadratic, 3=Cubic, 4=Quartic
                self.gamma = decode_float64_le(r)?;
                self.index_offset = decode_float64_le(r)?;
            }
            _ => {
                return Err(format!("Unknown index mapping subflag: {}", subflag).into());
            }
        }
        Ok(())
    }

    fn decode_store<R: Read>(&self, r: &mut R, subflag: u8) -> Result<Vec<(i32, f64)>, Box<dyn Error>> {
        match subflag {
            1 => {
                // IndexDeltasAndCounts
                let num_bins = decode_uvarint64(r)? as usize;
                let mut bins = Vec::with_capacity(num_bins);
                let mut prev_index: i32 = 0;

                for _ in 0..num_bins {
                    let delta = decode_varint64(r)? as i32;
                    let index = prev_index + delta;
                    let count = decode_varfloat64(r)?;
                    bins.push((index, count));
                    prev_index = index;
                }

                Ok(bins)
            }
            2 => {
                // IndexDeltas (count = 1 for each)
                let num_bins = decode_uvarint64(r)? as usize;
                let mut bins = Vec::with_capacity(num_bins);
                let mut prev_index: i32 = 0;

                for _ in 0..num_bins {
                    let delta = decode_varint64(r)? as i32;
                    let index = prev_index + delta;
                    bins.push((index, 1.0));
                    prev_index = index;
                }

                Ok(bins)
            }
            3 => {
                // ContiguousCounts
                // Format: numBins, startIndex, indexDelta, count1, count2, ...
                let num_bins = decode_uvarint64(r)? as usize;
                let start_index = decode_varint64(r)? as i32;
                let index_delta = decode_varint64(r)? as i32;
                let mut bins = Vec::with_capacity(num_bins);

                let mut index = start_index;
                for _ in 0..num_bins {
                    let count = decode_varfloat64(r)?;
                    bins.push((index, count));
                    index += index_delta;
                }

                Ok(bins)
            }
            _ => {
                Err(format!("Unknown bin encoding subflag: {}", subflag).into())
            }
        }
    }

    fn decode_feature<R: Read>(&mut self, r: &mut R, subflag: u8) -> Result<(), Box<dyn Error>> {
        match subflag {
            1 => {
                // ZeroCount (varfloat64)
                self.zero_count = decode_varfloat64(r)?;
            }
            0x21 => {
                // Sum (float64LE)
                self.sum = decode_float64_le(r)?;
            }
            0x22 => {
                // Min (float64LE)
                self.min = decode_float64_le(r)?;
            }
            0x23 => {
                // Max (float64LE)
                self.max = decode_float64_le(r)?;
            }
            0x28 => {
                // Count (varfloat64)
                self.count = decode_varfloat64(r)?;
            }
            _ => {
                // Unknown feature, skip it if we can
                // For now, just ignore unknown features
            }
        }
        Ok(())
    }

    /// Merge another sketch into this one
    pub fn merge(&mut self, other: &DataDogSketch) -> Result<(), Box<dyn Error>> {
        // Check gamma compatibility (must be same mapping)
        if (self.gamma - other.gamma).abs() > 1e-10 {
            return Err("Cannot merge sketches with different gamma values".into());
        }

        // Check index_offset compatibility (must be same to avoid silent corruption)
        if (self.index_offset - other.index_offset).abs() > 1e-10 {
            return Err("Cannot merge sketches with different index_offset values".into());
        }

        // Merge positive bins
        self.positive_bins = Self::merge_bins_impl(&self.positive_bins, &other.positive_bins);

        // Merge negative bins
        self.negative_bins = Self::merge_bins_impl(&self.negative_bins, &other.negative_bins);

        // Merge zero count
        self.zero_count += other.zero_count;

        // Merge sum and count
        self.sum += other.sum;
        self.count += other.count;

        // Update min/max
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);

        Ok(())
    }

    fn merge_bins_impl(bins1: &[(i32, f64)], bins2: &[(i32, f64)]) -> Vec<(i32, f64)> {
        // Simple merge: combine bins with same index, append others
        use std::collections::BTreeMap;

        let mut bin_map: BTreeMap<i32, f64> = BTreeMap::new();

        for (idx, count) in bins1.iter() {
            *bin_map.entry(*idx).or_insert(0.0) += count;
        }

        for (idx, count) in bins2.iter() {
            *bin_map.entry(*idx).or_insert(0.0) += count;
        }

        bin_map.into_iter().collect()
    }

    /// Get quantile value
    /// Matches Go's DDSketch.GetValueAtQuantile() exactly:
    /// - rank = quantile * (count - 1)
    /// - Uses ">" not ">=" for rank comparison (KeyAtRank uses n > rank)
    /// - Negative bins are searched with reversed rank (negativeValueCount - 1 - rank)
    pub fn quantile(&self, q: f64) -> Option<f64> {
        if self.count == 0.0 {
            return None;
        }

        if q < 0.0 || q > 1.0 {
            return None;
        }

        // Use Go's formula: rank = quantile * (count - 1)
        // Explicit f64 conversion to prevent FMA operations (matches Go)
        let rank: f64 = q * (self.count - 1.0);

        let negative_count: f64 = self.negative_bins.iter().map(|(_, c)| c).sum();

        if rank < negative_count {
            // Rank falls within negative store
            // Go reverses the rank: negativeValueCount - 1 - rank
            let neg_rank = negative_count - 1.0 - rank;
            return Some(-self.key_at_rank(&self.negative_bins, neg_rank));
        }

        if rank < negative_count + self.zero_count {
            // Rank falls within zero bucket
            return Some(0.0);
        }

        // Rank falls within positive store
        // Adjust rank for positive store
        let pos_rank = rank - self.zero_count - negative_count;
        Some(self.key_at_rank(&self.positive_bins, pos_rank))
    }

    /// Find the bin value at the given rank within a store
    /// Matches Go's Store.KeyAtRank() exactly - uses ">" not ">="
    fn key_at_rank(&self, bins: &[(i32, f64)], rank: f64) -> f64 {
        let rank = if rank < 0.0 { 0.0 } else { rank };
        let mut cumulative = 0.0;

        for (index, count) in bins {
            cumulative += count;
            if cumulative > rank {
                return self.bin_to_value(*index);
            }
        }

        // Return value at max index if rank exceeds total count
        if let Some((max_idx, _)) = bins.last() {
            self.bin_to_value(*max_idx)
        } else {
            0.0
        }
    }

    /// Convert bin index to value using logarithmic mapping
    /// Go's formula: Value(index) = LowerBound(index) * (1 + RelativeAccuracy)
    /// where LowerBound = gamma^(index - indexOffset)
    /// and RelativeAccuracy = 1 - 2/(1+gamma)
    fn bin_to_value(&self, index: i32) -> f64 {
        let adjusted = index as f64 - self.index_offset;
        let lower_bound = self.gamma.powf(adjusted);
        // RelativeAccuracy = 1 - 2/(1+gamma)
        let relative_accuracy = 1.0 - 2.0 / (1.0 + self.gamma);
        lower_bound * (1.0 + relative_accuracy)
    }

    /// Add a value to the sketch
    pub fn add(&mut self, value: f64) {
        self.add_with_count(value, 1.0);
    }

    /// Add a value with a specific count
    pub fn add_with_count(&mut self, value: f64, count: f64) {
        if count <= 0.0 {
            return;
        }

        self.count += count;
        self.sum += value * count;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }

        if value == 0.0 {
            self.zero_count += count;
        } else if value > 0.0 {
            let index = self.value_to_bin(value);
            Self::add_to_bin(&mut self.positive_bins, index, count);
        } else {
            let index = self.value_to_bin(-value);
            Self::add_to_bin(&mut self.negative_bins, index, count);
        }
    }

    /// Convert value to bin index using logarithmic mapping
    fn value_to_bin(&self, value: f64) -> i32 {
        let log_gamma = self.gamma.ln();
        (value.ln() / log_gamma + self.index_offset).ceil() as i32
    }

    fn add_to_bin(target: &mut Vec<(i32, f64)>, index: i32, count: f64) {
        // Find or insert bin
        match target.binary_search_by_key(&index, |(i, _)| *i) {
            Ok(pos) => {
                target[pos].1 += count;
            }
            Err(pos) => {
                target.insert(pos, (index, count));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // DataDog test vectors from encoding_test.go
    #[test]
    fn test_uvarint_datadog_vectors() {
        // Test vectors from sketches-go encoding_test.go
        let test_cases: Vec<(u64, Vec<u8>)> = vec![
            (0, vec![0x00]),
            (127, vec![0x7F]),
            (128, vec![0x80, 0x01]),
            (u64::MAX, vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]),
        ];

        for (value, expected_bytes) in test_cases {
            let mut buf = Vec::new();
            encode_uvarint64(&mut buf, value).unwrap();
            assert_eq!(buf, expected_bytes, "uvarint encoding mismatch for {}", value);

            let mut cursor = std::io::Cursor::new(&expected_bytes);
            let decoded = decode_uvarint64(&mut cursor).unwrap();
            assert_eq!(decoded, value, "uvarint decoding mismatch for {:?}", expected_bytes);
        }
    }

    #[test]
    fn test_varint_datadog_vectors() {
        // Test vectors from sketches-go encoding_test.go
        // Zig-zag encoding: 0->0, 1->2, -1->1, -64->127
        let test_cases: Vec<(i64, Vec<u8>)> = vec![
            (0, vec![0x00]),
            (1, vec![0x02]),
            (-1, vec![0x01]),
            (-64, vec![0x7F]),
            (i64::MAX, vec![0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]),
        ];

        for (value, expected_bytes) in test_cases {
            let mut buf = Vec::new();
            encode_varint64(&mut buf, value).unwrap();
            assert_eq!(buf, expected_bytes, "varint encoding mismatch for {}", value);

            let mut cursor = std::io::Cursor::new(&expected_bytes);
            let decoded = decode_varint64(&mut cursor).unwrap();
            assert_eq!(decoded, value, "varint decoding mismatch for {:?}", expected_bytes);
        }
    }

    #[test]
    fn test_uvarint_roundtrip() {
        let values = [0u64, 1, 127, 128, 16383, 16384, u64::MAX];

        for &v in &values {
            let mut buf = Vec::new();
            encode_uvarint64(&mut buf, v).unwrap();

            let mut cursor = std::io::Cursor::new(&buf);
            let decoded = decode_uvarint64(&mut cursor).unwrap();

            assert_eq!(v, decoded, "uvarint roundtrip failed for {}", v);
        }
    }

    #[test]
    fn test_varint_roundtrip() {
        let values = [0i64, 1, -1, 63, -64, 64, -65, i64::MIN, i64::MAX];

        for &v in &values {
            let mut buf = Vec::new();
            encode_varint64(&mut buf, v).unwrap();

            let mut cursor = std::io::Cursor::new(&buf);
            let decoded = decode_varint64(&mut cursor).unwrap();

            assert_eq!(v, decoded, "varint roundtrip failed for {}", v);
        }
    }

    #[test]
    fn test_varfloat_roundtrip() {
        let values = [0.0, 1.0, -1.0, std::f64::consts::PI, f64::MIN, f64::MAX];

        for &v in &values {
            let mut buf = Vec::new();
            encode_varfloat64(&mut buf, v).unwrap();

            let mut cursor = std::io::Cursor::new(&buf);
            let decoded = decode_varfloat64(&mut cursor).unwrap();

            assert!((v - decoded).abs() < 1e-15 || (v.is_nan() && decoded.is_nan()),
                    "varfloat roundtrip failed for {}: got {}", v, decoded);
        }
    }

    #[test]
    fn test_sketch_encode_decode() {
        let mut sketch = DataDogSketch::new(0.01);

        // Add some values
        for i in 1..=100 {
            sketch.add(i as f64);
        }

        // Encode
        let bytes = sketch.encode().unwrap();

        // Decode
        let decoded = DataDogSketch::decode(&bytes).unwrap();

        // Check basic properties - use relative tolerance since sum is computed from bins
        assert_eq!(sketch.count, decoded.count);
        // Sum computed from bins has relative accuracy error (can compound slightly)
        let rel_error = (sketch.sum - decoded.sum).abs() / sketch.sum;
        assert!(rel_error < 0.03, "sum relative error {} too high", rel_error);
        assert_eq!(sketch.positive_bins.len(), decoded.positive_bins.len());
    }

    #[test]
    fn test_sketch_merge() {
        let mut sketch1 = DataDogSketch::new(0.01);
        sketch1.add(10.0);
        sketch1.add(20.0);

        let mut sketch2 = DataDogSketch::new(0.01);
        sketch2.add(30.0);
        sketch2.add(40.0);

        sketch1.merge(&sketch2).unwrap();

        assert_eq!(sketch1.count, 4.0);
        assert_eq!(sketch1.sum, 100.0);
        assert_eq!(sketch1.min, 10.0);
        assert_eq!(sketch1.max, 40.0);
    }

    #[test]
    fn test_sketch_merge_rejects_different_gamma() {
        let mut sketch1 = DataDogSketch::new(0.01);
        sketch1.add(10.0);

        let mut sketch2 = DataDogSketch::new(0.02); // Different accuracy
        sketch2.add(20.0);

        let result = sketch1.merge(&sketch2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("gamma"));
    }

    #[test]
    fn test_sketch_merge_rejects_different_index_offset() {
        let mut sketch1 = DataDogSketch::new(0.01);
        sketch1.add(10.0);

        let mut sketch2 = DataDogSketch::new(0.01);
        sketch2.add(20.0);
        sketch2.index_offset = 5.0; // Different offset

        let result = sketch1.merge(&sketch2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("index_offset"));
    }

    #[test]
    fn test_sketch_quantiles() {
        let mut sketch = DataDogSketch::new(0.01);

        // Add values 1 to 100
        for i in 1..=100 {
            sketch.add(i as f64);
        }

        // Test quantiles - DDSketch returns bin center value, within relative accuracy
        let p50 = sketch.quantile(0.50).unwrap();
        assert!(p50 >= 48.0 && p50 <= 52.0, "p50 should be near 50, got {}", p50);

        let p99 = sketch.quantile(0.99).unwrap();
        assert!(p99 >= 97.0 && p99 <= 103.0, "p99 should be near 100, got {}", p99);
    }

    #[test]
    fn test_sketch_encode_decode_with_min_max() {
        let mut sketch = DataDogSketch::new(0.01);
        sketch.add(5.5);
        sketch.add(100.25);
        sketch.add(50.0);

        let bytes = sketch.encode().unwrap();
        let decoded = DataDogSketch::decode(&bytes).unwrap();

        // Min/max/sum are computed from bins using bin center value (1% accuracy + quantization)
        let rel_err_min = (decoded.min - 5.5).abs() / 5.5;
        let rel_err_max = (decoded.max - 100.25).abs() / 100.25;
        let rel_err_sum = (decoded.sum - 155.75).abs() / 155.75;

        assert!(rel_err_min < 0.03, "min error too high: {}", rel_err_min);
        assert!(rel_err_max < 0.03, "max error too high: {}", rel_err_max);
        assert_eq!(decoded.count, 3.0);
        assert!(rel_err_sum < 0.03, "sum error too high: {}", rel_err_sum);
    }
}

// ============================================================================
// DataDog Go Library Compatibility Tests
// Test vectors from github.com/DataDog/sketches-go v1.4.7
// Generated with: LogarithmicMapping(relativeAccuracy=0.01), DenseStore
// ============================================================================
#[cfg(test)]
mod compatibility_tests {
    use super::*;

    fn hex_decode(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    fn approx_eq(a: f64, b: f64, rel_tol: f64) -> bool {
        if a == b { return true; }
        if a.is_nan() && b.is_nan() { return true; }
        let diff = (a - b).abs();
        let max_val = a.abs().max(b.abs());
        if max_val == 0.0 { return diff < 1e-10; }
        diff <= max_val * rel_tol
    }

    #[test]
    fn test_decode_empty() {
        let bytes = hex_decode("02fd4a815abf52f03f0000000000000000");
        let sketch = DataDogSketch::decode(&bytes).unwrap();
        assert_eq!(sketch.count(), 0);
        assert_eq!(sketch.sum, 0.0);
    }

    #[test]
    fn test_decode_single_value() {
        // Single value: 42
        let bytes = hex_decode("02fd4a815abf52f03f00000000000000000501f40202");
        let sketch = DataDogSketch::decode(&bytes).unwrap();
        assert_eq!(sketch.count(), 1);
        assert!(approx_eq(sketch.sum, 41.682206632978456, 0.01),
            "sum: expected ~41.68, got {}", sketch.sum);
    }

    #[test]
    fn test_decode_two_values() {
        // Values: [10, 20]
        let bytes = hex_decode("02fd4a815abf52f03f00000000000000000502e601024402");
        let sketch = DataDogSketch::decode(&bytes).unwrap();
        assert_eq!(sketch.count(), 2);
        assert!(approx_eq(sketch.sum, 29.96136693037751, 0.01),
            "sum: expected ~29.96, got {}", sketch.sum);
    }

    #[test]
    fn test_decode_sequential_1_to_10() {
        // Values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let bytes = hex_decode("02fd4a815abf52f03f0000000000000000050a0002440228021e021602120210020c020c020c02");
        let sketch = DataDogSketch::decode(&bytes).unwrap();
        assert_eq!(sketch.count(), 10);
        assert!(approx_eq(sketch.sum, 54.948952462932695, 0.01),
            "sum: expected ~54.95, got {}", sketch.sum);

        // Test percentiles
        let p50 = sketch.quantile(0.50).unwrap();
        assert!(approx_eq(p50, 5.002829575110703, 0.02),
            "p50: expected ~5.0, got {}", p50);

        let p90 = sketch.quantile(0.90).unwrap();
        assert!(approx_eq(p90, 8.935418643763573, 0.02),
            "p90: expected ~8.94, got {}", p90);
    }

    #[test]
    fn test_decode_small_values() {
        // Values: [0.001, 0.002, 0.003, 0.004, 0.005]
        let bytes = hex_decode("02fd4a815abf52f03f00000000000000000505b30502460228021c021802");
        let sketch = DataDogSketch::decode(&bytes).unwrap();
        assert_eq!(sketch.count(), 5);
        assert!(approx_eq(sketch.sum, 0.015008577971483281, 0.01),
            "sum: expected ~0.015, got {}", sketch.sum);
    }

    #[test]
    fn test_decode_large_values() {
        // Values: [1000000, 2000000, 3000000]
        let bytes = hex_decode("02fd4a815abf52f03f00000000000000000503e40a0246022802");
        let sketch = DataDogSketch::decode(&bytes).unwrap();
        assert_eq!(sketch.count(), 3);
        assert!(approx_eq(sketch.sum, 5987460.634366453, 0.01),
            "sum: expected ~5987460.6, got {}", sketch.sum);
    }

    #[test]
    fn test_decode_mixed_magnitude() {
        // Values: [0.1, 1, 10, 100, 1000]
        let bytes = hex_decode("02fd4a815abf52f03f00000000000000000505e70102e80102e60102e60102e60102");
        let sketch = DataDogSketch::decode(&bytes).unwrap();
        assert_eq!(sketch.count(), 5);
        assert!(approx_eq(sketch.sum, 1114.1065215656804, 0.01),
            "sum: expected ~1114.1, got {}", sketch.sum);
    }

    #[test]
    fn test_decode_duplicate_values() {
        // Values: [50, 50, 50, 50, 50]
        let bytes = hex_decode("02fd4a815abf52f03f00000000000000000501860305");
        let sketch = DataDogSketch::decode(&bytes).unwrap();
        assert_eq!(sketch.count(), 5);
        assert!(approx_eq(sketch.sum, 249.51480474533258, 0.01),
            "sum: expected ~249.5, got {}", sketch.sum);
    }

    #[test]
    fn test_decode_latency_uniform() {
        // Values: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        let bytes = hex_decode("02fd4a815abf52f03f0000000000000000050ae6010244022a021c021602120210020e020a020c02");
        let sketch = DataDogSketch::decode(&bytes).unwrap();
        assert_eq!(sketch.count(), 10);
        assert!(approx_eq(sketch.sum, 550.3092442194343, 0.01),
            "sum: expected ~550.3, got {}", sketch.sum);

        let p50 = sketch.quantile(0.50).unwrap();
        assert!(approx_eq(p50, 49.90296094906652, 0.02),
            "p50: expected ~49.9, got {}", p50);

        let p90 = sketch.quantile(0.90).unwrap();
        assert!(approx_eq(p90, 89.1303293363591, 0.02),
            "p90: expected ~89.1, got {}", p90);
    }

    #[test]
    fn test_decode_latency_skewed() {
        // Typical API latency distribution with outliers
        // Values: [5, 5, 5, 6, 6, 7, 8, 10, 15, 50, 100, 500]
        let bytes = hex_decode("02fd4a815abf52f03f00000000000000000509a00104120310020c021802280278024602a00102");
        let sketch = DataDogSketch::decode(&bytes).unwrap();
        assert_eq!(sketch.count(), 12);
        assert!(approx_eq(sketch.sum, 715.2227847478571, 0.01),
            "sum: expected ~715.2, got {}", sketch.sum);

        let p50 = sketch.quantile(0.50).unwrap();
        assert!(approx_eq(p50, 7.028793021534767, 0.02),
            "p50: expected ~7.0, got {}", p50);

        let p99 = sketch.quantile(0.99).unwrap();
        assert!(approx_eq(p99, 100.49456770856489, 0.02),
            "p99: expected ~100.5, got {}", p99);
    }

    #[test]
    fn test_decode_merged_sketches() {
        // Result of merging [1,2,3,4,5] + [6,7,8,9,10]
        let bytes = hex_decode("02fd4a815abf52f03f0000000000000000050a0002440228021e021602120210020c020c020c02");
        let sketch = DataDogSketch::decode(&bytes).unwrap();
        assert_eq!(sketch.count(), 10);
        assert!(approx_eq(sketch.sum, 54.948952462932695, 0.01),
            "sum: expected ~54.95, got {}", sketch.sum);
    }

    #[test]
    fn test_decode_merged_overlapping() {
        // Merge sketches with overlapping ranges: [1,2,3,4,5] + [3,4,5,6,7]
        let bytes = hex_decode("02fd4a815abf52f03f000000000000000005070002440228031e03160312021002");
        let sketch = DataDogSketch::decode(&bytes).unwrap();
        assert_eq!(sketch.count(), 10);
        assert!(approx_eq(sketch.sum, 40.00576175735671, 0.01),
            "sum: expected ~40.0, got {}", sketch.sum);
    }

    #[test]
    fn test_roundtrip_compatibility() {
        // Create sketch, encode, decode, verify values match
        let mut sketch = DataDogSketch::new(0.01);
        for i in 1..=10 {
            sketch.add(i as f64);
        }

        let bytes = sketch.encode().unwrap();
        let decoded = DataDogSketch::decode(&bytes).unwrap();

        assert_eq!(sketch.count(), decoded.count());
        // Sum is computed from bins, use relative tolerance
        let rel_err = (sketch.sum - decoded.sum).abs() / sketch.sum;
        assert!(rel_err < 0.02, "sum relative error {} too high", rel_err);
        assert_eq!(sketch.positive_bins.len(), decoded.positive_bins.len());
    }

    fn from_hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i+2], 16).unwrap())
            .collect()
    }

    #[test]
    fn test_contiguous_counts_sketch() {
        // Sketch with values 51-100 using ContiguousCounts encoding (subflag=3)
        // Structure: IndexMapping(17) + PositiveStore(1+1+2+1+35) = 57 bytes
        // The extra byte is indexDelta (02 = 1 in zig-zag varint)
        let hex_str = "02fd4a815abf52f03f00000000000000000d238803020202020202020203020202030202030202030203020302030302030303020303030302";
        let bytes = from_hex(hex_str);

        let sketch = DataDogSketch::decode(&bytes).unwrap();

        assert_eq!(bytes.len(), 57);
        assert_eq!(sketch.positive_bins.len(), 35);
        assert_eq!(sketch.count as i64, 50);
        assert!(sketch.sum > 3700.0 && sketch.sum < 3800.0);
    }

    #[test]
    fn test_go_generated_roundtrip() {
        // Decode Go-generated sketch, re-encode with our encoder, decode again
        // This verifies we write a format that we can read back correctly
        let hex_str = "02fd4a815abf52f03f00000000000000000d238803020202020202020203020202030202030202030203020302030302030303020303030302";
        let original_bytes = from_hex(hex_str);

        // First decode
        let sketch1 = DataDogSketch::decode(&original_bytes).unwrap();
        let count1 = sketch1.count;
        let sum1 = sketch1.sum;
        let bins1 = sketch1.positive_bins.len();

        // Re-encode with our encoder
        let reencoded = sketch1.encode().unwrap();

        // Decode again
        let sketch2 = DataDogSketch::decode(&reencoded).unwrap();

        // Verify values match
        assert_eq!(sketch2.count as i64, count1 as i64, "count mismatch after roundtrip");
        assert!((sketch2.sum - sum1).abs() < 0.001, "sum mismatch: {} vs {}", sketch2.sum, sum1);
        assert_eq!(sketch2.positive_bins.len(), bins1, "bins count mismatch");

        // Also verify against expected values
        assert_eq!(sketch2.count as i64, 50);
        assert!(sketch2.sum > 3700.0 && sketch2.sum < 3800.0);
    }

    // Issue #1 regression tests: Quantile extraction must match Go's GetValueAtQuantile()
    #[test]
    fn test_issue1_count_1_quantile() {
        // Issue: count=1 sketches returned 0 instead of the actual value
        // This was because rank=0 and "cumulative >= 0" was true even with zero_count=0
        let mut sketch = DataDogSketch::new(0.01);
        sketch.add(1.0);

        let p50 = sketch.quantile(0.50).unwrap();
        // With count=1, rank=0. Should return the bin value for 1.0, not 0
        assert!(p50 > 0.5 && p50 < 1.5,
            "count=1 sketch p50 should be ~1.0, got {}", p50);

        // Also test with a larger value
        let mut sketch2 = DataDogSketch::new(0.01);
        sketch2.add(100.0);
        let p50_2 = sketch2.quantile(0.50).unwrap();
        // DDSketch returns bin center value, within relative accuracy (1% + quantization)
        assert!(approx_eq(p50_2, 100.0, 0.03),
            "count=1 sketch with value=100 p50 should be ~100, got {}", p50_2);
    }

    #[test]
    fn test_issue1_quantile_uses_gt_not_gte() {
        // Issue: quantile used ">=" instead of ">" causing wrong bin selection
        // Go's KeyAtRank uses "n > rank" not "n >= rank"
        let mut sketch = DataDogSketch::new(0.01);
        for i in 1..=10 {
            sketch.add(i as f64);
        }

        // count=10, p50: rank = 0.5 * 9 = 4.5
        // With ">": cumulative must exceed 4.5, so we need cumulative=5, returning bin5 (value ~5)
        // With ">=": cumulative=5 >= 4.5 would return bin5, but cumulative=4.5 >= 4.5 is also true
        // The key test is p50 should be ~5, not ~4
        let p50 = sketch.quantile(0.50).unwrap();
        assert!(p50 >= 4.5 && p50 <= 5.5,
            "p50 should be ~5.0, got {}", p50);

        // Go test vector verification: sequential 1-10 has p50 ≈ 5.0
        assert!(approx_eq(p50, 5.002829575110703, 0.05),
            "p50 should match Go's ~5.0, got {}", p50);
    }

    #[test]
    fn test_issue1_go_sketch_quantile() {
        // Test with an actual Go-generated sketch (values 51-100, count=50)
        let hex_str = "02fd4a815abf52f03f00000000000000000d238803020202020202020203020202030202030202030203020302030302030303020303030302";
        let bytes = from_hex(hex_str);
        let sketch = DataDogSketch::decode(&bytes).unwrap();

        assert_eq!(sketch.count as i64, 50);

        // p50 for values 51-100 should be around 75
        let p50 = sketch.quantile(0.50).unwrap();
        assert!(p50 >= 73.0 && p50 <= 77.0,
            "p50 for 51-100 should be ~75, got {}", p50);

        // p0 should be min (~51)
        let p0 = sketch.quantile(0.0);
        // For q=0, we should get the min value
        assert!(p0.is_some());

        // p100 should be max (~100)
        let p100 = sketch.quantile(1.0);
        assert!(p100.is_some());
    }

    #[test]
    fn test_issue1_boundary_quantiles() {
        // Test boundary conditions
        let mut sketch = DataDogSketch::new(0.01);
        for i in 1..=5 {
            sketch.add(i as f64);
        }

        // q=0 should return min
        let q0 = sketch.quantile(0.0);
        assert!(q0.is_some());

        // q=1 should return max
        let q1 = sketch.quantile(1.0);
        assert!(q1.is_some());

        // Invalid quantiles should return None
        assert!(sketch.quantile(-0.1).is_none());
        assert!(sketch.quantile(1.1).is_none());
    }

    #[test]
    fn test_go_compatibility_vectors() {
        // Test vectors from Go DDSketch to verify exact compatibility
        // These hex strings are from real Go sketches

        // Simple case (count=1): Go p50 = 1.01
        let hex1 = "02fd4a815abf52f03f000000000000000005010002";
        let sketch1 = DataDogSketch::decode(&from_hex(hex1)).unwrap();
        assert_eq!(sketch1.count as i64, 1);
        let p50_1 = sketch1.quantile(0.50).unwrap();
        // Go returns 1.01, we should match within tight tolerance
        assert!((p50_1 - 1.01).abs() < 0.02,
            "count=1 p50: expected ~1.01, got {}", p50_1);

        // Multi-bucket case (count=6): Go p50 = 1.01
        let hex2 = "040302fd4a815abf52f03f00000000000000000501008440";
        let sketch2 = DataDogSketch::decode(&from_hex(hex2)).unwrap();
        assert_eq!(sketch2.count as i64, 6);
        let p50_2 = sketch2.quantile(0.50).unwrap();
        assert!((p50_2 - 1.01).abs() < 0.02,
            "count=6 p50: expected ~1.01, got {}", p50_2);

        // High count, single bucket (count=15,399,717): Go p50 = 0.0019689445
        let hex3 = "02fd4a815abf52f03f00000000000000000501ef04afd5fb13";
        let sketch3 = DataDogSketch::decode(&from_hex(hex3)).unwrap();
        assert_eq!(sketch3.count as i64, 15399717);
        let p50_3 = sketch3.quantile(0.50).unwrap();
        // Within 1% relative accuracy
        let rel_err = (p50_3 - 0.0019689445).abs() / 0.0019689445;
        assert!(rel_err < 0.02,
            "high count p50: expected ~0.00197, got {} (rel_err={})", p50_3, rel_err);

        // High count, multi-bucket (count=15,435,728)
        let hex4 = "02fd4a815abf52f03f00000000000000000529ef04aad7cb660ea8fbc52c0aa8fbc6440ca8fbc8080ca8fbc9500ca8fbcb3c12a7b5983006a7b5992808a7b59a1806a7b59b0806a7b59c400c9fbf26069fbf29049fbf2b049fbf2b069fbf2e08989058049891080498913804989138029891382096e8701a96e9501a96e9501c96ea301a96ea30269aa01c0c9aa0340c9aa04c0c9aa06c0c9aa07c1690510c90570c90570a90570c90571887200c89100a87200c89700c8830";
        let sketch4 = DataDogSketch::decode(&from_hex(hex4)).unwrap();
        assert_eq!(sketch4.count as i64, 15435728);
        let p50_4 = sketch4.quantile(0.50).unwrap();
        let p25_4 = sketch4.quantile(0.25).unwrap();
        let p75_4 = sketch4.quantile(0.75).unwrap();
        let p90_4 = sketch4.quantile(0.90).unwrap();
        let p95_4 = sketch4.quantile(0.95).unwrap();
        let p99_4 = sketch4.quantile(0.99).unwrap();

        // Verify within 2% of Go's values
        assert!((p50_4 - 0.0031820117).abs() / 0.0031820117 < 0.02,
            "p50: expected ~0.00318, got {}", p50_4);
        assert!((p25_4 - 0.0022648358).abs() / 0.0022648358 < 0.02,
            "p25: expected ~0.00226, got {}", p25_4);
        assert!((p75_4 - 0.0045609257).abs() / 0.0045609257 < 0.02,
            "p75: expected ~0.00456, got {}", p75_4);
        assert!((p90_4 - 0.0052463378).abs() / 0.0052463378 < 0.02,
            "p90: expected ~0.00525, got {}", p90_4);
        assert!((p95_4 - 0.0055707643).abs() / 0.0055707643 < 0.02,
            "p95: expected ~0.00557, got {}", p95_4);
        assert!((p99_4 - 0.0072249545).abs() / 0.0072249545 < 0.02,
            "p99: expected ~0.00722, got {}", p99_4);
    }
}

    #[test]
    fn print_rust_encoded_hex() {
        // Create same sketches as Go test vectors
        let mut sketch1 = DataDogSketch::new(0.01);
        for i in 1..=10 {
            sketch1.add(i as f64);
        }
        let bytes1 = sketch1.encode().unwrap();
        println!("Rust values 1-10: {}", bytes1.iter().map(|b| format!("{:02x}", b)).collect::<String>());
        println!("  Count: {}, Sum: {}", sketch1.count, sketch1.sum);

        let mut sketch2 = DataDogSketch::new(0.01);
        sketch2.add(100.0);
        sketch2.add(200.0);
        sketch2.add(300.0);
        let bytes2 = sketch2.encode().unwrap();
        println!("Rust values 100,200,300: {}", bytes2.iter().map(|b| format!("{:02x}", b)).collect::<String>());
        println!("  Count: {}, Sum: {}", sketch2.count, sketch2.sum);
    }
