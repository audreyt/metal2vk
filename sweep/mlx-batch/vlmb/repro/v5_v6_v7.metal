// v5-v7 isolate the paired-update trigger found in v4 (float[4] + int[4]
// threadgroup arrays, one if updating both, second update reading the int array).

// v5: paired update, but the int side does NOT read the int array (local row).
[[kernel]] void v5_no_cross_read(
  const device float16_t* x [[buffer(0)]],
  device float16_t* tile_values [[buffer(3)]],
  device int* tile_indices [[buffer(4)]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 batch_index [[thread_position_in_grid]],
  uint simd_group [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float group_values[4];
  threadgroup int group_indices[4];
  for (int token = 0; token < 2; ++token) {
    float best_value = -3.4028234663852886e38f;
    int best_index = 0;
    for (int row = 0; row < 4; ++row) {
      float value = float(row);
      if (value > best_value) {
        best_value = value;
        best_index = row;
      }
    }
    if (lane == 0) {
      group_values[token * 2 + simd_group] = best_value;
      group_indices[token * 2 + simd_group] = best_index;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0 && lane == 0) {
    for (int token = 0; token < 2; ++token) {
      float best_value = group_values[token * 2];
      int best_index = group_indices[token * 2];
      for (int group = 1; group < 2; ++group) {
        float value = group_values[token * 2 + group];
        if (value > best_value) {
          best_value = value;
          best_index = group;
        }
      }
      int offset = token * 8 + int(threadgroup_position_in_grid.y);
      tile_values[offset] = best_value;
      tile_indices[offset] = best_index;
    }
  }
}

// v6: cross-array read present, but only ONE threadgroup array (int); the float
// best value lives in a private variable.
[[kernel]] void v6_one_array_cross_read(
  const device float16_t* x [[buffer(0)]],
  device float16_t* tile_values [[buffer(3)]],
  device int* tile_indices [[buffer(4)]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 batch_index [[thread_position_in_grid]],
  uint simd_group [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int group_indices[4];
  float private_values[4];
  for (int token = 0; token < 2; ++token) {
    float best_value = -3.4028234663852886e38f;
    int best_index = 0;
    for (int row = 0; row < 4; ++row) {
      float value = float(row);
      if (value > best_value) {
        best_value = value;
        best_index = row;
      }
    }
    private_values[token * 2 + simd_group] = best_value;
    if (lane == 0) {
      group_indices[token * 2 + simd_group] = best_index;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0 && lane == 0) {
    for (int token = 0; token < 2; ++token) {
      float best_value = private_values[token * 2];
      int best_index = group_indices[token * 2];
      for (int group = 1; group < 2; ++group) {
        float value = private_values[token * 2 + group];
        if (value > best_value) {
          best_value = value;
          best_index = group_indices[token * 2 + group];
        }
      }
      int offset = token * 8 + int(threadgroup_position_in_grid.y);
      tile_values[offset] = best_value;
      tile_indices[offset] = best_index;
    }
  }
}

// v7: two threadgroup arrays, cross-array read, but the guarded update is a
// plain if/else with explicit stores (no value-select candidate).
[[kernel]] void v7_two_arrays_ifelse(
  const device float16_t* x [[buffer(0)]],
  device float16_t* tile_values [[buffer(3)]],
  device int* tile_indices [[buffer(4)]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 batch_index [[thread_position_in_grid]],
  uint simd_group [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float group_values[4];
  threadgroup int group_indices[4];
  for (int token = 0; token < 2; ++token) {
    float best_value = -3.4028234663852886e38f;
    int best_index = 0;
    for (int row = 0; row < 4; ++row) {
      float value = float(row);
      if (value > best_value) {
        best_value = value;
        best_index = row;
      }
    }
    if (lane == 0) {
      group_values[token * 2 + simd_group] = best_value;
      group_indices[token * 2 + simd_group] = best_index;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0 && lane == 0) {
    for (int token = 0; token < 2; ++token) {
      float best_value = group_values[token * 2];
      int best_index = group_indices[token * 2];
      for (int group = 1; group < 2; ++group) {
        float value = group_values[token * 2 + group];
        int index = group_indices[token * 2 + group];
        if (value > best_value) {
          best_value = value;
          best_index = index;
        }
      }
      int offset = token * 8 + int(threadgroup_position_in_grid.y);
      tile_values[offset] = best_value;
      tile_indices[offset] = best_index;
    }
  }
}
