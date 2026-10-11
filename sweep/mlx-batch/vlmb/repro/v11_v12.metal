// v11/v12 split v6's combined update.

// v11: only the int threadgroup-array update in the guarded branch; float side dropped.
[[kernel]] void v11_int_update_only(
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

// v12: both updates, but the int side reads the PRIVATE array instead of the
// threadgroup array.
[[kernel]] void v12_both_private_int(
  const device float16_t* x [[buffer(0)]],
  device float16_t* tile_values [[buffer(3)]],
  device int* tile_indices [[buffer(4)]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 batch_index [[thread_position_in_grid]],
  uint simd_group [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int group_indices[4];
  float private_values[4];
  int private_indices[4];
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
    private_indices[token * 2 + simd_group] = best_index;
    if (lane == 0) {
      group_indices[token * 2 + simd_group] = best_index;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0 && lane == 0) {
    for (int token = 0; token < 2; ++token) {
      float best_value = private_values[token * 2];
      int best_index = private_indices[token * 2];
      for (int group = 1; group < 2; ++group) {
        float value = private_values[token * 2 + group];
        if (value > best_value) {
          best_value = value;
          best_index = private_indices[token * 2 + group];
        }
      }
      int offset = token * 8 + int(threadgroup_position_in_grid.y);
      tile_values[offset] = best_value;
      tile_indices[offset] = best_index;
    }
  }
}
