// v2: only the int threadgroup array reduction (group_indices), no group_values.
[[kernel]] void v2_indices_only(
  const device float16_t* x [[buffer(0)]],
  device int* tile_indices [[buffer(4)]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 batch_index [[thread_position_in_grid]],
  uint simd_group [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int group_indices[2][2];
  for (int token = 0; token < 2; ++token) {
    int best_index = 0;
    int best_rank = -1;
    for (int row = 0; row < 4; ++row) {
      if (row > best_rank) {
        best_rank = row;
        best_index = row;
      }
    }
    if (lane == 0) {
      group_indices[token][simd_group] = best_index;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0 && lane == 0) {
    for (int token = 0; token < 2; ++token) {
      int best_index = group_indices[token][0];
      int best_rank = 0;
      for (int group = 1; group < 2; ++group) {
        int rank = group;
        int index = group_indices[token][group];
        if (rank > best_rank) {
          best_rank = rank;
          best_index = index;
        }
      }
      tile_indices[token * 8 + int(threadgroup_position_in_grid.y)] = best_index;
    }
  }
}

// v3: only the float threadgroup array reduction (group_values).
[[kernel]] void v3_values_only(
  const device float16_t* x [[buffer(0)]],
  device float16_t* tile_values [[buffer(3)]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 batch_index [[thread_position_in_grid]],
  uint simd_group [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float group_values[2][2];
  for (int token = 0; token < 2; ++token) {
    float best_value = -3.4028234663852886e38f;
    for (int row = 0; row < 4; ++row) {
      float value = float(row);
      if (value > best_value) {
        best_value = value;
      }
    }
    if (lane == 0) {
      group_values[token][simd_group] = best_value;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0 && lane == 0) {
    for (int token = 0; token < 2; ++token) {
      float best_value = group_values[token][0];
      for (int group = 1; group < 2; ++group) {
        float value = group_values[token][group];
        if (value > best_value) {
          best_value = value;
        }
      }
      tile_values[token * 8 + int(threadgroup_position_in_grid.y)] = best_value;
    }
  }
}

// v4: v1's reduction with flat [4] threadgroup arrays (no [2][2]).
[[kernel]] void v4_flat_arrays(
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
          best_index = group_indices[token * 2 + group];
        }
      }
      int offset = token * 8 + int(threadgroup_position_in_grid.y);
      tile_values[offset] = best_value;
      tile_indices[offset] = best_index;
    }
  }
}
