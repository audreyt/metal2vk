// v19: v16 with the first-pass argmax removed (constant stored to the arrays).
[[kernel]] void v19_no_first_pass(
  const device float16_t* x [[buffer(0)]],
  device float16_t* tile_values [[buffer(3)]],
  device int* tile_indices [[buffer(4)]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 batch_index [[thread_position_in_grid]],
  uint simd_group [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int group_indices[4];
  float private_values[4];
  private_values[simd_group] = float(simd_group);
  group_indices[simd_group] = int(simd_group);
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

// v20: v19 with the outer token loop collapsed to one iteration body.
[[kernel]] void v20_single_token(
  const device float16_t* x [[buffer(0)]],
  device float16_t* tile_values [[buffer(3)]],
  device int* tile_indices [[buffer(4)]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 batch_index [[thread_position_in_grid]],
  uint simd_group [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int group_indices[4];
  float private_values[4];
  private_values[simd_group] = float(simd_group);
  group_indices[simd_group] = int(simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0 && lane == 0) {
    float best_value = private_values[0];
    int best_index = group_indices[0];
    for (int group = 1; group < 2; ++group) {
      float value = private_values[group];
      if (value > best_value) {
        best_value = value;
        best_index = group_indices[group];
      }
    }
    tile_values[int(threadgroup_position_in_grid.y)] = best_value;
    tile_indices[int(threadgroup_position_in_grid.y)] = best_index;
  }
}
