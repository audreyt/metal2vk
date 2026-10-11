// v13: v11 without the guarded (lane==0) store - store unconditionally.
[[kernel]] void v13_uncond_store(
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
    group_indices[token * 2 + simd_group] = best_index;
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

// v14: v8 (minimal guarded tg read) plus a guarded store to the same array.
[[kernel]] void v14_guarded_store_plus_read(
  const device float16_t* x [[buffer(0)]],
  device int* out [[buffer(4)]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 batch_index [[thread_position_in_grid]],
  uint simd_group [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int idx[4];
  if (lane == 0) {
    idx[simd_group] = int(simd_group);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int best = idx[0];
  for (int g = 1; g < 4; ++g) {
    if (float(idx[g]) > float(best)) {
      best = idx[g];
    }
  }
  out[int(threadgroup_position_in_grid.y)] = best;
}

// v15: v11's reduction nested inside the outer token loop with the guarded
// store, but the condition reads the threadgroup array (like v8) instead of
// the private array.
[[kernel]] void v15_tg_condition(
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
      int best_index = group_indices[token * 2];
      for (int group = 1; group < 2; ++group) {
        if (float(group) > float(best_index)) {
          best_index = group_indices[token * 2 + group];
        }
      }
      tile_indices[token * 8 + int(threadgroup_position_in_grid.y)] = best_index;
    }
  }
}
