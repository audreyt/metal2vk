// Repro reducer for the OpSelect spirv-val failure of
// quantized_verify_nvfp4_argmax (mlx-vlm@8f5dc3d mlx_vlm/models/quantized_verifier.py:1974,
// representative VERIFY_T=2, K_SIZE=64, N_SIZE=64, NUM_TILES=8, T=float16).
// Each variant keeps the failing construct's neighbourhood and drops the rest.
// Status: see repro-results.txt (v0 full fails; variants bisect the trigger).

// --- v1: drop the scoring loop; feed constant results into the reduction.
[[kernel]] void v1_no_scoring(
  const device float16_t* x [[buffer(0)]],
  const device uint8_t* w [[buffer(1)]],
  const device uint8_t* scales [[buffer(2)]],
  device float16_t* tile_values [[buffer(3)]],
  device int* tile_indices [[buffer(4)]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 batch_index [[thread_position_in_grid]],
  uint simd_group [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float group_values[2][2];
  threadgroup int group_indices[2][2];
  float results[2][4];
  for (int token = 0; token < 2; ++token) {
    for (int row = 0; row < 4; ++row) {
      results[token][row] = float(token) + float(row);
    }
  }
  for (int token = 0; token < 2; ++token) {
    float best_value = -3.4028234663852886e38f;
    int best_index = 0;
    for (int row = 0; row < 4; ++row) {
      float value = float(results[token][row]);
      if (value > best_value) {
        best_value = value;
        best_index = row;
      }
    }
    if (lane == 0) {
      group_values[token][simd_group] = best_value;
      group_indices[token][simd_group] = best_index;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0 && lane == 0) {
    for (int token = 0; token < 2; ++token) {
      float best_value = group_values[token][0];
      int best_index = group_indices[token][0];
      for (int group = 1; group < 2; ++group) {
        float value = group_values[token][group];
        if (value > best_value) {
          best_value = value;
          best_index = group_indices[token][group];
        }
      }
      int offset = token * 8 + int(threadgroup_position_in_grid.y);
      tile_values[offset] = best_value;
      tile_indices[offset] = best_index;
    }
  }
}
