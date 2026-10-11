// v8: minimal candidate - guarded update of a loop-carried scalar from a
// threadgroup array read, condition also from the array.
[[kernel]] void v8_min_guarded_tg_read(
  const device float16_t* x [[buffer(0)]],
  device int* out [[buffer(4)]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 batch_index [[thread_position_in_grid]],
  uint simd_group [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int idx[4];
  idx[lane] = int(lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int best = idx[0];
  for (int g = 1; g < 4; ++g) {
    if (float(idx[g]) > float(best)) {
      best = idx[g];
    }
  }
  out[int(threadgroup_position_in_grid.y)] = best;
}

// v9: like v8 but the guarded value is a constant (no threadgroup read in the if).
[[kernel]] void v9_min_guarded_const(
  const device float16_t* x [[buffer(0)]],
  device int* out [[buffer(4)]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 batch_index [[thread_position_in_grid]],
  uint simd_group [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int idx[4];
  idx[lane] = int(lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int best = idx[0];
  for (int g = 1; g < 4; ++g) {
    if (float(g) > float(best)) {
      best = 1;
    }
  }
  out[int(threadgroup_position_in_grid.y)] = best;
}

// v10: like v8 but the if reads the array while the update writes a local.
[[kernel]] void v10_min_read_no_tg_update(
  const device float16_t* x [[buffer(0)]],
  device int* out [[buffer(4)]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 batch_index [[thread_position_in_grid]],
  uint simd_group [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int idx[4];
  idx[lane] = int(lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int best = idx[0];
  for (int g = 1; g < 4; ++g) {
    int candidate = idx[g];
    if (float(g) > float(best)) {
      best = candidate + 1;
    }
  }
  out[int(threadgroup_position_in_grid.y)] = best;
}
