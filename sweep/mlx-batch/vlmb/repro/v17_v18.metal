// v17: minimal form. A threadgroup int array read inside a branch whose
// condition comes from a PRIVATE array. Fails spirv-val with
//   OpSelect %_ptr_Workgroup__arr_uint_uint_4 ...
// v18 is the same with the condition read from the threadgroup array (passes).
[[kernel]] void v17_min_private_cond_tg_update(
  const device float16_t* x [[buffer(0)]],
  device int* out [[buffer(4)]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 batch_index [[thread_position_in_grid]],
  uint simd_group [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int tg[4];
  float cond[4];
  tg[lane] = int(lane);
  cond[lane] = float(lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int best = tg[0];
  for (int g = 1; g < 4; ++g) {
    if (cond[g] > cond[best]) {
      best = tg[g];
    }
  }
  out[int(threadgroup_position_in_grid.y)] = best;
}

[[kernel]] void v18_min_tg_cond_tg_update(
  const device float16_t* x [[buffer(0)]],
  device int* out [[buffer(4)]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 batch_index [[thread_position_in_grid]],
  uint simd_group [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int tg[4];
  tg[lane] = int(lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int best = tg[0];
  for (int g = 1; g < 4; ++g) {
    if (float(tg[g]) > float(tg[best])) {
      best = tg[g];
    }
  }
  out[int(threadgroup_position_in_grid.y)] = best;
}
