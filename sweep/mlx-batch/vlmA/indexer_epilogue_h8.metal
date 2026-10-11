// Origin: mlx-vlm@8f5dc3d, mlx-vlm/mlx_vlm/models/longcat_flash_sparse/indexer_kernel.py:39.
// Representative call: float32 scores (1, 8 heads, L 2, Np 4), TG 256 as fixed by the module constant; grid=(256, 2, 1), threadgroup=(256, 1, 1).
template <int NHEADS, int TG>
[[kernel]] void custom_kernel_indexer_epilogue_h8__8_256_float_float_floatc_int32_tc_float(
  const device float* scores_in [[buffer(0)]],
  const device float* weights [[buffer(1)]],
  const constant float* scale [[buffer(2)]],
  const constant int32_t* meta [[buffer(3)]],
  device float* scores [[buffer(4)]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    uint tid   = thread_position_in_threadgroup.x;
    uint bl    = threadgroup_position_in_grid.y;
    uint stile = threadgroup_position_in_grid.z;

    int Np = meta[0];
    int L  = meta[1];
    int b  = int(bl) / L;
    int l  = int(bl) - b * L;
    uint s = stile * TG + tid;

    threadgroup float wc[NHEADS];
    if (tid < (uint)NHEADS) {
        wc[tid] = weights[(b * L + l) * NHEADS + int(tid)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (s < (uint)Np) {
        long base = (long(b) * NHEADS * L + l) * Np + long(s);
        long hstride = long(L) * Np;
        float acc = 0.0f;
        for (int h = 0; h < NHEADS; ++h) {
            float g = scores_in[base + long(h) * hstride];
            acc += max(g, 0.0f) * wc[h];
        }
        scores[(b * L + l) * Np + int(s)] = acc * scale[0];
    }

}

template [[host_name("custom_kernel_indexer_epilogue_h8__8_256_float_float_floatc_int32_tc_float")]] [[kernel]] decltype(custom_kernel_indexer_epilogue_h8__8_256_float_float_floatc_int32_tc_float<8, 256>) custom_kernel_indexer_epilogue_h8__8_256_float_float_floatc_int32_tc_float<8, 256>;
