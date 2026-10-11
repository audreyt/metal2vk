// Origin: mlx-vlm@8f5dc3d, mlx-vlm/mlx_vlm/models/inkling/language.py:652.
// Representative call: q4/g64 affine down projection; the kernel hard-requires IN=2048 (one 64-wide group per lane over 32 lanes); OUT 8, 2 experts; uint32 packed weights as mlx quantized affine passes them; grid=(256, 8, 1), threadgroup=(256, 1, 1).
template <typename T, int OUT, int IN, int GROUPS, int K>
[[kernel]] void custom_kernel_inkling_moe_down_combine__float_8_2048_32_2_float_uint32_t_float_float_uint32_tc_floatc_float(
  const device float* xin [[buffer(0)]],
  const device uint32_t* wq [[buffer(1)]],
  const device float* sc [[buffer(2)]],
  const device float* bi [[buffer(3)]],
  const constant uint32_t* idx [[buffer(4)]],
  const constant float* wk [[buffer(5)]],
  device float* out [[buffer(6)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    // Weighted routed-expert down-projection for decode: gather_qmm's
    // vector-per-expert mode runs at ~200 GB/s on the [2048 -> 4096] down
    // shape (vs ~750 broadcast), so this kernel dequantizes q4/g64 rows
    // directly: one threadgroup per output row, one simdgroup per selected
    // expert, one quant group per lane; the top-k weighted sum over experts
    // is folded in, so the [N, K, out] intermediate never exists.
    uint lane = thread_index_in_simdgroup;
    uint sg   = simdgroup_index_in_threadgroup;   // expert slot (8 sgs, K used)
    uint row  = threadgroup_position_in_grid.y;   // output row [0, OUT)
    uint n    = threadgroup_position_in_grid.z;   // token
    threadgroup float partial[8];
    if (sg < K) {
        uint e = idx[(size_t)n * K + sg];
        const device uint* wr = wq + ((size_t)e * OUT + row) * (IN / 8u);
        const device T* sr = sc + ((size_t)e * OUT + row) * GROUPS;
        const device T* br = bi + ((size_t)e * OUT + row) * GROUPS;
        const device T* xr = xin + ((size_t)n * K + sg) * IN;
        float s = (float)sr[lane];
        float b = (float)br[lane];
        float accq = 0.0f, accx = 0.0f;
        uint base = lane * 8u;      // 8 uint32 = 64 packed q4 values
        uint xbase = lane * 64u;
        for (uint u = 0; u < 8u; ++u) {
            uint w8 = wr[base + u];
            for (uint t = 0; t < 8u; ++t) {
                float xv = (float)xr[xbase + u * 8u + t];
                accq += (float)((w8 >> (4u * t)) & 0xFu) * xv;
                accx += xv;
            }
        }
        float dot = simd_sum(s * accq + b * accx);
        if (lane == 0) {
            // match the unfused chain: down output rounds to T, then the
            // per-expert weight multiply rounds again before the sum
            float dv = (float)((T)dot);
            partial[sg] = (float)((T)(dv * (float)wk[(size_t)n * K + sg]));
        }
    } else if (lane == 0) {
        partial[sg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float tot = 0.0f;
        for (uint t = 0; t < 8u; ++t) tot += partial[t];
        out[(size_t)n * OUT + row] = (T)tot;
    }

}

template [[host_name("custom_kernel_inkling_moe_down_combine__float_8_2048_32_2_float_uint32_t_float_float_uint32_tc_floatc_float")]] [[kernel]] decltype(custom_kernel_inkling_moe_down_combine__float_8_2048_32_2_float_uint32_t_float_float_uint32_tc_floatc_float<float, 8, 2048, 32, 2>) custom_kernel_inkling_moe_down_combine__float_8_2048_32_2_float_uint32_t_float_float_uint32_tc_floatc_float<float, 8, 2048, 32, 2>;
