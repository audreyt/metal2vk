// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:289.
// Representative call: q_proj (1,1,1,64) float16, norms/residual_norms (1,1,2) float16, signs (1,1,2,2) uint32, scale (1) float32; Dim=64,PackedWidth=2; grid=(32,1,2), threadgroup=(32,1,1); out (1,1,1,2) float32.
template <int Dim, int PackedWidth>
[[kernel]] void custom_kernel_turboquant_qjl_score__64_2_float16_t_float16_tc_float16_tc_uint32_tc_floatc_float(
  const device float16_t* q_proj [[buffer(0)]],
  const constant int* q_proj_shape [[buffer(1)]],
  const constant float16_t* norms [[buffer(2)]],
  const constant int* norms_shape [[buffer(3)]],
  const constant float16_t* residual_norms [[buffer(4)]],
  const constant uint32_t* signs [[buffer(5)]],
  const constant float* scale [[buffer(6)]],
  device float* out [[buffer(7)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

        auto lane = thread_position_in_grid.x;
        auto repeat_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = q_proj_shape[2];
        if (repeat_idx >= repeat_count) {
            return;
        }

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_ptr = q_proj + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto packed_ptr = signs + ((b * kv_heads + h) * token_count + t) * PackedWidth;

        float acc = 0.0f;
        for (int d = lane; d < Dim; d += 32) {
            int word_idx = d / 32;
            int offset = d % 32;
            uint bit = (packed_ptr[word_idx] >> offset) & 1u;
            float sign = bit ? 1.0f : -1.0f;
            acc += static_cast<float>(q_ptr[d]) * sign;
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            auto idx = (b * kv_heads + h) * token_count + t;
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                acc
                * static_cast<float>(norms[idx])
                * static_cast<float>(residual_norms[idx])
                * scale[0];
        }
    
}

template [[host_name("custom_kernel_turboquant_qjl_score__64_2_float16_t_float16_tc_float16_tc_uint32_tc_floatc_float")]] [[kernel]] decltype(custom_kernel_turboquant_qjl_score__64_2_float16_t_float16_tc_float16_tc_uint32_tc_floatc_float<64, 2>) custom_kernel_turboquant_qjl_score__64_2_float16_t_float16_tc_float16_tc_uint32_tc_floatc_float<64, 2>;
