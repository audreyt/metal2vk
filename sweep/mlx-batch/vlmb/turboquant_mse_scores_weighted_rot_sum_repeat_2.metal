// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:1051.
// Representative call: as scores-weighted-rot; sum variant: same inputs; Dim=64,Bits=4,PackedWidth=8; grid=(32,64,1), threadgroup=(32,1,1); out (1,1,2,64) float32.
template <int Dim, int Bits, int PackedWidth>
[[kernel]] void custom_kernel_turboquant_mse_scores_weighted_rot_sum_repeat_2__64_4_8_floatc_float16_tc_uint32_t_float_floatc_float(
  const constant float* scores [[buffer(0)]],
  const constant int* scores_shape [[buffer(1)]],
  const constant float16_t* norms [[buffer(2)]],
  const constant int* norms_shape [[buffer(3)]],
  const device uint32_t* packed [[buffer(4)]],
  const device float* codebook [[buffer(5)]],
  const constant float* max_scores [[buffer(6)]],
  device float* out [[buffer(7)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        auto lane = thread_position_in_grid.x;
        auto dim_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        if (dim_idx >= Dim) {
            return;
        }

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = scores_shape[2];
        auto b = n / kv_heads;
        auto h = n % kv_heads;

        auto scores_base = scores + ((b * kv_heads + h) * repeat_count) * token_count;
        auto norms_ptr = norms + (b * kv_heads + h) * token_count;
        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;
        auto max_base = max_scores + (b * kv_heads + h) * repeat_count;

        int bit_offset = dim_idx * Bits;
        int word_idx = bit_offset / 32;
        int offset = bit_offset % 32;

        float max_score_0 = static_cast<float>(max_base[0]);
        float acc_0 = 0.0f;
        float max_score_1 = static_cast<float>(max_base[1]);
        float acc_1 = 0.0f;

        for (int t = lane; t < token_count; t += 32) {
            auto token_ptr = packed_ptr + t * PackedWidth;
            uint value = token_ptr[word_idx] >> offset;
            int spill = offset + Bits - 32;
            if (spill > 0) {
                value |= token_ptr[word_idx + 1] << (Bits - spill);
            }
            value &= ((1u << Bits) - 1u);
            float code = codebook[value];
            float norm = static_cast<float>(norms_ptr[t]);
            float weight_0 = exp(static_cast<float>(scores_base[0 * token_count + t]) - max_score_0);
            acc_0 += weight_0 * norm * code;
            float weight_1 = exp(static_cast<float>(scores_base[1 * token_count + t]) - max_score_1);
            acc_1 += weight_1 * norm * code;
        }

        float acc_sum_0 = simd_sum(acc_0);
        float acc_sum_1 = simd_sum(acc_1);

        if (thread_index_in_simdgroup == 0) {
            out[((b * kv_heads + h) * repeat_count + 0) * Dim + dim_idx] = acc_sum_0;
            out[((b * kv_heads + h) * repeat_count + 1) * Dim + dim_idx] = acc_sum_1;
        }
}

template [[host_name("custom_kernel_turboquant_mse_scores_weighted_rot_sum_repeat_2__64_4_8_floatc_float16_tc_uint32_t_float_floatc_float")]] [[kernel]] decltype(custom_kernel_turboquant_mse_scores_weighted_rot_sum_repeat_2__64_4_8_floatc_float16_tc_uint32_t_float_floatc_float<64, 4, 8>) custom_kernel_turboquant_mse_scores_weighted_rot_sum_repeat_2__64_4_8_floatc_float16_tc_uint32_t_float_floatc_float<64, 4, 8>;
