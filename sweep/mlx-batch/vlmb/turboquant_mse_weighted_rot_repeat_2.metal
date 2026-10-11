// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:879.
// Representative call: weights (1,1,2,2) float32, norms (1,1,2) float16, indices (1,1,2,8) uint32, codebook (16) float32; Dim=64,RepeatCount=2,Bits=4,PackedWidth=8; grid=(32,64,1), threadgroup=(32,1,1); out (1,1,2,64) float32.
template <int Dim, int RepeatCount, int Bits, int PackedWidth>
[[kernel]] void custom_kernel_turboquant_mse_weighted_rot_repeat_2__64_2_4_8_floatc_float16_tc_uint32_t_float_float(
  const constant float* weights [[buffer(0)]],
  const constant int* weights_shape [[buffer(1)]],
  const constant float16_t* norms [[buffer(2)]],
  const constant int* norms_shape [[buffer(3)]],
  const device uint32_t* packed [[buffer(4)]],
  const device float* codebook [[buffer(5)]],
  device float* out [[buffer(6)]],
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
        auto repeat_count = weights_shape[2];
        auto b = n / kv_heads;
        auto h = n % kv_heads;

        auto weights_base = weights + ((b * kv_heads + h) * repeat_count) * token_count;
        auto norms_ptr = norms + (b * kv_heads + h) * token_count;
        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;

        int bit_offset = dim_idx * Bits;
        int word_idx = bit_offset / 32;
        int offset = bit_offset % 32;

        float acc_0 = 0.0f;
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
            acc_0 += static_cast<float>(weights_base[0 * token_count + t]) * norm * code;
            acc_1 += static_cast<float>(weights_base[1 * token_count + t]) * norm * code;
        }

        float acc_sum_0 = simd_sum(acc_0);
        float acc_sum_1 = simd_sum(acc_1);

        if (thread_index_in_simdgroup == 0) {
            out[((b * kv_heads + h) * repeat_count + 0) * Dim + dim_idx] = acc_sum_0;
            out[((b * kv_heads + h) * repeat_count + 1) * Dim + dim_idx] = acc_sum_1;
        }
}

template [[host_name("custom_kernel_turboquant_mse_weighted_rot_repeat_2__64_2_4_8_floatc_float16_tc_uint32_t_float_float")]] [[kernel]] decltype(custom_kernel_turboquant_mse_weighted_rot_repeat_2__64_2_4_8_floatc_float16_tc_uint32_t_float_float<64, 2, 4, 8>) custom_kernel_turboquant_mse_weighted_rot_repeat_2__64_2_4_8_floatc_float16_tc_uint32_t_float_float<64, 2, 4, 8>;
