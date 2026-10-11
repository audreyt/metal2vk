// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:421.
// Representative call: weights (1,1,1,2) float32, norms (1,1,2) float16, indices (1,1,2,8) uint32, codebook (16) float32; Dim=64,Bits=4,PackedWidth=8; grid=(32,64,1), threadgroup=(32,1,1); out (1,1,1,64) float32.
template <int Dim, int Bits, int PackedWidth>
[[kernel]] void custom_kernel_turboquant_mse_weighted_rot__64_4_8_floatc_float16_tc_uint32_t_float_float(
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
        auto b = n / (kv_heads * repeat_count);
        auto rem = n % (kv_heads * repeat_count);
        auto h = rem / repeat_count;
        auto repeat_idx = rem % repeat_count;

        auto weights_ptr = weights + ((b * kv_heads + h) * repeat_count + repeat_idx) * token_count;
        auto norms_ptr = norms + (b * kv_heads + h) * token_count;
        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;

        float acc = 0.0f;
        for (int t = lane; t < token_count; t += 32) {
            auto token_ptr = packed_ptr + t * PackedWidth;
            int bit_offset = dim_idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = token_ptr[word_idx] >> offset;
            int spill = offset + Bits - 32;
            if (spill > 0) {
                value |= token_ptr[word_idx + 1] << (Bits - spill);
            }
            value &= ((1u << Bits) - 1u);
            acc += static_cast<float>(weights_ptr[t])
                * static_cast<float>(norms_ptr[t])
                * codebook[value];
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * Dim + dim_idx] = acc;
        }
    
}

template [[host_name("custom_kernel_turboquant_mse_weighted_rot__64_4_8_floatc_float16_tc_uint32_t_float_float")]] [[kernel]] decltype(custom_kernel_turboquant_mse_weighted_rot__64_4_8_floatc_float16_tc_uint32_t_float_float<64, 4, 8>) custom_kernel_turboquant_mse_weighted_rot__64_4_8_floatc_float16_tc_uint32_t_float_float<64, 4, 8>;
