// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:66.
// Representative call: q_rot (1,1,1,64) float16, norms (1,1,2) float16, packed (1,1,2,8) uint32, codebook (16) float32; Dim=64,Bits=4,PackedWidth=8; grid=(32,1,2), threadgroup=(32,1,1); out (1,1,1,2) float32.
template <int Dim, int Bits, int PackedWidth>
[[kernel]] void custom_kernel_turboquant_mse_score__64_4_8_float16_t_float16_tc_uint32_t_float_float(
  const device float16_t* q_rot [[buffer(0)]],
  const constant int* q_rot_shape [[buffer(1)]],
  const constant float16_t* norms [[buffer(2)]],
  const constant int* norms_shape [[buffer(3)]],
  const device uint32_t* packed [[buffer(4)]],
  const device float* codebook [[buffer(5)]],
  device float* out [[buffer(6)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

        auto lane = thread_position_in_grid.x;
        auto repeat_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = q_rot_shape[2];
        if (repeat_idx >= repeat_count) {
            return;
        }

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_ptr = q_rot + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto packed_ptr = packed + ((b * kv_heads + h) * token_count + t) * PackedWidth;

        float acc = 0.0f;
        for (int d = lane; d < Dim; d += 32) {
            int bit_offset = d * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = packed_ptr[word_idx] >> offset;
            int spill = offset + Bits - 32;
            if (spill > 0) {
                value |= packed_ptr[word_idx + 1] << (Bits - spill);
            }
            value &= ((1u << Bits) - 1u);
            acc += static_cast<float>(q_ptr[d]) * codebook[value];
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                acc * static_cast<float>(norms[(b * kv_heads + h) * token_count + t]);
        }
    
}

template [[host_name("custom_kernel_turboquant_mse_score__64_4_8_float16_t_float16_tc_uint32_t_float_float")]] [[kernel]] decltype(custom_kernel_turboquant_mse_score__64_4_8_float16_t_float16_tc_uint32_t_float_float<64, 4, 8>) custom_kernel_turboquant_mse_score__64_4_8_float16_t_float16_tc_uint32_t_float_float<64, 4, 8>;
