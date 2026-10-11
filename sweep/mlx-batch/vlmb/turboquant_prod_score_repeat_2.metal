// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:523.
// Representative call: q_rot/q_proj (1,1,2,64) float16, norms/residual_norms (1,1,2) float16, mse_packed (1,1,2,8) uint32, signs (1,1,2,2) uint32, codebook (16) float32, scale (1) float32; Dim=64,RepeatCount=2,MseBits=4,MsePackedWidth=8,SignPackedWidth=2; grid=(32,1,2), threadgroup=(32,1,1); out (1,1,2,2) float32.
template <int Dim, int RepeatCount, int MseBits, int MsePackedWidth, int SignPackedWidth>
[[kernel]] void custom_kernel_turboquant_prod_score_repeat_2__64_2_4_8_2_float16_t_float16_t_float16_tc_float16_tc_uint32_t_uint32_tc_float_floatc_float(
  const device float16_t* q_rot [[buffer(0)]],
  const constant int* q_rot_shape [[buffer(1)]],
  const device float16_t* q_proj [[buffer(2)]],
  const constant float16_t* norms [[buffer(3)]],
  const constant int* norms_shape [[buffer(4)]],
  const constant float16_t* residual_norms [[buffer(5)]],
  const device uint32_t* mse_packed [[buffer(6)]],
  const constant uint32_t* signs [[buffer(7)]],
  const device float* codebook [[buffer(8)]],
  const constant float* scale [[buffer(9)]],
  device float* out [[buffer(10)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        auto lane = thread_position_in_grid.x;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = q_rot_shape[2];
        constexpr uint mse_mask = (1u << MseBits) - 1u;
        constexpr int num_entries = 1 << MseBits;

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_rot_base = q_rot + ((b * kv_heads + h) * repeat_count) * Dim;
        auto q_proj_base = q_proj + ((b * kv_heads + h) * repeat_count) * Dim;
        auto mse_ptr = mse_packed + ((b * kv_heads + h) * token_count + t) * MsePackedWidth;
        auto sign_ptr = signs + ((b * kv_heads + h) * token_count + t) * SignPackedWidth;

        auto idx = (b * kv_heads + h) * token_count + t;
        float norm = static_cast<float>(norms[idx]);
        float residual_norm = static_cast<float>(residual_norms[idx]);

        float grp_0[num_entries] = {};
        float qjl_acc_0 = 0.0f;
        float grp_1[num_entries] = {};
        float qjl_acc_1 = 0.0f;

        for (int d = lane; d < Dim; d += 32) {
            int bit_offset = d * MseBits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = mse_ptr[word_idx] >> offset;
            int spill = offset + MseBits - 32;
            if (spill > 0) {
                value |= mse_ptr[word_idx + 1] << (MseBits - spill);
            }
            value &= mse_mask;

            int sign_word = d / 32;
            int sign_offset = d % 32;
            uint bit = (sign_ptr[sign_word] >> sign_offset) & 1u;
            float sign = bit ? 1.0f : -1.0f;

            grp_0[value] += static_cast<float>(q_rot_base[0 * Dim + d]);
            qjl_acc_0 += static_cast<float>(q_proj_base[0 * Dim + d]) * sign;
            grp_1[value] += static_cast<float>(q_rot_base[1 * Dim + d]);
            qjl_acc_1 += static_cast<float>(q_proj_base[1 * Dim + d]) * sign;
        }

        // Dot product: group sums × codebook entries
        float mse_acc_0 = 0.0f;
        for (int k = 0; k < num_entries; k++)
            mse_acc_0 += grp_0[k] * codebook[k];
        float mse_acc_1 = 0.0f;
        for (int k = 0; k < num_entries; k++)
            mse_acc_1 += grp_1[k] * codebook[k];

        float mse_sum_0 = simd_sum(mse_acc_0);
        float qjl_sum_0 = simd_sum(qjl_acc_0);
        float mse_sum_1 = simd_sum(mse_acc_1);
        float qjl_sum_1 = simd_sum(qjl_acc_1);

        if (thread_index_in_simdgroup == 0) {
            out[((b * kv_heads + h) * repeat_count + 0) * token_count + t] =
                norm * (mse_sum_0 + scale[0] * residual_norm * qjl_sum_0);
            out[((b * kv_heads + h) * repeat_count + 1) * token_count + t] =
                norm * (mse_sum_1 + scale[0] * residual_norm * qjl_sum_1);
        }
}

template [[host_name("custom_kernel_turboquant_prod_score_repeat_2__64_2_4_8_2_float16_t_float16_t_float16_tc_float16_tc_uint32_t_uint32_tc_float_floatc_float")]] [[kernel]] decltype(custom_kernel_turboquant_prod_score_repeat_2__64_2_4_8_2_float16_t_float16_t_float16_tc_float16_tc_uint32_t_uint32_tc_float_floatc_float<64, 2, 4, 8, 2>) custom_kernel_turboquant_prod_score_repeat_2__64_2_4_8_2_float16_t_float16_t_float16_tc_float16_tc_uint32_t_uint32_tc_float_floatc_float<64, 2, 4, 8, 2>;
