// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:802.
// Representative call: polar turbo levels (2,2,2,2): q_rot/q_proj (1,1,1,64) float16, norms (1,1,2) float16, radii uint32 (1,1,2,1), angle levels uint32, residual_norms (1,1,2) float16, signs (1,1,2,2) uint32, scale (1) float32, cos/sin tables (4) float32 x8; Dim=64,Levels=4,RepeatCount=1,BlockCount=1,SignPackedWidth=2,PackedWidth1=2,PackedWidth2..4=1; grid=(32,1,2), threadgroup=(32,1,1); out (1,1,1,2) float32.
template <int Dim, int Levels, int RepeatCount, int BlockCount, int SignPackedWidth, int PackedWidth1, int PackedWidth2, int PackedWidth3, int PackedWidth4>
[[kernel]] void custom_kernel_turboquant_polar_turbo_score_2_2_2_2_repeat_1__64_4_1_1_2_2_1_1_1_float16_t_float16_t_float16_tc_uint32_tc_uint32_tc_uint32_tc_uint32_tc_uint32_tc_float16_tc_uint32_tc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_float(
  const device float16_t* q_rot [[buffer(0)]],
  const device float16_t* q_proj [[buffer(1)]],
  const constant float16_t* norms [[buffer(2)]],
  const constant int* norms_shape [[buffer(3)]],
  const constant uint32_t* radii [[buffer(4)]],
  const constant uint32_t* angles_1 [[buffer(5)]],
  const constant uint32_t* angles_2 [[buffer(6)]],
  const constant uint32_t* angles_3 [[buffer(7)]],
  const constant uint32_t* angles_4 [[buffer(8)]],
  const constant float16_t* residual_norms [[buffer(9)]],
  const constant uint32_t* signs [[buffer(10)]],
  const constant float* scale [[buffer(11)]],
  const constant float* cos_1 [[buffer(12)]],
  const constant float* sin_1 [[buffer(13)]],
  const constant float* cos_2 [[buffer(14)]],
  const constant float* sin_2 [[buffer(15)]],
  const constant float* cos_3 [[buffer(16)]],
  const constant float* sin_3 [[buffer(17)]],
  const constant float* cos_4 [[buffer(18)]],
  const constant float* sin_4 [[buffer(19)]],
  device float* out [[buffer(20)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        auto lane = thread_position_in_grid.x;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_rot_base = q_rot + ((b * kv_heads + h) * RepeatCount) * Dim;
        auto q_proj_base = q_proj + ((b * kv_heads + h) * RepeatCount) * Dim;
        auto radii_ptr = radii + ((b * kv_heads + h) * token_count + t) * BlockCount;
        auto angle1_ptr = angles_1 + ((b * kv_heads + h) * token_count + t) * PackedWidth1;
        auto angle2_ptr = angles_2 + ((b * kv_heads + h) * token_count + t) * PackedWidth2;
        auto angle3_ptr = angles_3 + ((b * kv_heads + h) * token_count + t) * PackedWidth3;
        auto angle4_ptr = angles_4 + ((b * kv_heads + h) * token_count + t) * PackedWidth4;
        auto sign_ptr = signs + ((b * kv_heads + h) * token_count + t) * SignPackedWidth;

        auto idx = (b * kv_heads + h) * token_count + t;
        float norm = static_cast<float>(norms[idx]);
        float residual_norm = static_cast<float>(residual_norms[idx]);

        threadgroup float level1_0[Dim / 2];
        threadgroup float level2_0[Dim / 4];
        threadgroup float level3_0[Dim / 8];
        threadgroup float level4_0[BlockCount];

        float qjl_acc_0 = 0.0f;

        for (int pair_idx = lane; pair_idx < Dim / 2; pair_idx += 32) {
            int bit_offset_1 = pair_idx * 2;
            int word_idx_1 = bit_offset_1 / 32;
            int offset_1 = bit_offset_1 % 32;
            uint value_1 = angle1_ptr[word_idx_1] >> offset_1;
            int spill_1 = offset_1 + 2 - 32;
            if (spill_1 > 0) {
                value_1 |= angle1_ptr[word_idx_1 + 1] << (2 - spill_1);
            }
            value_1 &= 3u;
            float cos_1_val = static_cast<float>(cos_1[value_1]);
            float sin_1_val = static_cast<float>(sin_1[value_1]);
            int d0 = pair_idx << 1;
            int d1 = d0 + 1;

            level1_0[pair_idx] = static_cast<float>(q_rot_base[0 * Dim + d0]) * cos_1_val + static_cast<float>(q_rot_base[0 * Dim + d1]) * sin_1_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int pair_idx = lane; pair_idx < Dim / 4; pair_idx += 32) {
            int bit_offset_2 = pair_idx * 2;
            int word_idx_2 = bit_offset_2 / 32;
            int offset_2 = bit_offset_2 % 32;
            uint value_2 = angle2_ptr[word_idx_2] >> offset_2;
            int spill_2 = offset_2 + 2 - 32;
            if (spill_2 > 0) {
                value_2 |= angle2_ptr[word_idx_2 + 1] << (2 - spill_2);
            }
            value_2 &= 3u;
            float cos_2_val = static_cast<float>(cos_2[value_2]);
            float sin_2_val = static_cast<float>(sin_2[value_2]);
            int child = pair_idx << 1;

            level2_0[pair_idx] = level1_0[child] * cos_2_val + level1_0[child + 1] * sin_2_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int pair_idx = lane; pair_idx < Dim / 8; pair_idx += 32) {
            int bit_offset_3 = pair_idx * 2;
            int word_idx_3 = bit_offset_3 / 32;
            int offset_3 = bit_offset_3 % 32;
            uint value_3 = angle3_ptr[word_idx_3] >> offset_3;
            int spill_3 = offset_3 + 2 - 32;
            if (spill_3 > 0) {
                value_3 |= angle3_ptr[word_idx_3 + 1] << (2 - spill_3);
            }
            value_3 &= 3u;
            float cos_3_val = static_cast<float>(cos_3[value_3]);
            float sin_3_val = static_cast<float>(sin_3[value_3]);
            int child = pair_idx << 1;

            level3_0[pair_idx] = level2_0[child] * cos_3_val + level2_0[child + 1] * sin_3_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int pair_idx = lane; pair_idx < BlockCount; pair_idx += 32) {
            int bit_offset_4 = pair_idx * 2;
            int word_idx_4 = bit_offset_4 / 32;
            int offset_4 = bit_offset_4 % 32;
            uint value_4 = angle4_ptr[word_idx_4] >> offset_4;
            int spill_4 = offset_4 + 2 - 32;
            if (spill_4 > 0) {
                value_4 |= angle4_ptr[word_idx_4 + 1] << (2 - spill_4);
            }
            value_4 &= 3u;
            float cos_4_val = static_cast<float>(cos_4[value_4]);
            float sin_4_val = static_cast<float>(sin_4[value_4]);
            int child = pair_idx << 1;

            level4_0[pair_idx] = level3_0[child] * cos_4_val + level3_0[child + 1] * sin_4_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int d = lane; d < Dim; d += 32) {
            int sign_word = d / 32;
            int sign_offset = d % 32;
            uint bit = (sign_ptr[sign_word] >> sign_offset) & 1u;
            float sign = bit ? 1.0f : -1.0f;

            qjl_acc_0 += static_cast<float>(q_proj_base[0 * Dim + d]) * sign;
        }

        float polar_acc_0 = lane < BlockCount ? static_cast<float>(radii_ptr[lane]) * level4_0[lane] : 0.0f;
        float polar_sum_0 = simd_sum(polar_acc_0);
        float qjl_sum_0 = simd_sum(qjl_acc_0);

        if (thread_index_in_simdgroup == 0) {
            out[((b * kv_heads + h) * RepeatCount + 0) * token_count + t] =
                norm * (polar_sum_0 + scale[0] * residual_norm * qjl_sum_0);
        }
}

template [[host_name("custom_kernel_turboquant_polar_turbo_score_2_2_2_2_repeat_1__64_4_1_1_2_2_1_1_1_float16_t_float16_t_float16_tc_uint32_tc_uint32_tc_uint32_tc_uint32_tc_uint32_tc_float16_tc_uint32_tc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_float")]] [[kernel]] decltype(custom_kernel_turboquant_polar_turbo_score_2_2_2_2_repeat_1__64_4_1_1_2_2_1_1_1_float16_t_float16_t_float16_tc_uint32_tc_uint32_tc_uint32_tc_uint32_tc_uint32_tc_float16_tc_uint32_tc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_float<64, 4, 1, 1, 2, 2, 1, 1, 1>) custom_kernel_turboquant_polar_turbo_score_2_2_2_2_repeat_1__64_4_1_1_2_2_1_1_1_float16_t_float16_t_float16_tc_uint32_tc_uint32_tc_uint32_tc_uint32_tc_uint32_tc_float16_tc_uint32_tc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_float<64, 4, 1, 1, 2, 2, 1, 1, 1>;
