// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:610.
// Representative call: polar levels (2,2,2,2): q_rot (1,1,1,64) float16, norms (1,1,2) float16, radii (1,1,2,1) uint32, angle levels uint32 (1,1,2,2)/(1,1,2,1)x3, cos/sin tables (4) float32 x8; Dim=64,Levels=4,BlockCount=1,PackedWidth1=2,PackedWidth2..4=1; grid=(32,1,2), threadgroup=(32,1,1); out (1,1,1,2) float32.
template <int Dim, int Levels, int BlockCount, int PackedWidth1, int PackedWidth2, int PackedWidth3, int PackedWidth4>
[[kernel]] void custom_kernel_turboquant_polar_prod_score_2_2_2_2__64_4_1_2_1_1_1_float16_t_float16_tc_uint32_tc_uint32_tc_uint32_tc_uint32_tc_uint32_tc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_float(
  const device float16_t* q_rot [[buffer(0)]],
  const constant int* q_rot_shape [[buffer(1)]],
  const constant float16_t* norms [[buffer(2)]],
  const constant int* norms_shape [[buffer(3)]],
  const constant uint32_t* radii [[buffer(4)]],
  const constant uint32_t* angles_1 [[buffer(5)]],
  const constant uint32_t* angles_2 [[buffer(6)]],
  const constant uint32_t* angles_3 [[buffer(7)]],
  const constant uint32_t* angles_4 [[buffer(8)]],
  const constant float* cos_1 [[buffer(9)]],
  const constant float* sin_1 [[buffer(10)]],
  const constant float* cos_2 [[buffer(11)]],
  const constant float* sin_2 [[buffer(12)]],
  const constant float* cos_3 [[buffer(13)]],
  const constant float* sin_3 [[buffer(14)]],
  const constant float* cos_4 [[buffer(15)]],
  const constant float* sin_4 [[buffer(16)]],
  device float* out [[buffer(17)]],
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
        auto radii_ptr = radii + ((b * kv_heads + h) * token_count + t) * BlockCount;

        float acc = 0.0f;
        for (int d = lane; d < Dim; d += 32) {
            int block_idx = d >> Levels;
            float coeff = static_cast<float>(radii_ptr[block_idx]);

            auto angle_ptr_1 = angles_1 + ((b * kv_heads + h) * token_count + t) * PackedWidth1;
            int angle_idx_1 = d >> 1;
            int bit_offset_1 = angle_idx_1 * 2;
            int word_idx_1 = bit_offset_1 / 32;
            int offset_1 = bit_offset_1 % 32;
            uint value_1 = angle_ptr_1[word_idx_1] >> offset_1;
            int spill_1 = offset_1 + 2 - 32;
            if (spill_1 > 0) {
                value_1 |= angle_ptr_1[word_idx_1 + 1] << (2 - spill_1);
            }
            value_1 &= 3u;
            bool use_sin_1 = ((d >> 0) & 1) != 0;
            coeff *= use_sin_1 ? static_cast<float>(sin_1[value_1]) : static_cast<float>(cos_1[value_1]);

            auto angle_ptr_2 = angles_2 + ((b * kv_heads + h) * token_count + t) * PackedWidth2;
            int angle_idx_2 = d >> 2;
            int bit_offset_2 = angle_idx_2 * 2;
            int word_idx_2 = bit_offset_2 / 32;
            int offset_2 = bit_offset_2 % 32;
            uint value_2 = angle_ptr_2[word_idx_2] >> offset_2;
            int spill_2 = offset_2 + 2 - 32;
            if (spill_2 > 0) {
                value_2 |= angle_ptr_2[word_idx_2 + 1] << (2 - spill_2);
            }
            value_2 &= 3u;
            bool use_sin_2 = ((d >> 1) & 1) != 0;
            coeff *= use_sin_2 ? static_cast<float>(sin_2[value_2]) : static_cast<float>(cos_2[value_2]);

            auto angle_ptr_3 = angles_3 + ((b * kv_heads + h) * token_count + t) * PackedWidth3;
            int angle_idx_3 = d >> 3;
            int bit_offset_3 = angle_idx_3 * 2;
            int word_idx_3 = bit_offset_3 / 32;
            int offset_3 = bit_offset_3 % 32;
            uint value_3 = angle_ptr_3[word_idx_3] >> offset_3;
            int spill_3 = offset_3 + 2 - 32;
            if (spill_3 > 0) {
                value_3 |= angle_ptr_3[word_idx_3 + 1] << (2 - spill_3);
            }
            value_3 &= 3u;
            bool use_sin_3 = ((d >> 2) & 1) != 0;
            coeff *= use_sin_3 ? static_cast<float>(sin_3[value_3]) : static_cast<float>(cos_3[value_3]);

            auto angle_ptr_4 = angles_4 + ((b * kv_heads + h) * token_count + t) * PackedWidth4;
            int angle_idx_4 = d >> 4;
            int bit_offset_4 = angle_idx_4 * 2;
            int word_idx_4 = bit_offset_4 / 32;
            int offset_4 = bit_offset_4 % 32;
            uint value_4 = angle_ptr_4[word_idx_4] >> offset_4;
            int spill_4 = offset_4 + 2 - 32;
            if (spill_4 > 0) {
                value_4 |= angle_ptr_4[word_idx_4 + 1] << (2 - spill_4);
            }
            value_4 &= 3u;
            bool use_sin_4 = ((d >> 3) & 1) != 0;
            coeff *= use_sin_4 ? static_cast<float>(sin_4[value_4]) : static_cast<float>(cos_4[value_4]);

            acc += static_cast<float>(q_ptr[d]) * coeff;
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            auto idx = (b * kv_heads + h) * token_count + t;
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                acc * static_cast<float>(norms[idx]);
        }
}

template [[host_name("custom_kernel_turboquant_polar_prod_score_2_2_2_2__64_4_1_2_1_1_1_float16_t_float16_tc_uint32_tc_uint32_tc_uint32_tc_uint32_tc_uint32_tc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_float")]] [[kernel]] decltype(custom_kernel_turboquant_polar_prod_score_2_2_2_2__64_4_1_2_1_1_1_float16_t_float16_tc_uint32_tc_uint32_tc_uint32_tc_uint32_tc_uint32_tc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_float<64, 4, 1, 2, 1, 1, 1>) custom_kernel_turboquant_polar_prod_score_2_2_2_2__64_4_1_2_1_1_1_float16_t_float16_tc_uint32_tc_uint32_tc_uint32_tc_uint32_tc_uint32_tc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_floatc_float<64, 4, 1, 2, 1, 1, 1>;
