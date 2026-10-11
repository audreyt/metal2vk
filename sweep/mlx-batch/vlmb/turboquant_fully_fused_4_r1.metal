// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:2209.
// Representative call: fully fused: as fused integer decode plus rot_input (64,64) float32; Dim=64,RepeatCount=1,DimsPerLane=2,KMsePackedWidth=8,KSignPackedWidth=2,VPackedWidth=8; grid=(32,1,1), threadgroup=(32,1,1); out (1,1,1,64) float32.
template <int Dim, int RepeatCount, int DimsPerLane, int KMsePackedWidth, int KSignPackedWidth, int VPackedWidth>
[[kernel]] void custom_kernel_turboquant_fully_fused_4_r1__64_1_2_8_2_8_float16_t_float16_t_float16_tc_uint32_t_float16_tc_uint32_tc_float16_tc_uint32_t_float_floatc_float_float_float(
  const device float16_t* q_rot [[buffer(0)]],
  const device float16_t* q_proj [[buffer(1)]],
  const constant float16_t* key_norms [[buffer(2)]],
  const constant int* key_norms_shape [[buffer(3)]],
  const device uint32_t* key_mse [[buffer(4)]],
  const constant float16_t* key_res_norms [[buffer(5)]],
  const constant uint32_t* key_signs [[buffer(6)]],
  const constant float16_t* val_norms [[buffer(7)]],
  const device uint32_t* val_packed [[buffer(8)]],
  const device float* key_codebook [[buffer(9)]],
  const constant float* key_scale [[buffer(10)]],
  const device float* val_codebook [[buffer(11)]],
  const device float* rotation_t [[buffer(12)]],
  device float* out [[buffer(13)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        auto lane = thread_position_in_grid.x;
        auto bh = thread_position_in_grid.z;
        auto token_count = key_norms_shape[2];
        auto kv_heads = key_norms_shape[1];
        auto b = bh / kv_heads;
        auto h = bh % kv_heads;
        auto base = (b * kv_heads + h);

        auto k_norms = key_norms + base * token_count;
        auto k_mse = key_mse + base * token_count * KMsePackedWidth;
        auto k_res = key_res_norms + base * token_count;
        auto k_signs = key_signs + base * token_count * KSignPackedWidth;
        auto v_norms = val_norms + base * token_count;
        auto v_packed = val_packed + base * token_count * VPackedWidth;

        // Precompute value bit offsets
        int v_words[DimsPerLane], v_offs[DimsPerLane];
        bool v_spills[DimsPerLane], v_valids[DimsPerLane];
        for (int i = 0, vd = lane; i < DimsPerLane; i++, vd += 32) {
            v_valids[i] = vd < Dim;
            int vbo = vd * 4;
            v_words[i] = vbo / 32;
            v_offs[i] = vbo % 32;
            v_spills[i] = (vbo % 32 + 4) > 32;
        }

        auto qr_0 = q_rot + (base * RepeatCount + 0) * Dim;
        auto qp_0 = q_proj + (base * RepeatCount + 0) * Dim;
        float lmax_0 = -INFINITY, lsum_0 = 0.0f;
        float lacc_0[DimsPerLane] = {};

        for (int t = 0; t < (int)token_count; t++) {
            auto mse_t = k_mse + t * KMsePackedWidth;
            auto sign_t = k_signs + t * KSignPackedWidth;
            float kn = static_cast<float>(k_norms[t]);
            float ksr = kn * key_scale[0] * static_cast<float>(k_res[t]);

            float ps_0 = 0.0f;
            for (int d = lane; d < Dim; d += 32) {
                int bo = d * 4;
                uint idx = (mse_t[bo >> 5] >> (bo & 31));
                if (((bo & 31) + 4) > 32) idx |= mse_t[(bo >> 5) + 1] << (4 - ((bo & 31) + 4 - 32));
                idx &= 15u;
                float code = key_codebook[idx];
                uint sb = (sign_t[d >> 5] >> (d & 31)) & 1u;
                ps_0 += kn * static_cast<float>(qr_0[d]) * code + ksr * (sb ? static_cast<float>(qp_0[d]) : -static_cast<float>(qp_0[d]));
            }
            float s_0 = simd_sum(ps_0);

            auto vt = v_packed + t * VPackedWidth;
            float vnorm = static_cast<float>(v_norms[t]);
            { float om = lmax_0;
              lmax_0 = max(lmax_0, s_0);
              float rs = exp(om - lmax_0);
              float w = exp(s_0 - lmax_0);
              lsum_0 = lsum_0 * rs + w;
              for (int i = 0; i < DimsPerLane; i++) {
                  lacc_0[i] *= rs;
                  if (v_valids[i]) {
                      uint vv = (vt[v_words[i]] >> v_offs[i]);
                      if (v_spills[i]) vv |= vt[v_words[i]+1] << (4 - (v_offs[i]+4-32));
                      lacc_0[i] += w * val_codebook[vv & 15u] * vnorm;
                  }
              }
            }
        }

        float inv_sum_0 = 1.0f / lsum_0;
        for (int i = 0; i < DimsPerLane; i++)
            lacc_0[i] *= inv_sum_0;

        threadgroup float shared_rot[1 * Dim];
        for (int i = 0, vd = lane; i < DimsPerLane; i++, vd += 32)
            if (vd < Dim) shared_rot[0 * Dim + vd] = lacc_0[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply dense rotation: output in original space
        for (int i = 0, vd = lane; i < DimsPerLane; i++, vd += 32) {
            if (vd < Dim) {
                float rot_val = 0.0f;
                auto rot_row = rotation_t + vd * Dim;
                for (int j = 0; j < (int)Dim; j++)
                    rot_val += shared_rot[0 * Dim + j] * rot_row[j];
                out[(bh * RepeatCount + 0) * Dim + vd] = rot_val;
            }
        }
}

template [[host_name("custom_kernel_turboquant_fully_fused_4_r1__64_1_2_8_2_8_float16_t_float16_t_float16_tc_uint32_t_float16_tc_uint32_tc_float16_tc_uint32_t_float_floatc_float_float_float")]] [[kernel]] decltype(custom_kernel_turboquant_fully_fused_4_r1__64_1_2_8_2_8_float16_t_float16_t_float16_tc_uint32_t_float16_tc_uint32_tc_float16_tc_uint32_t_float_floatc_float_float_float<64, 1, 2, 8, 2, 8>) custom_kernel_turboquant_fully_fused_4_r1__64_1_2_8_2_8_float16_t_float16_t_float16_tc_uint32_t_float16_tc_uint32_tc_float16_tc_uint32_t_float_floatc_float_float_float<64, 1, 2, 8, 2, 8>;
