// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:1744.
// Representative call: multi-query: q_rot/q_proj (2,2,64) float16, key_norms/key_res_norms (1,1,2) float16, key_mse (1,1,2,8) uint32, key_signs (1,1,2,2) uint32, codebook (16) float32, scale (1) float32; Dim=64,KMsePackedWidth=8,KSignPackedWidth=2; grid=(32,2,2), threadgroup=(32,1,1); out (2,2,2) float32.
template <int Dim, int KMsePackedWidth, int KSignPackedWidth>
[[kernel]] void custom_kernel_mq_prod_score_4_r2_l2__64_8_2_float16_t_float16_t_float16_tc_uint32_t_float16_tc_uint32_tc_float_floatc_float(
  const device float16_t* q_rot [[buffer(0)]],
  const device float16_t* q_proj [[buffer(1)]],
  const constant float16_t* key_norms [[buffer(2)]],
  const constant int* key_norms_shape [[buffer(3)]],
  const device uint32_t* key_mse [[buffer(4)]],
  const constant float16_t* key_res_norms [[buffer(5)]],
  const constant uint32_t* key_signs [[buffer(6)]],
  const device float* key_codebook [[buffer(7)]],
  const constant float* key_scale [[buffer(8)]],
  device float* out [[buffer(9)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

            auto lane = thread_position_in_grid.x;
            auto ri = thread_position_in_grid.y;
            auto n = thread_position_in_grid.z;
            auto tc = key_norms_shape[2];
            auto kh = key_norms_shape[1];
            auto b = n / (kh * tc);
            auto rem = n % (kh * tc);
            auto h = rem / tc;
            auto t = rem % tc;
            if (ri >= 2) return;

            auto mt = key_mse + ((b*kh+h)*tc+t) * KMsePackedWidth;
            auto st = key_signs + ((b*kh+h)*tc+t) * KSignPackedWidth;
            float kn = static_cast<float>(key_norms[(b*kh+h)*tc+t]);
            float ksr = kn * key_scale[0] * static_cast<float>(key_res_norms[(b*kh+h)*tc+t]);

            // Unpack key ONCE
            float kc[2], ksf[2];
            for (int i=0, d=lane; d < Dim; i++, d+=32) {
                int bo = d * 4;
                uint idx = (mt[bo>>5] >> (bo&31));
                if (((bo&31)+4) > 32) idx |= mt[(bo>>5)+1] << (4 - ((bo&31)+4-32));
                idx &= 15u;
                kc[i] = key_codebook[idx];
                uint sb = (st[d>>5] >> (d&31)) & 1u;
                ksf[i] = sb ? 1.0f : -1.0f;
            }

            // Loop over L queries, reusing unpacked key
            auto bq = (b*kh+h) * 2 + ri;
            for (int l = 0; l < 2; l++) {
                float ps = 0.0f;
                for (int i=0, d=lane; d < Dim; i++, d+=32) {
                    ps += kn * static_cast<float>(q_rot[(bq*2+l)*Dim+d]) * kc[i]
                        + ksr * static_cast<float>(q_proj[(bq*2+l)*Dim+d]) * ksf[i];
                }
                float s = simd_sum(ps);
                if (lane == 0)
                    out[((b*kh+h)*2+ri)*2*tc + l*tc + t] = s;
            }
        
}

template [[host_name("custom_kernel_mq_prod_score_4_r2_l2__64_8_2_float16_t_float16_t_float16_tc_uint32_t_float16_tc_uint32_tc_float_floatc_float")]] [[kernel]] decltype(custom_kernel_mq_prod_score_4_r2_l2__64_8_2_float16_t_float16_t_float16_tc_uint32_t_float16_tc_uint32_tc_float_floatc_float<64, 8, 2>) custom_kernel_mq_prod_score_4_r2_l2__64_8_2_float16_t_float16_t_float16_tc_uint32_t_float16_tc_uint32_tc_float_floatc_float<64, 8, 2>;
