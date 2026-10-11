// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:1713.
// Representative call: q_rot/q_proj (1,1,1,64) float16, key_norms/val_norms (1,1,2) float16, key_mse/val_packed (1,1,2,8) uint32, key_res_norms (1,1,2) float16, key_signs (1,1,2,2) uint32, codebooks (16) float32, key_scale (1) float32; Dim=64,RepeatCount=1,TokTileSize=1024,KMsePackedWidth=8,KSignPackedWidth=2,VPackedWidth=8,ValBits=4; grid=(32,1,1), threadgroup=(32,1,1); out_acc (1,1,1,64) float32, out_sum/out_max (1,1,1) float32.
template <int Dim, int RepeatCount, int TokTileSize, int KMsePackedWidth, int KSignPackedWidth, int VPackedWidth, int ValBits>
[[kernel]] void custom_kernel_turboquant_fused_integer_decode_4_r1__64_1_1024_8_2_8_4_float16_t_float16_t_float16_tc_uint32_t_float16_tc_uint32_tc_float16_tc_uint32_t_float_floatc_float_float_float_float(
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
  device float* out_acc [[buffer(12)]],
  device float* out_sum [[buffer(13)]],
  device float* out_max [[buffer(14)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        auto lane = thread_position_in_grid.x;
        auto val_tile = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        int val_dim = val_tile * 32 + lane;

        auto token_count = key_norms_shape[2];
        auto kv_heads = key_norms_shape[1];
        auto num_tok_tiles = (token_count + TokTileSize - 1) / TokTileSize;
        auto bh = n / num_tok_tiles;
        auto tok_tile = n % num_tok_tiles;
        auto b = bh / kv_heads;
        auto h = bh % kv_heads;
        auto base = (b * kv_heads + h);

        int t_start = tok_tile * TokTileSize;
        int t_end = min(t_start + TokTileSize, (int)token_count);

        auto k_norms = key_norms + base * token_count;
        auto k_mse = key_mse + base * token_count * KMsePackedWidth;
        auto k_res = key_res_norms + base * token_count;
        auto k_signs = key_signs + base * token_count * KSignPackedWidth;
        auto v_norms = val_norms + base * token_count;
        auto v_packed = val_packed + base * token_count * VPackedWidth;

        bool v_valid = val_dim < Dim;
        int v_bo = val_dim * ValBits;
        int v_word = v_bo / 32;
        int v_off = v_bo % 32;
        bool v_spills = (v_off + ValBits > 32);

        auto qr_0 = q_rot + (base * RepeatCount + 0) * Dim;
        auto qp_0 = q_proj + (base * RepeatCount + 0) * Dim;
        float lmax_0 = -INFINITY;
        float lsum_0 = 0.0f;
        float lacc_0 = 0.0f;

        for (int t = t_start; t < t_end; t++) {
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

            float v_code = 0.0f;
            if (v_valid) {
                auto vt = v_packed + t * VPackedWidth;
                uint vv = (vt[v_word] >> v_off);
                if (v_spills) vv |= vt[v_word + 1] << (ValBits - (v_off + ValBits - 32));
                v_code = val_codebook[vv & 15u] * static_cast<float>(v_norms[t]);
            }
            float om_0 = lmax_0;
            lmax_0 = max(lmax_0, s_0);
            float rs_0 = exp(om_0 - lmax_0);
            float w_0 = exp(s_0 - lmax_0);
            lsum_0 = lsum_0 * rs_0 + w_0;
            lacc_0 = lacc_0 * rs_0 + w_0 * v_code;
        }

        int out_stride = Dim;
        if (v_valid) {
            out_acc[((bh * num_tok_tiles + tok_tile) * RepeatCount + 0) * out_stride + val_dim] = lacc_0;
        }
        if (val_dim == 0) {
            int sm_base = (bh * num_tok_tiles + tok_tile) * RepeatCount + 0;
            out_sum[sm_base] = lsum_0;
            out_max[sm_base] = lmax_0;
        }
}

template [[host_name("custom_kernel_turboquant_fused_integer_decode_4_r1__64_1_1024_8_2_8_4_float16_t_float16_t_float16_tc_uint32_t_float16_tc_uint32_tc_float16_tc_uint32_t_float_floatc_float_float_float_float")]] [[kernel]] decltype(custom_kernel_turboquant_fused_integer_decode_4_r1__64_1_1024_8_2_8_4_float16_t_float16_t_float16_tc_uint32_t_float16_tc_uint32_tc_float16_tc_uint32_t_float_floatc_float_float_float_float<64, 1, 1024, 8, 2, 8, 4>) custom_kernel_turboquant_fused_integer_decode_4_r1__64_1_1024_8_2_8_4_float16_t_float16_t_float16_tc_uint32_t_float16_tc_uint32_tc_float16_tc_uint32_t_float_floatc_float_float_float_float<64, 1, 1024, 8, 2, 8, 4>;
