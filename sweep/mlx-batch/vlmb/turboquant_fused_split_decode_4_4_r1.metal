// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:3346.
// Representative call: split decode (4,4): q_rot/q_proj low+high (1,1,1,64) float16, key low/high norms and residual norms (1,1,2) float16, mse/indices (1,1,2,8) uint32, signs (1,1,2,2) uint32, codebooks (16) float32, scales (1) float32; DimLow=DimHigh=64,RepeatCount=1,TokTileSize=1024, packed widths 8/2; grid=(32,2,1), threadgroup=(32,1,1); out_acc (1,1,1,64) float32, out_sum/out_max (1,1,1) float32.
template <int DimLow, int DimHigh, int RepeatCount, int TokTileSize, int KLMsePackedWidth, int KLSignPackedWidth, int KHMsePackedWidth, int KHSignPackedWidth, int VLPackedWidth, int VHPackedWidth, int VLBits, int VHBits>
[[kernel]] void custom_kernel_turboquant_fused_split_decode_4_4_r1__64_64_1_1024_8_2_8_2_8_8_4_4_float16_t_float16_t_float16_t_float16_t_float16_tc_uint32_t_float16_tc_uint32_tc_float16_tc_uint32_t_float16_tc_uint32_tc_float16_tc_uint32_t_float16_tc_uint32_t_float_float_floatc_floatc_float_float_float_float_float(
  const device float16_t* q_rot_low [[buffer(0)]],
  const device float16_t* q_proj_low [[buffer(1)]],
  const device float16_t* q_rot_high [[buffer(2)]],
  const device float16_t* q_proj_high [[buffer(3)]],
  const constant float16_t* key_low_norms [[buffer(4)]],
  const constant int* key_low_norms_shape [[buffer(5)]],
  const device uint32_t* key_low_mse [[buffer(6)]],
  const constant float16_t* key_low_res_norms [[buffer(7)]],
  const constant uint32_t* key_low_signs [[buffer(8)]],
  const constant float16_t* key_high_norms [[buffer(9)]],
  const device uint32_t* key_high_mse [[buffer(10)]],
  const constant float16_t* key_high_res_norms [[buffer(11)]],
  const constant uint32_t* key_high_signs [[buffer(12)]],
  const constant float16_t* val_low_norms [[buffer(13)]],
  const device uint32_t* val_low_packed [[buffer(14)]],
  const constant float16_t* val_high_norms [[buffer(15)]],
  const device uint32_t* val_high_packed [[buffer(16)]],
  const device float* key_low_codebook [[buffer(17)]],
  const device float* key_high_codebook [[buffer(18)]],
  const constant float* key_low_scale [[buffer(19)]],
  const constant float* key_high_scale [[buffer(20)]],
  const device float* val_low_codebook [[buffer(21)]],
  const device float* val_high_codebook [[buffer(22)]],
  device float* out_acc [[buffer(23)]],
  device float* out_sum [[buffer(24)]],
  device float* out_max [[buffer(25)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        auto lane = thread_position_in_grid.x;
        auto val_tile = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        int val_dim = val_tile * 32 + lane;

        auto token_count = key_low_norms_shape[2];
        auto kv_heads = key_low_norms_shape[1];
        auto num_tok_tiles = (token_count + TokTileSize - 1) / TokTileSize;
        auto bh = n / num_tok_tiles;
        auto tok_tile = n % num_tok_tiles;
        auto b = bh / kv_heads;
        auto h = bh % kv_heads;
        auto base = (b * kv_heads + h);

        int t_start = tok_tile * TokTileSize;
        int t_end = min(t_start + TokTileSize, (int)token_count);

        auto kl_norms = key_low_norms + base * token_count;
        auto kl_mse = key_low_mse + base * token_count * KLMsePackedWidth;
        auto kl_res = key_low_res_norms + base * token_count;
        auto kl_signs = key_low_signs + base * token_count * KLSignPackedWidth;
        auto kh_norms = key_high_norms + base * token_count;
        auto kh_mse = key_high_mse + base * token_count * KHMsePackedWidth;
        auto kh_res = key_high_res_norms + base * token_count;
        auto kh_signs = key_high_signs + base * token_count * KHSignPackedWidth;

        auto vl_norms = val_low_norms + base * token_count;
        auto vl_packed = val_low_packed + base * token_count * VLPackedWidth;
        auto vh_norms = val_high_norms + base * token_count;
        auto vh_packed = val_high_packed + base * token_count * VHPackedWidth;

        // Value bit offset for this lane's dim
        bool is_low_val = val_dim < DimLow;
        int vd_local = is_low_val ? val_dim : (val_dim - DimLow);
        int v_bits = is_low_val ? VLBits : VHBits;
        int v_bo = vd_local * v_bits;
        int v_word = v_bo / 32;
        int v_off = v_bo % 32;
        uint v_mask = (1u << v_bits) - 1u;
        bool v_spills = (v_off + v_bits > 32);
        bool v_valid = val_dim < (DimLow + DimHigh);

        auto qrl_0 = q_rot_low + (base * RepeatCount + 0) * DimLow;
        auto qpl_0 = q_proj_low + (base * RepeatCount + 0) * DimLow;
        auto qrh_0 = q_rot_high + (base * RepeatCount + 0) * DimHigh;
        auto qph_0 = q_proj_high + (base * RepeatCount + 0) * DimHigh;
        float lmax_0 = -INFINITY;
        float lsum_0 = 0.0f;
        float lacc_0 = 0.0f;

        for (int t = t_start; t < t_end; t++) {
            float ps_0 = 0.0f;
            {
                auto mse_t = kl_mse + t * KLMsePackedWidth;
                auto sign_t = kl_signs + t * KLSignPackedWidth;
                float kn = static_cast<float>(kl_norms[t]);
                float ksr = kn * key_low_scale[0] * static_cast<float>(kl_res[t]);
                for (int d = lane; d < DimLow; d += 32) {
                    int bo = d * 3;
                    uint idx = (mse_t[bo >> 5] >> (bo & 31));
                    if (((bo & 31) + 3) > 32) idx |= mse_t[(bo >> 5) + 1] << (3 - ((bo & 31) + 3 - 32));
                    idx &= 7u;
                    float code = key_low_codebook[idx];
                    uint sb = (sign_t[d >> 5] >> (d & 31)) & 1u;
                    ps_0 += kn * static_cast<float>(qrl_0[d]) * code + ksr * (sb ? static_cast<float>(qpl_0[d]) : -static_cast<float>(qpl_0[d]));
                }
            }
            {
                auto mse_t = kh_mse + t * KHMsePackedWidth;
                auto sign_t = kh_signs + t * KHSignPackedWidth;
                float kn = static_cast<float>(kh_norms[t]);
                float ksr = kn * key_high_scale[0] * static_cast<float>(kh_res[t]);
                for (int d = lane; d < DimHigh; d += 32) {
                    int bo = d * 3;
                    uint idx = (mse_t[bo >> 5] >> (bo & 31));
                    if (((bo & 31) + 3) > 32) idx |= mse_t[(bo >> 5) + 1] << (3 - ((bo & 31) + 3 - 32));
                    idx &= 7u;
                    float code = key_high_codebook[idx];
                    uint sb = (sign_t[d >> 5] >> (d & 31)) & 1u;
                    ps_0 += kn * static_cast<float>(qrh_0[d]) * code + ksr * (sb ? static_cast<float>(qph_0[d]) : -static_cast<float>(qph_0[d]));
                }
            }
            float s_0 = simd_sum(ps_0);

            float v_code = 0.0f;
            if (v_valid) {
                if (is_low_val) {
                    auto vt = vl_packed + t * VLPackedWidth;
                    uint vv = (vt[v_word] >> v_off);
                    if (v_spills) vv |= vt[v_word + 1] << (v_bits - (v_off + v_bits - 32));
                    v_code = val_low_codebook[vv & v_mask] * static_cast<float>(vl_norms[t]);
                } else {
                    auto vt = vh_packed + t * VHPackedWidth;
                    uint vv = (vt[v_word] >> v_off);
                    if (v_spills) vv |= vt[v_word + 1] << (v_bits - (v_off + v_bits - 32));
                    v_code = val_high_codebook[vv & v_mask] * static_cast<float>(vh_norms[t]);
                }
            }
            float om_0 = lmax_0;
            lmax_0 = max(lmax_0, s_0);
            float rs_0 = exp(om_0 - lmax_0);
            float w_0 = exp(s_0 - lmax_0);
            lsum_0 = lsum_0 * rs_0 + w_0;
            lacc_0 = lacc_0 * rs_0 + w_0 * v_code;
        }

        int out_stride = (DimLow + DimHigh);
        if (v_valid) {
            out_acc[((bh * num_tok_tiles + tok_tile) * RepeatCount + 0) * out_stride + val_dim] = lacc_0;
        }
        if (val_dim == 0) {
            int sm_base = (bh * num_tok_tiles + tok_tile) * RepeatCount + 0;
            out_sum[sm_base] = lsum_0;
            out_max[sm_base] = lmax_0;
        }
}

template [[host_name("custom_kernel_turboquant_fused_split_decode_4_4_r1__64_64_1_1024_8_2_8_2_8_8_4_4_float16_t_float16_t_float16_t_float16_t_float16_tc_uint32_t_float16_tc_uint32_tc_float16_tc_uint32_t_float16_tc_uint32_tc_float16_tc_uint32_t_float16_tc_uint32_t_float_float_floatc_floatc_float_float_float_float_float")]] [[kernel]] decltype(custom_kernel_turboquant_fused_split_decode_4_4_r1__64_64_1_1024_8_2_8_2_8_8_4_4_float16_t_float16_t_float16_t_float16_t_float16_tc_uint32_t_float16_tc_uint32_tc_float16_tc_uint32_t_float16_tc_uint32_tc_float16_tc_uint32_t_float16_tc_uint32_t_float_float_floatc_floatc_float_float_float_float_float<64, 64, 1, 1024, 8, 2, 8, 2, 8, 8, 4, 4>) custom_kernel_turboquant_fused_split_decode_4_4_r1__64_64_1_1024_8_2_8_2_8_8_4_4_float16_t_float16_t_float16_t_float16_t_float16_tc_uint32_t_float16_tc_uint32_tc_float16_tc_uint32_t_float16_tc_uint32_tc_float16_tc_uint32_t_float16_tc_uint32_t_float_float_floatc_floatc_float_float_float_float_float<64, 64, 1, 1024, 8, 2, 8, 2, 8, 8, 4, 4>;
