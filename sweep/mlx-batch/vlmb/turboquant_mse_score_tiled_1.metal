// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:161.
// Representative call: tiled: q_rot (1,1,1,64) float16, norms (1,1,2) float16, packed (1,1,2,8) uint32, codebook (16) float32; Dim=64,DimsPerLane=2,Bits=4,PackedWidth=8,RepeatCount=1,TokTileSize=64; grid=(32,1,1), threadgroup=(32,1,1); out (1,1,1,2) float32.
template <int Dim, int DimsPerLane, int Bits, int PackedWidth, int RepeatCount, int TokTileSize>
[[kernel]] void custom_kernel_turboquant_mse_score_tiled_1__64_2_4_8_1_64_float16_t_float16_tc_uint32_t_float_float(
  const device float16_t* q_rot [[buffer(0)]],
  const constant float16_t* norms [[buffer(1)]],
  const constant int* norms_shape [[buffer(2)]],
  const device uint32_t* packed [[buffer(3)]],
  const device float* codebook [[buffer(4)]],
  device float* out [[buffer(5)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        auto lane = thread_position_in_grid.x;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto num_tok_tiles = (token_count + TokTileSize - 1) / TokTileSize;
        auto bh = n / num_tok_tiles;
        auto tok_tile = n % num_tok_tiles;
        int t_start = tok_tile * TokTileSize;
        int t_end = min(t_start + TokTileSize, (int)token_count);

        auto q_base = q_rot + (bh * RepeatCount) * Dim;
        auto pk = packed + bh * token_count * PackedWidth;
        auto nm = norms + bh * token_count;

        // Preload query values into registers (constant across all tokens)
        float qr_0[DimsPerLane];
        for (int i = 0, d = lane; i < DimsPerLane; i++, d += 32)
            qr_0[i] = (d < Dim) ? static_cast<float>(q_base[0 * Dim + d]) : 0.0f;

        // Precompute bit offsets for key unpacking
        int k_words[DimsPerLane], k_offs[DimsPerLane];
        bool k_spills[DimsPerLane];
        for (int i = 0, d = lane; i < DimsPerLane; i++, d += 32) {
            int bo = d * Bits;
            k_words[i] = bo / 32;
            k_offs[i] = bo % 32;
            k_spills[i] = (bo % 32 + Bits) > 32;
        }
        constexpr uint mask = (1u << Bits) - 1u;

        // Token loop: key data changes, queries stay in registers
        for (int t = t_start; t < t_end; t++) {
            auto pt = pk + t * PackedWidth;
            float kn = static_cast<float>(nm[t]);

            float ps_0 = 0.0f;
            for (int i = 0; i < DimsPerLane; i++) {
                uint vv = (pt[k_words[i]] >> k_offs[i]);
                if (k_spills[i]) vv |= pt[k_words[i]+1] << (Bits - (k_offs[i]+Bits-32));
                float code = codebook[vv & mask];
                ps_0 += qr_0[i] * code;
            }

            float s_0 = simd_sum(ps_0);
            if (thread_index_in_simdgroup == 0) {
                out[(bh * RepeatCount + 0) * token_count + t] = kn * s_0;
            }
        }
}

template [[host_name("custom_kernel_turboquant_mse_score_tiled_1__64_2_4_8_1_64_float16_t_float16_tc_uint32_t_float_float")]] [[kernel]] decltype(custom_kernel_turboquant_mse_score_tiled_1__64_2_4_8_1_64_float16_t_float16_tc_uint32_t_float_float<64, 2, 4, 8, 1, 64>) custom_kernel_turboquant_mse_score_tiled_1__64_2_4_8_1_64_float16_t_float16_tc_uint32_t_float_float<64, 2, 4, 8, 1, 64>;
