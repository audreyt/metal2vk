// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:1859.
// Representative call: single-tile value: weights (1,2,2) float32, norms (1,1,2) float16, indices (1,1,2,8) uint32, codebook (16) float32; Dim=64,RepeatCount=2,TokTileSize=1024,DimsPerLane=2,PackedWidth=8; grid=(64,1,1), threadgroup=(64,1,1); out (1,2,64) float32.
template <int Dim, int RepeatCount, int TokTileSize, int DimsPerLane, int PackedWidth>
[[kernel]] void custom_kernel_turboquant_single_tile_value_4_r2__64_2_1024_2_8_floatc_float16_tc_uint32_t_float_float(
  const constant float* weights [[buffer(0)]],
  const constant float16_t* norms [[buffer(1)]],
  const constant int* norms_shape [[buffer(2)]],
  const device uint32_t* packed [[buffer(3)]],
  const device float* codebook [[buffer(4)]],
  device float* out [[buffer(5)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        auto dim = thread_position_in_grid.x;
        auto n = thread_position_in_grid.z;
        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto num_tok_tiles = (token_count + TokTileSize - 1) / TokTileSize;
        auto bh = n / num_tok_tiles;
        auto tok_tile = n % num_tok_tiles;
        int t_start = tok_tile * TokTileSize;
        int t_end = min(t_start + TokTileSize, (int)token_count);

        auto wt = weights + bh * RepeatCount * token_count;
        auto nm = norms + bh * token_count;
        auto pk = packed + bh * token_count * PackedWidth;

        int bo = dim * 4;
        int v_word = bo / 32;
        int v_shift = bo % 32;
        bool v_spill = (bo % 32 + 4) > 32;

        float acc_0 = 0.0f;
        float acc_1 = 0.0f;

        for (int t = t_start; t < t_end; t++) {
            auto pt = pk + t * PackedWidth;
            uint vv = (pt[v_word] >> v_shift);
            if (v_spill) vv |= pt[v_word+1] << (4 - (v_shift+4-32));
            float val = codebook[vv & 15u] * static_cast<float>(nm[t]);

            acc_0 += wt[0*token_count+t] * val;
            acc_1 += wt[1*token_count+t] * val;
        }

        if (dim < Dim) out[((bh*num_tok_tiles+tok_tile)*RepeatCount+0)*Dim+dim] = acc_0;
        if (dim < Dim) out[((bh*num_tok_tiles+tok_tile)*RepeatCount+1)*Dim+dim] = acc_1;
}

template [[host_name("custom_kernel_turboquant_single_tile_value_4_r2__64_2_1024_2_8_floatc_float16_tc_uint32_t_float_float")]] [[kernel]] decltype(custom_kernel_turboquant_single_tile_value_4_r2__64_2_1024_2_8_floatc_float16_tc_uint32_t_float_float<64, 2, 1024, 2, 8>) custom_kernel_turboquant_single_tile_value_4_r2__64_2_1024_2_8_floatc_float16_tc_uint32_t_float_float<64, 2, 1024, 2, 8>;
