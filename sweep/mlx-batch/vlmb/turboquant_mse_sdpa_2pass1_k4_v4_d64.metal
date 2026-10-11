// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:2525.
// Representative call: fused mse sdpa two-pass 1: same inputs as single pass; Dim=64,RepeatCount=2,Blocks=64,KPackedWidth=8,VPackedWidth=8; grid=(32,2,64), threadgroup=(32,2,1); out_acc (128,64) float32, out_sums/out_maxs (128) float32.
template <int Dim, int RepeatCount, int Blocks, int KPackedWidth, int VPackedWidth>
[[kernel]] void custom_kernel_turboquant_mse_sdpa_2pass1_k4_v4_d64__64_2_64_8_8_float16_t_float16_tc_uint32_t_float_float16_tc_uint32_t_float_float_float_float(
  const device float16_t* queries [[buffer(0)]],
  const constant float16_t* key_norms [[buffer(1)]],
  const constant int* key_norms_shape [[buffer(2)]],
  const device uint32_t* key_packed [[buffer(3)]],
  const device float* key_codebook [[buffer(4)]],
  const constant float16_t* val_norms [[buffer(5)]],
  const device uint32_t* val_packed [[buffer(6)]],
  const device float* val_codebook [[buffer(7)]],
  device float* out_acc [[buffer(8)]],
  device float* out_sums [[buffer(9)]],
  device float* out_maxs [[buffer(10)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

        constexpr int BD = 32;
        constexpr int qk_per_thread = Dim / BD;
        constexpr int v_per_thread = Dim / BD;
        constexpr uint k_mask = 15u;
        constexpr uint v_mask = 15u;
        constexpr int k_bits = 4;
        constexpr int v_bits = 4;
        typedef float U;

        // Thread identity  -  matches sdpa_vector_2pass_1 layout
        auto kv_head_idx = threadgroup_position_in_grid.x;
        auto batch_idx = threadgroup_position_in_grid.y;
        auto block_idx = threadgroup_position_in_grid.z;
        auto simd_lid = thread_index_in_simdgroup;
        auto gqa_idx = thread_position_in_threadgroup.y;  // which q_head within kv_head

        auto token_count = key_norms_shape[2];
        auto kv_heads = key_norms_shape[1];
        auto bh = batch_idx * kv_heads + kv_head_idx;
        auto bqh = batch_idx * kv_heads * RepeatCount + kv_head_idx * RepeatCount + gqa_idx;

        auto k_nm = key_norms + bh * token_count;
        auto k_pk = key_packed + bh * token_count * KPackedWidth;
        auto v_nm = val_norms + bh * token_count;
        auto v_pk = val_packed + bh * token_count * VPackedWidth;

        // Load pre-rotated query
        thread U q[qk_per_thread];
        auto qr = queries + bqh * Dim + simd_lid * qk_per_thread;
        for (int i = 0; i < qk_per_thread; i++)
            q[i] = static_cast<U>(qr[i]);

        thread U o[v_per_thread] = {};
        U max_score = -INFINITY;
        U sum_exp_score = 0;

        // Byte/bit offset for this lane's first element (constant across all tokens)
        int k_bit_start = simd_lid * qk_per_thread * k_bits;
        int v_bit_start = simd_lid * v_per_thread * v_bits;
        int k_byte_base = k_bit_start >> 3;
        int v_byte_base = v_bit_start >> 3;
        
        

        // KV loop: stride by blocks (each block handles a different subset)
        for (int t = block_idx; t < (int)token_count; t += Blocks) {
            U kn = static_cast<U>(k_nm[t]);

            // Key score  -  unrolled byte extraction
            auto kb = (const device uint8_t*)(k_pk + t * KPackedWidth) + k_byte_base;
            U score = q[0] * key_codebook[kb[0] & 15u]
                + q[1] * key_codebook[(kb[0] >> 4) & 15u];
            score = simd_sum(score) * kn;

            auto vb = (const device uint8_t*)(v_pk + t * VPackedWidth) + v_byte_base;
            U vn = static_cast<U>(v_nm[t]);

            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;

                        o[0] = o[0] * factor + exp_score * val_codebook[vb[0] & 15u] * vn;
            o[1] = o[1] * factor + exp_score * val_codebook[(vb[0] >> 4) & 15u] * vn;
        }

        // Write partial results for this block
        if (simd_lid == 0) {
            out_sums[bqh * Blocks + block_idx] = sum_exp_score;
            out_maxs[bqh * Blocks + block_idx] = max_score;
        }
        for (int i = 0; i < v_per_thread; i++)
            out_acc[(bqh * Blocks + block_idx) * Dim + simd_lid * v_per_thread + i] =
                static_cast<U>(o[i]);
    
}

template [[host_name("custom_kernel_turboquant_mse_sdpa_2pass1_k4_v4_d64__64_2_64_8_8_float16_t_float16_tc_uint32_t_float_float16_tc_uint32_t_float_float_float_float")]] [[kernel]] decltype(custom_kernel_turboquant_mse_sdpa_2pass1_k4_v4_d64__64_2_64_8_8_float16_t_float16_tc_uint32_t_float_float16_tc_uint32_t_float_float_float_float<64, 2, 64, 8, 8>) custom_kernel_turboquant_mse_sdpa_2pass1_k4_v4_d64__64_2_64_8_8_float16_t_float16_tc_uint32_t_float_float16_tc_uint32_t_float_float_float_float<64, 2, 64, 8, 8>;
