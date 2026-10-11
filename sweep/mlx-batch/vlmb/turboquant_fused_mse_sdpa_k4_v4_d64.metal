// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:2421.
// Representative call: fused mse sdpa single pass: q_rot_flat (2,64) float16, key/val norms (1,1,2) float16, key/val packed (1,1,2,8) uint32, codebooks (16) float32; Dim=64,RepeatCount=2,KPackedWidth=8,VPackedWidth=8; grid=(2048,1,1), threadgroup=(1024,1,1); out (2,64) float32.
template <int Dim, int RepeatCount, int KPackedWidth, int VPackedWidth>
[[kernel]] void custom_kernel_turboquant_fused_mse_sdpa_k4_v4_d64__64_2_8_8_float16_t_float16_tc_uint32_t_float_float16_tc_uint32_t_float_float(
  const device float16_t* queries [[buffer(0)]],
  const constant float16_t* key_norms [[buffer(1)]],
  const constant int* key_norms_shape [[buffer(2)]],
  const device uint32_t* key_packed [[buffer(3)]],
  const device float* key_codebook [[buffer(4)]],
  const constant float16_t* val_norms [[buffer(5)]],
  const device uint32_t* val_packed [[buffer(6)]],
  const device float* val_codebook [[buffer(7)]],
  device float* out [[buffer(8)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

        constexpr int BN = 32;  // simdgroups per threadgroup
        constexpr int BD = 32;  // lanes per simdgroup
        constexpr int qk_per_thread = Dim / BD;
        constexpr int v_per_thread = Dim / BD;
        constexpr uint k_mask = 15u;
        constexpr uint v_mask = 15u;
        constexpr int k_bits = 4;
        constexpr int v_bits = 4;

        typedef float U;

        // Thread identity  -  grid is total threads, so use threadgroup position
        auto bqh = threadgroup_position_in_grid.x;  // batch*q_head index
        auto simd_gid = simdgroup_index_in_threadgroup;
        auto simd_lid = thread_index_in_simdgroup;

        auto token_count = key_norms_shape[2];
        auto kv_heads = key_norms_shape[1];
        auto bh = bqh / RepeatCount;  // map q_head -> kv_head

        auto k_nm = key_norms + bh * token_count;
        auto k_pk = key_packed + bh * token_count * KPackedWidth;
        auto v_nm = val_norms + bh * token_count;
        auto v_pk = val_packed + bh * token_count * VPackedWidth;

        // Shared memory for cross-simdgroup reduction
        threadgroup U max_scores[BN];
        threadgroup U sum_exp_scores[BN];
        threadgroup U shared[BN * BD];

        // Preload pre-rotated query into registers
        thread U q[qk_per_thread];
        auto qr = queries + bqh * Dim + simd_lid * qk_per_thread;
        for (int i = 0; i < qk_per_thread; i++)
            q[i] = static_cast<U>(qr[i]);

        // Initialize accumulators
        thread U o[v_per_thread] = {};
        U max_score = -INFINITY;
        U sum_exp_score = 0;

        // Byte/bit offset for this lane's first element (constant across all tokens)
        int k_bit_start = simd_lid * qk_per_thread * k_bits;
        int v_bit_start = simd_lid * v_per_thread * v_bits;
        int k_byte_base = k_bit_start >> 3;
        int v_byte_base = v_bit_start >> 3;
        
        

        // KV loop: each simdgroup handles tokens simd_gid, simd_gid+32, ...
        for (int t = simd_gid; t < (int)token_count; t += BN) {
            U kn = static_cast<U>(k_nm[t]);

            // Key score  -  unrolled byte extraction
            auto kb = (const device uint8_t*)(k_pk + t * KPackedWidth) + k_byte_base;
            U score = q[0] * key_codebook[kb[0] & 15u]
                + q[1] * key_codebook[(kb[0] >> 4) & 15u];
            score = simd_sum(score) * kn;

            // Online softmax
            auto vb = (const device uint8_t*)(v_pk + t * VPackedWidth) + v_byte_base;
            U vn = static_cast<U>(v_nm[t]);

            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;

            // Value accumulation  -  unrolled byte extraction
                        o[0] = o[0] * factor + exp_score * val_codebook[vb[0] & 15u] * vn;
            o[1] = o[1] * factor + exp_score * val_codebook[(vb[0] >> 4) & 15u] * vn;
        }

        // Cross-simdgroup reduction (matches MLX SDPA pattern)
        // 1. Communicate max and sum_exp across simdgroups
        if (simd_lid == 0) {
            max_scores[simd_gid] = max_score;
            sum_exp_scores[simd_gid] = sum_exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Reduce max and sum across all simdgroups
        U sg_max = max_scores[simd_lid];
        U new_max = simd_max(sg_max);
        U factor = fast::exp(sg_max - new_max);
        U total_sum = simd_sum(sum_exp_scores[simd_lid] * factor);

        // 3. Rescale this simdgroup's factor for the global max
        U my_factor = fast::exp(max_score - new_max);

        // 4. Transpose-reduce outputs through shared memory
        for (int i = 0; i < v_per_thread; i++) {
            shared[simd_lid * BD + simd_gid] = o[i] * my_factor;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(shared[simd_gid * BD + simd_lid]);
            o[i] = total_sum > 0 ? o[i] / total_sum : 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Write output (in rotated space  -  caller applies inverse rotation)
        if (simd_lid == 0) {
            for (int i = 0; i < v_per_thread; i++) {
                out[bqh * Dim + simd_gid * v_per_thread + i] = static_cast<U>(o[i]);
            }
        }
    
}

template [[host_name("custom_kernel_turboquant_fused_mse_sdpa_k4_v4_d64__64_2_8_8_float16_t_float16_tc_uint32_t_float_float16_tc_uint32_t_float_float")]] [[kernel]] decltype(custom_kernel_turboquant_fused_mse_sdpa_k4_v4_d64__64_2_8_8_float16_t_float16_tc_uint32_t_float_float16_tc_uint32_t_float_float<64, 2, 8, 8>) custom_kernel_turboquant_fused_mse_sdpa_k4_v4_d64__64_2_8_8_float16_t_float16_tc_uint32_t_float_float16_tc_uint32_t_float_float<64, 2, 8, 8>;
