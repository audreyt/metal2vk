// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:2617.
// Representative call: fused mse sdpa two-pass 2: out_acc (128,64) float32, out_sums/out_maxs (128) float32; Dim=64,Blocks=64; grid=(2048,1,1), threadgroup=(1024,1,1); out (2,64) float32.
template <int Dim, int Blocks>
[[kernel]] void custom_kernel_turboquant_mse_sdpa_2pass2__64_64_float_float_float_float(
  const device float* partials [[buffer(0)]],
  const device float* sums [[buffer(1)]],
  const device float* maxs [[buffer(2)]],
  device float* out [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int elem_per_thread = Dim / BD;
        typedef float U;

        thread U o[elem_per_thread] = {};
        threadgroup U outputs[BN * BD];

        auto head_idx = threadgroup_position_in_grid.x;
        auto simd_gid = simdgroup_index_in_threadgroup;
        auto simd_lid = thread_index_in_simdgroup;

        auto p = partials + head_idx * Blocks * Dim + simd_gid * Dim +
                 simd_lid * elem_per_thread;
        auto s = sums + head_idx * Blocks;
        auto m = maxs + head_idx * Blocks;

        U max_score = -INFINITY;
        U sum_exp_score = 0;

        // Each simdgroup handles a subset of blocks (stride BN)
        for (int b = simd_gid; b < Blocks; b += BN) {
            U block_max = m[b];
            U block_sum = s[b];

            U new_max = max(max_score, block_max);
            U factor = fast::exp(max_score - new_max);
            U block_factor = fast::exp(block_max - new_max);

            sum_exp_score = sum_exp_score * factor + block_sum * block_factor;
            for (int i = 0; i < elem_per_thread; i++) {
                o[i] = o[i] * factor +
                    partials[(head_idx * Blocks + b) * Dim +
                             simd_lid * elem_per_thread + i] * block_factor;
            }
            max_score = new_max;
        }

        // Cross-simdgroup reduction (same as sdpa_vector_2pass_2)
        threadgroup U sg_maxs[BN];
        threadgroup U sg_sums[BN];
        if (simd_lid == 0) {
            sg_maxs[simd_gid] = max_score;
            sg_sums[simd_gid] = sum_exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        U sg_max = sg_maxs[simd_lid];
        U new_max = simd_max(sg_max);
        U factor = fast::exp(sg_max - new_max);
        U total_sum = simd_sum(sg_sums[simd_lid] * factor);

        U my_factor = fast::exp(max_score - new_max);

        for (int i = 0; i < elem_per_thread; i++) {
            outputs[simd_lid * BD + simd_gid] = o[i] * my_factor;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
            o[i] = total_sum > 0 ? o[i] / total_sum : 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_lid == 0) {
            for (int i = 0; i < elem_per_thread; i++)
                out[head_idx * Dim + simd_gid * elem_per_thread + i] =
                    static_cast<U>(o[i]);
        }
    
}

template [[host_name("custom_kernel_turboquant_mse_sdpa_2pass2__64_64_float_float_float_float")]] [[kernel]] decltype(custom_kernel_turboquant_mse_sdpa_2pass2__64_64_float_float_float_float<64, 64>) custom_kernel_turboquant_mse_sdpa_2pass2__64_64_float_float_float_float<64, 64>;
