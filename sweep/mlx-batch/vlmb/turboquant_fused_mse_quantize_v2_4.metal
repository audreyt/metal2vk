// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:2981.
// Representative call: fused mse quantize: flat (1,64) float16, rotation (64,64) float32, midpoints (15) float32; Dim=64,DimsPerLane=2,PackedWidth=8; grid=(64,1,1), threadgroup=(64,1,1); out norms (1) float16, packed (1,8) uint32.
template <int Dim, int DimsPerLane, int PackedWidth>
[[kernel]] void custom_kernel_turboquant_fused_mse_quantize_v2_4__64_2_8_float16_t_float_float_float16_t_uint32_t(
  const device float16_t* vectors [[buffer(0)]],
  const device float* rotation [[buffer(1)]],
  const device float* midpoints [[buffer(2)]],
  device float16_t* out_norms [[buffer(3)]],
  device uint32_t* out_packed [[buffer(4)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

        auto d = thread_position_in_threadgroup.x;  // 0..Dim-1, one dim per thread
        auto bh = threadgroup_position_in_grid.x;
        auto sg_id = simdgroup_index_in_threadgroup;
        auto sg_lid = thread_index_in_simdgroup;
        auto vec_ptr = vectors + bh * Dim;

        // Step 1: Load & compute norm (2-stage cross-simdgroup reduction)
        float val = (d < Dim) ? static_cast<float>(vec_ptr[d]) : 0.0f;
        float sq = val * val;
        float sg_sum = simd_sum(sq);

        constexpr int n_sg = (Dim + 31) / 32;
        threadgroup float sg_norms[32];
        if (sg_lid == 0) sg_norms[sg_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float total_sq = (sg_id == 0 && sg_lid < n_sg) ? sg_norms[sg_lid] : 0.0f;
        total_sq = simd_sum(total_sq);
        // Broadcast norm to all threads
        if (sg_id == 0 && sg_lid == 0) sg_norms[0] = total_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float norm = sqrt(sg_norms[0]);
        float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
        if (d == 0) out_norms[bh] = half(norm);

        // Step 2: Unit vector → shared for rotation
        threadgroup float shared[Dim];
        if (d < Dim) shared[d] = val * inv_norm;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: Rotate — 1 dim per thread, 256 FMAs (was 2048 at TG=32)
        float rotated = 0.0f;
        if (d < Dim) {
            auto row = rotation + d * Dim;
            for (int j = 0; j < (int)Dim; j++)
                rotated += shared[j] * row[j];
        }

        // Step 4: Comparison-based quantize
        threadgroup uint shared_idx[Dim];
        uint idx = 0;
        if (d < Dim) {
            for (int m = 0; m < 15; m++)
                idx += (rotated > midpoints[m]) ? 1u : 0u;
            shared_idx[d] = idx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5: Pack indices — thread-per-word, race-free, no atomics.
        // Each thread d (d < PackedWidth) walks the dims whose
        // [i*bits, (i+1)*bits) range intersects [32d, 32(d+1)) and
        // accumulates them into a private register, then writes the
        // word once. This avoids both the original `|=` race and the
        // GPU watchdog hangs that high-contention atomic_or sees on
        // some Metal devices.
        if (d < PackedWidth) {
            uint w_val = 0u;
            int word_start = (int)d * 32;
            int i_min = word_start / 4;
            int i_max = (word_start + 31) / 4;
            if (i_max >= (int)Dim) i_max = (int)Dim - 1;
            for (int i = i_min; i <= i_max; i++) {
                uint idx_val = shared_idx[i] & 15u;
                int bit_off = i * 4 - word_start;
                if (bit_off >= 0) {
                    w_val |= idx_val << bit_off;
                } else {
                    w_val |= idx_val >> (-bit_off);
                }
            }
            out_packed[bh * PackedWidth + d] = w_val;
        }
    
}

template [[host_name("custom_kernel_turboquant_fused_mse_quantize_v2_4__64_2_8_float16_t_float_float_float16_t_uint32_t")]] [[kernel]] decltype(custom_kernel_turboquant_fused_mse_quantize_v2_4__64_2_8_float16_t_float_float_float16_t_uint32_t<64, 2, 8>) custom_kernel_turboquant_fused_mse_quantize_v2_4__64_2_8_float16_t_float_float_float16_t_uint32_t<64, 2, 8>;
