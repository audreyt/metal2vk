// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:2800.
// Representative call: fused kv quantize: key/vectors (1,64) float16, rotations (64,64) float32, midpoints (15) float32; Dim=64,KPackedWidth=VPackedWidth=2; grid=(64,2,1), threadgroup=(64,1,1); out key/val norms (1) float16, key/val packed (1,2) uint32.
template <int Dim, int KPackedWidth, int VPackedWidth>
[[kernel]] void custom_kernel_turboquant_fused_kv_quantize_k4_v4__64_2_2_float16_t_float16_t_float_float_float_float_float16_t_uint32_t_float16_t_uint32_t(
  const device float16_t* key_vectors [[buffer(0)]],
  const device float16_t* val_vectors [[buffer(1)]],
  const device float* key_rotation [[buffer(2)]],
  const device float* val_rotation [[buffer(3)]],
  const device float* key_midpoints [[buffer(4)]],
  const device float* val_midpoints [[buffer(5)]],
  device float16_t* out_key_norms [[buffer(6)]],
  device uint32_t* out_key_packed [[buffer(7)]],
  device float16_t* out_val_norms [[buffer(8)]],
  device uint32_t* out_val_packed [[buffer(9)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

        auto d = thread_position_in_threadgroup.x;
        auto bh = threadgroup_position_in_grid.x;
        auto is_val = threadgroup_position_in_grid.y;  // 0=key, 1=value
        auto sg_id = simdgroup_index_in_threadgroup;
        auto sg_lid = thread_index_in_simdgroup;

        // Select key or value params based on grid.y
        int bits = is_val ? 4 : 4;
        int n_mid = is_val ? 15 : 15;
        uint idx_mask = is_val ? 15u : 15u;
        int pw = is_val ? VPackedWidth : KPackedWidth;

        // Step 1: Load vector element
        float v;
        if (is_val)
            v = (d < Dim) ? static_cast<float>(val_vectors[bh * Dim + d]) : 0.0f;
        else
            v = (d < Dim) ? static_cast<float>(key_vectors[bh * Dim + d]) : 0.0f;

        // Compute norm (2-stage cross-simdgroup reduction)
        float sq = v * v;
        float sg_sum = simd_sum(sq);
        threadgroup float sg_norms[8];
        if (sg_lid == 0) sg_norms[sg_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_sq = (sg_id == 0 && sg_lid < 8) ? sg_norms[sg_lid] : 0.0f;
        total_sq = simd_sum(total_sq);
        if (sg_id == 0 && sg_lid == 0) sg_norms[0] = total_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float norm = sqrt(sg_norms[0]);
        float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
        if (d == 0) {
            if (is_val) out_val_norms[bh] = half(norm);
            else out_key_norms[bh] = half(norm);
        }

        // Step 2: Unit vector → shared
        threadgroup float shared[Dim];
        if (d < Dim) shared[d] = v * inv_norm;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: Rotate (1 dim per thread, 256 FMAs)
        float rotated = 0.0f;
        if (d < Dim) {
            auto row = is_val ? (val_rotation + d * Dim) : (key_rotation + d * Dim);
            for (int j = 0; j < (int)Dim; j++)
                rotated += shared[j] * row[j];
        }

        // Step 4: Comparison-based quantize
        threadgroup uint shared_idx[Dim];
        uint idx = 0;
        if (d < Dim) {
            if (is_val) {
                for (int m = 0; m < n_mid; m++)
                    idx += (rotated > val_midpoints[m]) ? 1u : 0u;
            } else {
                for (int m = 0; m < n_mid; m++)
                    idx += (rotated > key_midpoints[m]) ? 1u : 0u;
            }
            shared_idx[d] = idx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5: Pack indices — thread-per-word, race-free, no atomics.
        // Each thread d (d < pw) walks the dims whose
        // [i*bits, (i+1)*bits) range intersects [32d, 32(d+1)) and
        // accumulates them into a private register, then writes the
        // word once. `bits` and `pw` are runtime-uniform here so the
        // loop bound is well-defined per dispatch.
        if (d < pw) {
            uint w_val = 0u;
            int word_start = (int)d * 32;
            int i_min = word_start / bits;
            int i_max = (word_start + 31) / bits;
            if (i_max >= (int)Dim) i_max = (int)Dim - 1;
            for (int i = i_min; i <= i_max; i++) {
                uint idx_val = shared_idx[i] & idx_mask;
                int bit_off = i * bits - word_start;
                if (bit_off >= 0) {
                    w_val |= idx_val << bit_off;
                } else {
                    w_val |= idx_val >> (-bit_off);
                }
            }
            if (is_val)
                out_val_packed[bh * pw + d] = w_val;
            else
                out_key_packed[bh * pw + d] = w_val;
        }
    
}

template [[host_name("custom_kernel_turboquant_fused_kv_quantize_k4_v4__64_2_2_float16_t_float16_t_float_float_float_float_float16_t_uint32_t_float16_t_uint32_t")]] [[kernel]] decltype(custom_kernel_turboquant_fused_kv_quantize_k4_v4__64_2_2_float16_t_float16_t_float_float_float_float_float16_t_uint32_t_float16_t_uint32_t<64, 2, 2>) custom_kernel_turboquant_fused_kv_quantize_k4_v4__64_2_2_float16_t_float16_t_float_float_float_float_float16_t_uint32_t_float16_t_uint32_t<64, 2, 2>;
