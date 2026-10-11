// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:3122.
// Representative call: fused prod quantize: flat (1,64) float16, rotation (64,64) float32, midpoints (15) float32, codebook (16) float32, combined proj (64,64) float32; Dim=64,DimsPerLane=2,MsePackedWidth=8,SignPackedWidth=2; grid=(32,1,1), threadgroup=(32,1,1); out norms/res_norms (1) float16, mse_packed (1,8) uint32, signs (1,2) uint32.
template <int Dim, int DimsPerLane, int MsePackedWidth, int SignPackedWidth>
[[kernel]] void custom_kernel_turboquant_fused_prod_quantize_4__64_2_8_2_float16_t_float_float_float_float_float16_t_uint32_t_float16_t_uint32_t(
  const device float16_t* vectors [[buffer(0)]],
  const device float* mse_rotation [[buffer(1)]],
  const device float* midpoints [[buffer(2)]],
  const device float* codebook [[buffer(3)]],
  const device float* combined_proj_t [[buffer(4)]],
  device float16_t* out_norms [[buffer(5)]],
  device uint32_t* out_mse_packed [[buffer(6)]],
  device float16_t* out_res_norms [[buffer(7)]],
  device uint32_t* out_signs [[buffer(8)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]) {
        auto lane = thread_position_in_threadgroup.x;
        auto bh = thread_position_in_grid.z;
        auto vec_ptr = vectors + bh * Dim;

        // Step 1: Load & compute norm
        float v[DimsPerLane];
        float partial_sq = 0.0f;
        for (int i = 0, d = lane; i < DimsPerLane; i++, d += 32) {
            float x = (d < Dim) ? static_cast<float>(vec_ptr[d]) : 0.0f;
            v[i] = x;
            partial_sq += x * x;
        }
        float norm = sqrt(simd_sum(partial_sq));
        float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
        if (lane == 0) out_norms[bh] = half(norm);

        threadgroup float shared[Dim];
        for (int i = 0, d = lane; i < DimsPerLane; i++, d += 32)
            if (d < Dim) shared[d] = v[i] * inv_norm;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rotated[DimsPerLane];
        for (int i = 0, d = lane; i < DimsPerLane; i++, d += 32) {
            float sum = 0.0f;
            if (d < Dim) {
                auto row = mse_rotation + d * Dim;
                for (int j = 0; j < (int)Dim; j++)
                    sum += shared[j] * row[j];
            }
            rotated[i] = sum;
        }

        // Step 4: MSE quantize + compute rotated residual
        float rot_residual[DimsPerLane];
        threadgroup uint shared_mse_idx[Dim];
        for (int i = 0, d = lane; i < DimsPerLane; i++, d += 32) {
            uint idx = 0;
            if (d < Dim) {
                for (int m = 0; m < 15; m++)
                    idx += (rotated[i] > midpoints[m]) ? 1u : 0u;
            }
            float estimate = codebook[idx];
            rot_residual[i] = rotated[i] - estimate;
            if (d < Dim) shared_mse_idx[d] = idx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5: Pack MSE indices
        for (int w = lane; w < MsePackedWidth; w += 32) {
            uint word = 0;
            for (int b = 0; b < 32; b += 4) {
                int dim = (w * 32 + b) / 4;
                if (dim < Dim) word |= (shared_mse_idx[dim] & 15u) << b;
            }
            out_mse_packed[bh * MsePackedWidth + w] = word;
        }

        // Step 6: Residual norm
        float res_sq = 0.0f;
        for (int i = 0; i < DimsPerLane; i++)
            res_sq += rot_residual[i] * rot_residual[i];
        float res_norm = sqrt(simd_sum(res_sq));
        if (lane == 0) out_res_norms[bh] = half(res_norm);

        // Step 7: QJL projection (rotated_residual @ combined_proj_t)
        //   combined_proj_t = (mse_rotation @ projection_t).T
        //   so row d gives the projection for output dim d
        for (int i = 0, d = lane; i < DimsPerLane; i++, d += 32)
            if (d < Dim) shared[d] = rot_residual[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup uint shared_signs[Dim];
        for (int i = 0, d = lane; i < DimsPerLane; i++, d += 32) {
            float proj = 0.0f;
            if (d < Dim) {
                auto row = combined_proj_t + d * Dim;
                for (int j = 0; j < (int)Dim; j++)
                    proj += shared[j] * row[j];
            }
            shared_signs[d < Dim ? d : 0] = (proj >= 0.0f) ? 1u : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 8: Pack sign bits (1 bit per dim)
        for (int w = lane; w < SignPackedWidth; w += 32) {
            uint word = 0;
            for (int b = 0; b < 32; b++) {
                int dim = w * 32 + b;
                if (dim < Dim) word |= shared_signs[dim] << b;
            }
            out_signs[bh * SignPackedWidth + w] = word;
        }
}

template [[host_name("custom_kernel_turboquant_fused_prod_quantize_4__64_2_8_2_float16_t_float_float_float_float_float16_t_uint32_t_float16_t_uint32_t")]] [[kernel]] decltype(custom_kernel_turboquant_fused_prod_quantize_4__64_2_8_2_float16_t_float_float_float_float_float16_t_uint32_t_float16_t_uint32_t<64, 2, 8, 2>) custom_kernel_turboquant_fused_prod_quantize_4__64_2_8_2_float16_t_float_float_float_float_float16_t_uint32_t_float16_t_uint32_t<64, 2, 8, 2>;
