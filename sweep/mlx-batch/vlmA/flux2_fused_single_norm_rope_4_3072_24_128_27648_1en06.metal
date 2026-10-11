// Origin: mlx-vlm@8f5dc3d, mlx-vlm/mlx_vlm/models/bonsai/klein_fast/blocks.py:434.
// Representative call: default spec, seq_len 4, batch 1; fused width qkv_mlp_out_dim = 3*3072 + 2*9216 = 27648; bfloat16 per call site; grid=(3072, 1, 1), threadgroup=(32, 1, 1).
template <typename T>
[[kernel]] void custom_kernel_flux2_fused_single_norm_rope_4_3072_24_128_27648_1en06__bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t(
  const device bfloat16_t* fused [[buffer(0)]],
  const device bfloat16_t* norm_q [[buffer(1)]],
  const device bfloat16_t* norm_k [[buffer(2)]],
  const device bfloat16_t* cos_vals [[buffer(3)]],
  const device bfloat16_t* sin_vals [[buffer(4)]],
  device bfloat16_t* q_out [[buffer(5)]],
  device bfloat16_t* k_out [[buffer(6)]],
  device bfloat16_t* v_out [[buffer(7)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

    uint global_tid = thread_position_in_grid.x;
    uint pos = global_tid / 32;
    uint tid = global_tid % 32;

    const int S_DIM = 4;
    const int H_DIM = 24;
    const int D_DIM = 128;
    const int D_HALF = D_DIM / 2;
    const int FUSED_DIM = 27648;
    const int DIM = 3072;
    const float eps = 1e-06f;
    const int ELEMS = D_DIM / 32;

    uint b = pos / (S_DIM * H_DIM);
    uint s = (pos / H_DIM) % S_DIM;
    uint h = pos % H_DIM;

    uint fused_base = b * S_DIM * FUSED_DIM + s * FUSED_DIM;
    uint q_col = h * D_DIM;
    uint k_col = DIM + h * D_DIM;
    uint v_col = 2 * DIM + h * D_DIM;

    float qv[4], kv[4], vv[4];
    float sq_q = 0.0f, sq_k = 0.0f;

    for (int i = 0; i < ELEMS; i++) {
        uint d = tid * ELEMS + i;
        qv[i] = float(fused[fused_base + q_col + d]);
        kv[i] = float(fused[fused_base + k_col + d]);
        vv[i] = float(fused[fused_base + v_col + d]);
        sq_q += qv[i] * qv[i];
        sq_k += kv[i] * kv[i];
    }

    float total_sq_q = simd_sum(sq_q);
    float total_sq_k = simd_sum(sq_k);
    float inv_rms_q = metal::rsqrt(total_sq_q / float(D_DIM) + eps);
    float inv_rms_k = metal::rsqrt(total_sq_k / float(D_DIM) + eps);

    for (int i = 0; i < ELEMS; i++) {
        uint d = tid * ELEMS + i;
        qv[i] = qv[i] * inv_rms_q * float(norm_q[d]);
        kv[i] = kv[i] * inv_rms_k * float(norm_k[d]);
    }

    uint cos_base = s * D_HALF;
    for (int p = 0; p < ELEMS; p += 2) {
        uint d = tid * ELEMS + p;
        uint d_half = d >> 1;
        float c = float(cos_vals[cos_base + d_half]);
        float sn = float(sin_vals[cos_base + d_half]);
        float q0 = qv[p], q1 = qv[p + 1];
        float k0 = kv[p], k1 = kv[p + 1];
        qv[p]     = q0 * c - q1 * sn;
        qv[p + 1] = q1 * c + q0 * sn;
        kv[p]     = k0 * c - k1 * sn;
        kv[p + 1] = k1 * c + k0 * sn;
    }

    uint out_base = b * H_DIM * S_DIM * D_DIM + h * S_DIM * D_DIM + s * D_DIM;
    for (int i = 0; i < ELEMS; i++) {
        uint d = tid * ELEMS + i;
        q_out[out_base + d] = T(qv[i]);
        k_out[out_base + d] = T(kv[i]);
        v_out[out_base + d] = T(vv[i]);
    }

}

template [[host_name("custom_kernel_flux2_fused_single_norm_rope_4_3072_24_128_27648_1en06__bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t")]] [[kernel]] decltype(custom_kernel_flux2_fused_single_norm_rope_4_3072_24_128_27648_1en06__bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t<bfloat16_t>) custom_kernel_flux2_fused_single_norm_rope_4_3072_24_128_27648_1en06__bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t<bfloat16_t>;
