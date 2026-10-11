// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/rope_utils.py:647.
// Representative call: x (1,2,4,32) float32, position_ids (1,4) int32, inv_freq (16) float32, position_selector (16) int32; T=float32, rotary_dim=32, position_ndim=2, half_split; grid=(128,1,1), threadgroup=(256,1,1); x_out (1,2,4,32).
template <typename T>
[[kernel]] void custom_kernel_mrope_apply_half_split_32_2d__float_float_int32_tc_float_int32_t_float(
  const device float* x [[buffer(0)]],
  const constant int* x_shape [[buffer(1)]],
  const constant int32_t* position_ids [[buffer(2)]],
  const device float* inv_freq [[buffer(3)]],
  const device int32_t* position_selector [[buffer(4)]],
  device float* x_out [[buffer(5)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

        uint elem = thread_position_in_grid.x;

        const int half_dim = 16;
        const int q_bsz = x_shape[0];
        const int q_heads = x_shape[1];
        const int q_len = x_shape[2];
        const int q_dim = x_shape[3];
        const int slots = half_dim + q_dim - 32;
        const int work_size = q_bsz * q_heads * q_len * slots;

        if (elem >= uint(work_size)) {
            return;
        }

        int local = int(elem);
        int slot = local % slots;
        int tmp = local / slots;
        int t = tmp % q_len;
        tmp = tmp / q_len;
        int h = tmp % q_heads;
        int b = tmp / q_heads;
        int base = ((b * q_heads + h) * q_len + t) * q_dim;

        if (slot >= half_dim) {
            int pass_d = 32 + slot - half_dim;
            int pass_idx = base + pass_d;
            x_out[pass_idx] = x[pass_idx];
            return;
        }

        
        int freq_idx = slot;
        int d = freq_idx;
        int pair_d = d + half_dim;
        
        float pos = static_cast<float>(position_ids[b * q_len + t]);
        float angle = pos * static_cast<float>(inv_freq[freq_idx]);
        float c = metal::cos(angle);
        float s = metal::sin(angle);
        

        int idx = base + d;
        float xv = static_cast<float>(x[idx]);
        float xp = static_cast<float>(x[base + pair_d]);
        x_out[idx] = static_cast<T>(xv * c - xp * s);
        x_out[base + pair_d] = static_cast<T>(xp * c + xv * s);
    
}

template [[host_name("custom_kernel_mrope_apply_half_split_32_2d__float_float_int32_tc_float_int32_t_float")]] [[kernel]] decltype(custom_kernel_mrope_apply_half_split_32_2d__float_float_int32_tc_float_int32_t_float<float>) custom_kernel_mrope_apply_half_split_32_2d__float_float_int32_tc_float_int32_t_float<float>;
