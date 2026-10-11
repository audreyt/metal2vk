// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/rope_utils.py:863.
// Representative call: x (1,2,4,32) float32, cos/sin (1,4,16) float32; T=float32, rotary_dim=32, half_split, cos_ndim=3, not sectioned, half cos layout; grid=(128,1,1), threadgroup=(256,1,1); x_out (1,2,4,32).
template <typename T>
[[kernel]] void custom_kernel_rotary_apply_half_split_half_preselected_32_3d__float_float_float_float_float(
  const device float* x [[buffer(0)]],
  const constant int* x_shape [[buffer(1)]],
  const device float* cos [[buffer(2)]],
  const device float* sin [[buffer(3)]],
  device float* x_out [[buffer(4)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

        uint elem = thread_position_in_grid.x;

        const int q_bsz = x_shape[0];
        const int q_heads = x_shape[1];
        const int q_len = x_shape[2];
        const int q_dim = x_shape[3];
        const int half_dim = 16;
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
        
        
        float c = static_cast<float>(cos[(b * q_len + t) * 32 + freq_idx]);
        float s = static_cast<float>(sin[(b * q_len + t) * 32 + freq_idx]);
        float cp = static_cast<float>(cos[(b * q_len + t) * 32 + pair_d]);
        float sp = static_cast<float>(sin[(b * q_len + t) * 32 + pair_d]);

        int idx = base + d;
        float xv = static_cast<float>(x[idx]);
        float xp = static_cast<float>(x[base + pair_d]);
        x_out[idx] = static_cast<T>(xv * c - xp * s);
        x_out[base + pair_d] = static_cast<T>(xp * cp + xv * sp);
    
}

template [[host_name("custom_kernel_rotary_apply_half_split_half_preselected_32_3d__float_float_float_float_float")]] [[kernel]] decltype(custom_kernel_rotary_apply_half_split_half_preselected_32_3d__float_float_float_float_float<float>) custom_kernel_rotary_apply_half_split_half_preselected_32_3d__float_float_float_float_float<float>;
