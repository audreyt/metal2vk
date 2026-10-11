// Origin: mlx-audio@17001a6, mlx_audio/sts/models/mossformer2_se/flash_attention_kernels.py:28.
// Representative call: a, b, scale float32 (16 each), out float32 (16), T=float, grid=(16,1,1), threadgroup=(16,1,1).
template <typename T>
[[kernel]] void custom_kernel_fused_multiply_add__float_float_float_float_float(
  const device float* a [[buffer(0)]],
  const device float* b [[buffer(1)]],
  const device float* scale [[buffer(2)]],
  device float* out [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        uint elem = thread_position_in_grid.x;
        out[elem] = a[elem] * b[elem] * scale[0];
}

template [[host_name("custom_kernel_fused_multiply_add__float_float_float_float_float")]] [[kernel]] decltype(custom_kernel_fused_multiply_add__float_float_float_float_float<float>) custom_kernel_fused_multiply_add__float_float_float_float_float<float>;
