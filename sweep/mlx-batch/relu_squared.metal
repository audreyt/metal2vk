// Origin: mlx-audio@17001a6, mlx_audio/sts/models/mossformer2_se/flash_attention_kernels.py:15.
// Representative call: inp float32 (16), out float32 (16), T=float, grid=(16,1,1), threadgroup=(16,1,1).
template <typename T>
[[kernel]] void custom_kernel_relu_squared__float_float_float(
  const device float* inp [[buffer(0)]],
  device float* out [[buffer(1)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        uint elem = thread_position_in_grid.x;
        T val = inp[elem];
        T relu_val = val > T(0) ? val : T(0);
        out[elem] = relu_val * relu_val;
}

template [[host_name("custom_kernel_relu_squared__float_float_float")]] [[kernel]] decltype(custom_kernel_relu_squared__float_float_float<float>) custom_kernel_relu_squared__float_float_float<float>;
