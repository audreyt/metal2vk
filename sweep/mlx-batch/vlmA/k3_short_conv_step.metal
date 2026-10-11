// Origin: mlx-vlm@8f5dc3d, mlx-vlm/mlx_vlm/models/kimi_k3/language.py:81.
// Representative call: float32 x (1,1,8), kernel_size 4 short conv step over the (1,3,8) state; grid=(8, 1, 1), threadgroup=(8, 1, 1).
template <typename T, int C, int KS>
[[kernel]] void custom_kernel_k3_short_conv_step__float_8_4_float_float_float_float_float(
  const device float* x [[buffer(0)]],
  const device float* state [[buffer(1)]],
  const device float* w [[buffer(2)]],
  device float* y [[buffer(3)]],
  device float* new_state [[buffer(4)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

    auto c = thread_position_in_grid.x;
    auto b = thread_position_in_grid.y;

    const device T* s = state + b * (KS - 1) * C;
    float v = 0.0f;
    for (int j = 0; j < KS - 1; ++j) {
      v += static_cast<float>(w[c * KS + j]) * static_cast<float>(s[j * C + c]);
    }
    v += static_cast<float>(w[c * KS + KS - 1]) * static_cast<float>(x[b * C + c]);
    y[b * C + c] = static_cast<T>(v / (1.0f + metal::exp(-v)));

    device T* ns = new_state + b * (KS - 1) * C;
    for (int j = 0; j < KS - 2; ++j) {
      ns[j * C + c] = s[(j + 1) * C + c];
    }
    ns[(KS - 2) * C + c] = x[b * C + c];

}

template [[host_name("custom_kernel_k3_short_conv_step__float_8_4_float_float_float_float_float")]] [[kernel]] decltype(custom_kernel_k3_short_conv_step__float_8_4_float_float_float_float_float<float, 8, 4>) custom_kernel_k3_short_conv_step__float_8_4_float_float_float_float_float<float, 8, 4>;
