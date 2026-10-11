// Origin: mlx-lm@2184db2, mlx_lm/models/bitlinear_layers.py:80.
// Representative call: float32 x (1,32), uint8 packed_weights (4,32), float32 weight_scale (1).
// T=float, invert_weight_scales=false, in_features=32, out_features=16; grid=(32,4,1), threadgroup=(32,1,1).
template <typename T, bool invert_weight_scales, int in_features, int out_features>
[[kernel]] void custom_kernel_bitlinear_matmul__float_false_32_16_float_uint8_t_floatc_float(
  const device float* x [[buffer(0)]],
  const device uint8_t* packed_weights [[buffer(1)]],
  const constant float* weight_scale [[buffer(2)]],
  device float* out [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    constexpr int M = 4;
    constexpr int BLOCK = 32;

    uint tid = thread_position_in_grid.y;
    uint in_offset = thread_position_in_grid.x;

    uint batch_idx = tid / (out_features / 4);
    uint row_idx = tid % (out_features / 4);

    float sum[4] = {0.0};

    for (uint i = in_offset * M; i < in_features; i += BLOCK * M) {
        float v[M];
        for (int j=0; j<M; j++) {
            v[j] = x[batch_idx * in_features + i + j];
        }

        for (int j=0; j<M; j++) {
            uint8_t w = packed_weights[row_idx * in_features + i + j];
            sum[0] += v[j] * ((w & 3) - 1);
            sum[1] += v[j] * (((w >> 2) & 3) - 1);
            sum[2] += v[j] * (((w >> 4) & 3) - 1);
            sum[3] += v[j] * (((w >> 6) & 3) - 1);
        }
    }

    for (int j=0; j<4; j++) {
        sum[j] = simd_sum(sum[j]);
    }

    if (in_offset == 0) {
        float scale = invert_weight_scales ? 1 / weight_scale[0] : weight_scale[0];
        for (int i=0; i<4; i++) {
            out[batch_idx * out_features + row_idx + i * (out_features/4)] = static_cast<T>(sum[i] * scale);
        }
    }
}

template [[host_name("custom_kernel_bitlinear_matmul__float_false_32_16_float_uint8_t_floatc_float")]] [[kernel]] decltype(custom_kernel_bitlinear_matmul__float_false_32_16_float_uint8_t_floatc_float<float, false, 32, 16>) custom_kernel_bitlinear_matmul__float_false_32_16_float_uint8_t_floatc_float<float, false, 32, 16>;
