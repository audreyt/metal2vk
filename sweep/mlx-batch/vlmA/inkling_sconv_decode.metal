// Origin: mlx-vlm@8f5dc3d, mlx-vlm/mlx_vlm/models/inkling/language.py:290.
// Representative call: float32 x (1,4,8), K=4 causal conv over the (1,3,8) state, HAS_RES=false (residual passed as x); float32 keeps the documented fp32-stability path; grid=(32, 1, 1), threadgroup=(32, 1, 1).
template <typename T, int B, int L, int C, int K, bool HAS_RES>
[[kernel]] void custom_kernel_inkling_sconv_decode__float_1_4_8_4_false_float_float_float_float_float_float(
  const device float* x [[buffer(0)]],
  const device float* state [[buffer(1)]],
  const device float* w [[buffer(2)]],
  const device float* res [[buffer(3)]],
  device float* out [[buffer(4)]],
  device float* nstate [[buffer(5)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

    uint c = thread_position_in_grid.x;   // channel
    uint b = thread_position_in_grid.y;   // batch row
    if (c >= C || b >= B) return;
    float w0 = (float)w[c * K + 0];
    float w1 = (float)w[c * K + 1];
    float w2 = (float)w[c * K + 2];
    float w3 = (float)w[c * K + 3];
    // Virtual padded input xp = [state (K-1 rows, fp32); x (L rows, T)].
    for (uint i = 0; i < L; ++i) {
        float acc = 0.0f;
        for (uint k = 0; k < K; ++k) {
            int r = (int)(i + k) - (int)(K - 1);  // row into x; negative -> state
            float v = (r < 0)
                ? state[(b * (K - 1) + (uint)(r + (int)(K - 1))) * C + c]
                : (float)x[(b * L + (uint)r) * C + c];
            float wk = (k == 0) ? w0 : (k == 1) ? w1 : (k == 2) ? w2 : w3;
            acc += wk * v;
        }
        // Match the unfused path's rounding: the conv emits bf16 (rounded)
        // before the fp32 residual add; the layer residual is a second bf16
        // add on top (as the decoder layer's x + sconv(r) was).
        float conv_r = (float)((T)acc);
        T inner = (T)(conv_r + (float)x[(b * L + i) * C + c]);
        if (HAS_RES) {
            out[(b * L + i) * C + c] =
                (T)((float)inner + (float)res[(b * L + i) * C + c]);
        } else {
            out[(b * L + i) * C + c] = inner;
        }
    }
    for (uint s = 0; s < K - 1; ++s) {
        int r = (int)(L + s) - (int)(K - 1);
        nstate[(b * (K - 1) + s) * C + c] = (r < 0)
            ? state[(b * (K - 1) + (uint)(r + (int)(K - 1))) * C + c]
            : (float)x[(b * L + (uint)r) * C + c];
    }

}

template [[host_name("custom_kernel_inkling_sconv_decode__float_1_4_8_4_false_float_float_float_float_float_float")]] [[kernel]] decltype(custom_kernel_inkling_sconv_decode__float_1_4_8_4_false_float_float_float_float_float_float<float, 1, 4, 8, 4, false>) custom_kernel_inkling_sconv_decode__float_1_4_8_4_false_float_float_float_float_float_float<float, 1, 4, 8, 4, false>;
