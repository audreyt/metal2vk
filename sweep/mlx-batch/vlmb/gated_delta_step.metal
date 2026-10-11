// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/qwen3_5/gated_delta.py:109.
// Representative call: q/k/v float32 (1,2,1,32), g (1,2,1) float32, beta (1,2,1) float32, state (1,1,32,32) float32, T scalar int32; grid=(32,32,1), threadgroup=(32,4,1); y (1,2,1,32), state_out (1,1,32,32).
template <typename InT, typename StT, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void custom_kernel_gated_delta_step__float_float_32_32_1_1_float_float_float_floatc_floatc_float_int32_ts_float_float(
  const device float* q [[buffer(0)]],
  const device float* k [[buffer(1)]],
  const device float* v [[buffer(2)]],
  const constant float* g [[buffer(3)]],
  const constant float* beta [[buffer(4)]],
  const device float* state_in [[buffer(5)]],
  const constant int32_t& T [[buffer(6)]],
  device float* y [[buffer(7)]],
  device float* state_out [[buffer(8)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_grid [[thread_position_in_grid]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]) {

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          if (true) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          } else {
            y[dv_idx] = static_cast<InT>(0);
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
    
}

template [[host_name("custom_kernel_gated_delta_step__float_float_32_32_1_1_float_float_float_floatc_floatc_float_int32_ts_float_float")]] [[kernel]] decltype(custom_kernel_gated_delta_step__float_float_32_32_1_1_float_float_float_floatc_floatc_float_int32_ts_float_float<float, float, 32, 32, 1, 1>) custom_kernel_gated_delta_step__float_float_32_32_1_1_float_float_float_floatc_floatc_float_int32_ts_float_float<float, float, 32, 32, 1, 1>;
