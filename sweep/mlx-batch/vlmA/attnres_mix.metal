// Origin: mlx-vlm@8f5dc3d, mlx-vlm/mlx_vlm/models/kimi_k3/language.py:295.
// Representative call: float32; K=2 prior residual rows plus the partial row, D 8, THREADS 512 as in the caller; eps and N arrive as small buffers (N a scalar); grid=(512, 1, 1), threadgroup=(512, 1, 1).
template <typename InT, int K, int D, int THREADS>
[[kernel]] void custom_kernel_attnres_mix__float_2_8_512_float_floatc_float_float_floatc_int32_ts_float(
  const device float* raw [[buffer(0)]],
  const constant float* inv_rms [[buffer(1)]],
  const device float* partial [[buffer(2)]],
  const device float* w_eff [[buffer(3)]],
  const constant float* eps [[buffer(4)]],
  const constant int32_t& N [[buffer(5)]],
  device float* out [[buffer(6)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    constexpr int NACC = K + 2;
    constexpr int NSIMD = THREADS / 32;

    auto n = threadgroup_position_in_grid.y;
    auto tid = thread_position_in_threadgroup.x;
    auto lane = thread_index_in_simdgroup;
    auto sg = simdgroup_index_in_threadgroup;

    auto partial_ = partial + n * D;

    float acc[NACC];
    for (int i = 0; i < NACC; ++i) {
      acc[i] = 0.0f;
    }
    for (uint d = tid; d < D; d += THREADS) {
      float w = static_cast<float>(w_eff[d]);
      float pv = static_cast<float>(partial_[d]);
      for (int k = 0; k < K; ++k) {
        acc[k] += static_cast<float>(raw[(k * N + n) * D + d]) * w;
      }
      acc[K] += pv * w;
      acc[K + 1] += pv * pv;
    }

    threadgroup float shm[NACC * NSIMD];
    for (int i = 0; i < NACC; ++i) {
      float s = simd_sum(acc[i]);
      if (lane == 0) {
        shm[i * NSIMD + sg] = s;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float weights[K + 1];
    if (tid == 0) {
      float tot[NACC];
      for (int i = 0; i < NACC; ++i) {
        tot[i] = 0.0f;
        for (int j = 0; j < NSIMD; ++j) {
          tot[i] += shm[i * NSIMD + j];
        }
      }
      float rinv = metal::rsqrt(tot[K + 1] / D + eps[0]);
      float logits[K + 1];
      float m = -1e30f;
      for (int k = 0; k < K; ++k) {
        logits[k] = tot[k] * inv_rms[k * N + n];
        m = metal::max(m, logits[k]);
      }
      logits[K] = tot[K] * rinv;
      m = metal::max(m, logits[K]);
      float denom = 0.0f;
      for (int k = 0; k <= K; ++k) {
        logits[k] = metal::exp(logits[k] - m);
        denom += logits[k];
      }
      for (int k = 0; k <= K; ++k) {
        weights[k] = logits[k] / denom;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float wp = weights[K];
    auto out_ = out + n * D;
    for (uint d = tid; d < D; d += THREADS) {
      float o = wp * static_cast<float>(partial_[d]);
      for (int k = 0; k < K; ++k) {
        o += weights[k] * static_cast<float>(raw[(k * N + n) * D + d]);
      }
      out_[d] = static_cast<InT>(o);
    }

}

template [[host_name("custom_kernel_attnres_mix__float_2_8_512_float_floatc_float_float_floatc_int32_ts_float")]] [[kernel]] decltype(custom_kernel_attnres_mix__float_2_8_512_float_floatc_float_float_floatc_int32_ts_float<float, 2, 8, 512>) custom_kernel_attnres_mix__float_2_8_512_float_floatc_float_float_floatc_int32_ts_float<float, 2, 8, 512>;
