// Origin: mlx-vlm@8f5dc3d, mlx-vlm/mlx_vlm/models/inkling/language.py:593.
// Representative call: float32 logits (2, 8 routed + 2 shared), top_k 2, intermediate 4; one simdgroup per token; grid=(32, 2, 1), threadgroup=(32, 1, 1).
template <typename T, int N, int R, int SH, int K, int I>
[[kernel]] void custom_kernel_inkling_moe_route__float_2_8_2_2_4_float_float_floatc_uint32_t_float_float(
  const device float* logits [[buffer(0)]],
  const device float* corr [[buffer(1)]],
  const constant float* wscale [[buffer(2)]],
  device uint32_t* idx [[buffer(3)]],
  device float* wk [[buffer(4)]],
  device float* gamma [[buffer(5)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

    // One simdgroup (32 lanes) per token; each lane owns ceil(R/32) experts
    // (strided by 32 so logit reads coalesce). Top-K via K rounds of
    // simd_max; ties resolved to the lowest lane (deterministic).
    uint lane = thread_position_in_grid.x;   // 0..31, simd lane
    uint n = thread_position_in_grid.y;      // token
    if (n >= N) return;
    const device T* lg = logits + (size_t)n * (R + SH);
    float ws = wscale[0];
    constexpr int PER = (R + 31) / 32;
    float sc[PER];
    float tl_[PER];
    bool taken[PER];
    for (int t = 0; t < PER; ++t) {
        uint j = lane + (uint)t * 32u;
        taken[t] = false;
        if (j < R) {
            float l = (float)lg[j];
            tl_[t] = l;
            sc[t] = 1.0f / (1.0f + metal::exp(-l)) + (float)corr[j];
        } else {
            sc[t] = -INFINITY;
        }
    }
    uint bidx[K];
    float btl[K];
    for (uint kk = 0; kk < K; ++kk) {
        float lb = -INFINITY; int lt = -1;
        for (int t = 0; t < PER; ++t)
            if (!taken[t] && sc[t] > lb) { lb = sc[t]; lt = t; }
        float gb = simd_max(lb);
        ushort wl = (ushort)simd_min(lb == gb ? lane : 32u);
        uint wj = simd_shuffle(lt >= 0 ? lane + (uint)lt * 32u : 0u, wl);
        float wtl = simd_shuffle(lt >= 0 ? tl_[lt] : 0.0f, wl);
        if (lane == (uint)wl && lt >= 0 && sc[lt] == gb) taken[lt] = true;
        bidx[kk] = wj; btl[kk] = wtl;
    }
    // Routing weights: softmax over logsigmoid of the K routed + SH shared
    // logits, times route_scale * global_scale (folded into ws). Computed
    // redundantly on every lane (cheap; K + SH values).
    float lp[K + SH];
    float m = -INFINITY;
    for (uint t = 0; t < K + SH; ++t) {
        float tv = (t < K) ? btl[t] : (float)lg[R + (t - K)];
        float a = -tv;
        float lad = metal::max(a, 0.0f)
                  + metal::log(1.0f + metal::exp(-metal::fabs(a)));
        lp[t] = -lad;
        m = metal::max(m, lp[t]);
    }
    float se = 0.0f;
    for (uint t = 0; t < K + SH; ++t) se += metal::exp(lp[t] - m);
    float lse = m + metal::log(se);
    if (lane < K) {
        idx[(size_t)n * K + lane] = bidx[lane];
        wk[(size_t)n * K + lane] = (T)(metal::exp(lp[lane] - lse) * ws);
    }
    T gv[SH];
    for (uint s = 0; s < SH; ++s) gv[s] = (T)(metal::exp(lp[K + s] - lse) * ws);
    device T* gp = gamma + (size_t)n * SH * I;
    for (uint t = lane; t < SH * I; t += 32u) gp[t] = gv[t / I];

}

template [[host_name("custom_kernel_inkling_moe_route__float_2_8_2_2_4_float_float_floatc_uint32_t_float_float")]] [[kernel]] decltype(custom_kernel_inkling_moe_route__float_2_8_2_2_4_float_float_floatc_uint32_t_float_float<float, 2, 8, 2, 2, 4>) custom_kernel_inkling_moe_route__float_2_8_2_2_4_float_float_floatc_uint32_t_float_float<float, 2, 8, 2, 2, 4>;
