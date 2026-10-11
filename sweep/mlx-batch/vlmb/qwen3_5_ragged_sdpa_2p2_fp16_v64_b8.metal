// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/qwen3_5/language.py:596.
// Representative call: decode two-pass 2: partials (1,2,1,8,64) float16, sums/maxs (1,2,1,8) float32; T=float16,D_SIZE=64,BLOCKS=8; grid=(1024,2,1), threadgroup=(1024,1,1); out (1,2,1,64).
#include <metal_simdgroup>
using namespace metal;
template <typename T, int D_SIZE, int BLOCKS>
[[kernel]] void custom_kernel_qwen3_5_ragged_sdpa_2p2_fp16_v64_b8__float16_t_64_8_float16_t_float_float_float16_t(
  const device float16_t* partials [[buffer(0)]],
  const device float* sums [[buffer(1)]],
  const device float* maxs [[buffer(2)]],
  device float16_t* out [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    uint q_batch_head_idx = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int elem_per_thread = D_SIZE / BD;

    typedef float U;
    thread U o[elem_per_thread] = {0};
    threadgroup U outputs[BN * BD];

    const device T* part =
        partials + int(q_batch_head_idx) * BLOCKS * D_SIZE +
        int(simd_gid) * D_SIZE + int(simd_lid) * elem_per_thread;
    const device float* sump = sums + int(q_batch_head_idx) * BLOCKS;
    const device float* maxp = maxs + int(q_batch_head_idx) * BLOCKS;
    device T* optr =
        out + int(q_batch_head_idx) * D_SIZE + int(simd_gid) * elem_per_thread;

    U sum_exp_score = 0.0;
    U max_score = -3.4028234663852886e38f;

    for (int b = 0; b < BLOCKS / BN; ++b) {
        max_score = max(max_score, maxp[int(simd_lid) + BN * b]);
    }
    max_score = simd_max(max_score);

    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(maxp[int(simd_lid) + BN * b] - max_score);
        sum_exp_score += factor * sump[int(simd_lid) + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);

    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(maxp[int(simd_gid)] - max_score);
        for (int i = 0; i < elem_per_thread; i++) {
            o[i] += factor * static_cast<U>(part[i]);
        }
        maxp += BN;
        sump += BN;
        part += BN * D_SIZE;
    }

    for (int i = 0; i < elem_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < elem_per_thread; i++) {
            optr[i] = static_cast<T>(o[i]);
        }
    }

}

template [[host_name("custom_kernel_qwen3_5_ragged_sdpa_2p2_fp16_v64_b8__float16_t_64_8_float16_t_float_float_float16_t")]] [[kernel]] decltype(custom_kernel_qwen3_5_ragged_sdpa_2p2_fp16_v64_b8__float16_t_64_8_float16_t_float_float_float16_t<float16_t, 64, 8>) custom_kernel_qwen3_5_ragged_sdpa_2p2_fp16_v64_b8__float16_t_64_8_float16_t_float_float_float16_t<float16_t, 64, 8>;
