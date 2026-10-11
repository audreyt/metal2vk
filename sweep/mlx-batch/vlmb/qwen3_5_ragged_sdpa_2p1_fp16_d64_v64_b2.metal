// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/qwen3_5/language.py:582.
// Representative call: decode two-pass 1: queries (1,2,1,64) float16, keys/values (1,1,8,64), pads (1) int32, scale (1) float32, k_size (1) int32; T=float16,D_SIZE=64,V_SIZE=64,NUM_Q_HEADS=2,NUM_KV_HEADS=1,GQA_FACTOR=2,BLOCKS=2; grid=(32,2,2), threadgroup=(32,2,1); partials (1,2,1,2,64) float16, sums/maxs (1,2,1,2) float32.
#include <metal_simdgroup>
using namespace metal;
template <typename T, int D_SIZE, int V_SIZE, int NUM_Q_HEADS, int NUM_KV_HEADS, int GQA_FACTOR, int BLOCKS>
[[kernel]] void custom_kernel_qwen3_5_ragged_sdpa_2p1_fp16_d64_v64_b2__float16_t_64_64_2_1_2_2_float16_t_float16_t_float16_t_int32_tc_floatc_int32_tc_float16_t_float_float(
  const device float16_t* queries [[buffer(0)]],
  const device float16_t* keys [[buffer(1)]],
  const device float16_t* values [[buffer(2)]],
  const constant int32_t* pads [[buffer(3)]],
  const constant float* scale [[buffer(4)]],
  const constant int32_t* k_size [[buffer(5)]],
  device float16_t* partials [[buffer(6)]],
  device float* sums [[buffer(7)]],
  device float* maxs [[buffer(8)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    uint simd_lid = thread_index_in_simdgroup;
    uint kv_head_idx = threadgroup_position_in_grid.x;
    uint batch_idx = threadgroup_position_in_grid.y;
    uint block_idx = threadgroup_position_in_grid.z;
    uint gqa_idx = thread_position_in_threadgroup.y;

    constexpr int BD = 32;
    constexpr int qk_per_thread = D_SIZE / BD;
    constexpr int v_per_thread = V_SIZE / BD;

    typedef float U;
    thread U q[qk_per_thread];
    thread U o[v_per_thread] = {0};

    int K_SIZE = int(k_size[0]);
    int q_head_idx = int(GQA_FACTOR * kv_head_idx + gqa_idx);
    int q_batch_head_idx = int(batch_idx) * NUM_Q_HEADS + q_head_idx;
    int pad = int(pads[batch_idx]);
    int N = K_SIZE - pad;

    const device T* qptr =
        queries + q_batch_head_idx * D_SIZE + int(simd_lid) * qk_per_thread;
    const device T* kptr =
        keys + (int(batch_idx) * NUM_KV_HEADS + int(kv_head_idx)) *
                   K_SIZE * D_SIZE +
        (pad + int(block_idx)) * D_SIZE + int(simd_lid) * qk_per_thread;
    const device T* vptr =
        values + (int(batch_idx) * NUM_KV_HEADS + int(kv_head_idx)) *
                     K_SIZE * V_SIZE +
        (pad + int(block_idx)) * V_SIZE + int(simd_lid) * v_per_thread;
    device T* optr =
        partials + q_batch_head_idx * BLOCKS * V_SIZE +
        int(block_idx) * V_SIZE + int(simd_lid) * v_per_thread;
    device float* sump =
        sums + q_batch_head_idx * BLOCKS + int(block_idx);
    device float* maxp =
        maxs + q_batch_head_idx * BLOCKS + int(block_idx);

    U s = U(scale[0]);
    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = s * qptr[i];
    }

    U max_score = -3.4028234663852886e38f;
    U sum_exp_score = 0;

    for (int i = int(block_idx); i < N; i += BLOCKS) {
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
            score += q[j] * kptr[j];
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (int j = 0; j < v_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * vptr[j];
        }

        kptr += BLOCKS * D_SIZE;
        vptr += BLOCKS * V_SIZE;
    }

    if (simd_lid == 0) {
        sump[0] = sum_exp_score;
        maxp[0] = max_score;
    }
    for (int i = 0; i < v_per_thread; i++) {
        optr[i] = static_cast<T>(o[i]);
    }

}

template [[host_name("custom_kernel_qwen3_5_ragged_sdpa_2p1_fp16_d64_v64_b2__float16_t_64_64_2_1_2_2_float16_t_float16_t_float16_t_int32_tc_floatc_int32_tc_float16_t_float_float")]] [[kernel]] decltype(custom_kernel_qwen3_5_ragged_sdpa_2p1_fp16_d64_v64_b2__float16_t_64_64_2_1_2_2_float16_t_float16_t_float16_t_int32_tc_floatc_int32_tc_float16_t_float_float<float16_t, 64, 64, 2, 1, 2, 2>) custom_kernel_qwen3_5_ragged_sdpa_2p1_fp16_d64_v64_b2__float16_t_64_64_2_1_2_2_float16_t_float16_t_float16_t_int32_tc_floatc_int32_tc_float16_t_float_float<float16_t, 64, 64, 2, 1, 2, 2>;
