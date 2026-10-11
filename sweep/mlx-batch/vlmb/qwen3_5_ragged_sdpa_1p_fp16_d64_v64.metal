// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/qwen3_5/language.py:570.
// Representative call: decode: queries (1,2,1,64) float16, keys/values (1,1,8,64), pads (1) int32, scale (1) float32, k_size (1) int32; T=float16,D_SIZE=64,V_SIZE=64,NUM_Q_HEADS=2,NUM_KV_HEADS=1,GQA_FACTOR=2; grid=(1024,2,1), threadgroup=(1024,1,1); out (1,2,1,64).
#include <metal_simdgroup>
using namespace metal;
template <typename T, int D_SIZE, int V_SIZE, int NUM_Q_HEADS, int NUM_KV_HEADS, int GQA_FACTOR>
[[kernel]] void custom_kernel_qwen3_5_ragged_sdpa_1p_fp16_d64_v64__float16_t_64_64_2_1_2_float16_t_float16_t_float16_t_int32_tc_floatc_int32_tc_float16_t(
  const device float16_t* queries [[buffer(0)]],
  const device float16_t* keys [[buffer(1)]],
  const device float16_t* values [[buffer(2)]],
  const constant int32_t* pads [[buffer(3)]],
  const constant float* scale [[buffer(4)]],
  const constant int32_t* k_size [[buffer(5)]],
  device float16_t* out [[buffer(6)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    uint q_batch_head_idx = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D_SIZE / BD;
    constexpr int v_per_thread = V_SIZE / BD;

    typedef float U;
    thread U q[qk_per_thread];
    thread U k[qk_per_thread];
    thread U o[v_per_thread];
    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];

    int K_SIZE = int(k_size[0]);
    int batch_idx = int(q_batch_head_idx) / NUM_Q_HEADS;
    int q_head_idx = int(q_batch_head_idx) - batch_idx * NUM_Q_HEADS;
    int kv_head_idx = q_head_idx / GQA_FACTOR;
    int pad = int(pads[batch_idx]);
    int N = K_SIZE - pad;

    const device T* qptr =
        queries + int(q_batch_head_idx) * D_SIZE + int(simd_lid) * qk_per_thread;
    const device T* kptr =
        keys + (batch_idx * NUM_KV_HEADS + kv_head_idx) * K_SIZE * D_SIZE +
        (pad + int(simd_gid)) * D_SIZE + int(simd_lid) * qk_per_thread;
    const device T* vptr =
        values + (batch_idx * NUM_KV_HEADS + kv_head_idx) * K_SIZE * V_SIZE +
        (pad + int(simd_gid)) * V_SIZE + int(simd_lid) * v_per_thread;
    device T* optr =
        out + int(q_batch_head_idx) * V_SIZE + int(simd_gid) * v_per_thread;

    U s = U(scale[0]);
    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = s * qptr[i];
    }
    for (int i = 0; i < v_per_thread; i++) {
        o[i] = 0;
    }

    U max_score = -3.4028234663852886e38f;
    U sum_exp_score = 0;

    for (int i = int(simd_gid); i < N; i += BN) {
        for (int j = 0; j < qk_per_thread; j++) {
            k[j] = kptr[j];
        }

        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
            score += q[j] * k[j];
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

        kptr += BN * D_SIZE;
        vptr += BN * V_SIZE;
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

    for (int i = 0; i < v_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < v_per_thread; i++) {
            optr[i] = static_cast<T>(o[i]);
        }
    }

}

template [[host_name("custom_kernel_qwen3_5_ragged_sdpa_1p_fp16_d64_v64__float16_t_64_64_2_1_2_float16_t_float16_t_float16_t_int32_tc_floatc_int32_tc_float16_t")]] [[kernel]] decltype(custom_kernel_qwen3_5_ragged_sdpa_1p_fp16_d64_v64__float16_t_64_64_2_1_2_float16_t_float16_t_float16_t_int32_tc_floatc_int32_tc_float16_t<float16_t, 64, 64, 2, 1, 2>) custom_kernel_qwen3_5_ragged_sdpa_1p_fp16_d64_v64__float16_t_64_64_2_1_2_float16_t_float16_t_float16_t_int32_tc_floatc_int32_tc_float16_t<float16_t, 64, 64, 2, 1, 2>;
