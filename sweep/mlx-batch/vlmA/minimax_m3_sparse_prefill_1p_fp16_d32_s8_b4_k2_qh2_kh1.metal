// Origin: mlx-vlm@8f5dc3d, mlx-vlm/mlx_vlm/models/minimax_m3_vl/language.py:396.
// Representative call: fp16 (the caller gates dtype to bfloat16/float16), D_SIZE 32, 2 top-k blocks of 4, selected 8, key_length 56 = 7*selected per the caller guard; grid=(1024, 2, 1), threadgroup=(1024, 1, 1).
#include <metal_simdgroup>
using namespace metal;
template <typename T, int D_SIZE, int Q_LEN, int NUM_Q_HEADS, int NUM_KV_HEADS, int GQA_FACTOR, int BLOCK_SIZE, int TOPK_BLOCKS, int SELECTED_LENGTH>
[[kernel]] void custom_kernel_minimax_m3_sparse_prefill_1p_fp16_d32_s8_b4_k2_qh2_kh1__float16_t_32_1_2_1_2_4_2_8_float16_t_float16_t_float16_t_int32_tc_int32_tc_floatc_int32_tc_float16_t(
  const device float16_t* queries [[buffer(0)]],
  const device float16_t* keys [[buffer(1)]],
  const device float16_t* values [[buffer(2)]],
  const constant int32_t* block_indices [[buffer(3)]],
  const constant int32_t* q_positions [[buffer(4)]],
  const constant float* scale [[buffer(5)]],
  const constant int32_t* k_size [[buffer(6)]],
  device float16_t* out [[buffer(7)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    uint row_idx = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D_SIZE / BD;
    constexpr int v_per_thread = D_SIZE / BD;

    typedef float U;
    thread U q[qk_per_thread];
    thread U o[v_per_thread];
    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];

    int key_length = int(k_size[0]);
    int query_idx = int(row_idx % Q_LEN);
    int batch_head_idx = int(row_idx / Q_LEN);
    int batch_idx = batch_head_idx / NUM_Q_HEADS;
    int q_head_idx = batch_head_idx - batch_idx * NUM_Q_HEADS;
    int kv_head_idx = q_head_idx / GQA_FACTOR;

    const device T* qptr =
        queries + ((batch_head_idx * Q_LEN + query_idx) * D_SIZE) +
        int(simd_lid) * qk_per_thread;
    device T* optr =
        out + ((batch_head_idx * Q_LEN + query_idx) * D_SIZE) +
        int(simd_gid) * v_per_thread;

    U s = U(scale[0]);
    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = s * static_cast<U>(qptr[i]);
    }
    for (int i = 0; i < v_per_thread; i++) {
        o[i] = 0;
    }

    int qpos = int(q_positions[batch_idx * Q_LEN + query_idx]);
    int blocks_offset = (batch_idx * Q_LEN + query_idx) * TOPK_BLOCKS;

    U max_score = -3.4028234663852886e38f;
    U sum_exp_score = 0;

    for (int selected_idx = int(simd_gid); selected_idx < SELECTED_LENGTH;
         selected_idx += BN) {
        int block_slot = selected_idx / BLOCK_SIZE;
        int block_offset = selected_idx - block_slot * BLOCK_SIZE;
        int block_idx = int(block_indices[blocks_offset + block_slot]);
        int key_pos = block_idx * BLOCK_SIZE + block_offset;
        bool valid = block_idx >= 0 && key_pos < key_length && key_pos <= qpos;

        U score = -3.4028234663852886e38f;
        if (valid) {
            const device T* kptr =
                keys + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                         key_pos) *
                        D_SIZE) +
                int(simd_lid) * qk_per_thread;
            score = 0;
            for (int j = 0; j < qk_per_thread; j++) {
                score += q[j] * static_cast<U>(kptr[j]);
            }
            score = simd_sum(score);
        }

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = valid ? fast::exp(score - new_max) : U(0);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        if (valid) {
            const device T* vptr =
                values + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                           key_pos) *
                          D_SIZE) +
                int(simd_lid) * v_per_thread;
            for (int j = 0; j < v_per_thread; j++) {
                o[j] = o[j] * factor + exp_score * static_cast<U>(vptr[j]);
            }
        } else {
            for (int j = 0; j < v_per_thread; j++) {
                o[j] = o[j] * factor;
            }
        }
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    U total_sum = simd_sum(sum_exp_scores[simd_lid] * factor);

    for (int i = 0; i < v_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = total_sum == 0 ? U(0) : (o[i] / total_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < v_per_thread; i++) {
            optr[i] = static_cast<T>(o[i]);
        }
    }

}

template [[host_name("custom_kernel_minimax_m3_sparse_prefill_1p_fp16_d32_s8_b4_k2_qh2_kh1__float16_t_32_1_2_1_2_4_2_8_float16_t_float16_t_float16_t_int32_tc_int32_tc_floatc_int32_tc_float16_t")]] [[kernel]] decltype(custom_kernel_minimax_m3_sparse_prefill_1p_fp16_d32_s8_b4_k2_qh2_kh1__float16_t_32_1_2_1_2_4_2_8_float16_t_float16_t_float16_t_int32_tc_int32_tc_floatc_int32_tc_float16_t<float16_t, 32, 1, 2, 1, 2, 4, 2, 8>) custom_kernel_minimax_m3_sparse_prefill_1p_fp16_d32_s8_b4_k2_qh2_kh1__float16_t_32_1_2_1_2_4_2_8_float16_t_float16_t_float16_t_int32_tc_int32_tc_floatc_int32_tc_float16_t<float16_t, 32, 1, 2, 1, 2, 4, 2, 8>;
