// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/qwen4_exp/qsa_kernel.py:180.
// Representative call: queries/keys/values (1,2,4,64)/(1,1,8,64)/(1,1,8,64) float16, block_indices (1,4,1) int32, query_ends (1,4) int32, scale (1) float32, k_size (1) int32; T=float16,D_SIZE=64,Q_LEN=4,NUM_Q_HEADS=2,NUM_KV_HEADS=1,GQA_FACTOR=2,BLOCK_SIZE=8,TOPK_BLOCKS=1,SELECTED_LENGTH=8; grid=(1024,8,1), threadgroup=(1024,1,1); out (1,2,4,64).
#include <metal_simdgroup>
using namespace metal;
template <typename T, int D_SIZE, int Q_LEN, int NUM_Q_HEADS, int NUM_KV_HEADS, int GQA_FACTOR, int BLOCK_SIZE, int TOPK_BLOCKS, int SELECTED_LENGTH>
[[kernel]] void custom_kernel_qwen4_exp_qsa_sparse_attention_fp16_d64_s8_b8_k1_qh2_kh1__float16_t_64_4_2_1_2_8_1_8_float16_t_float16_t_float16_t_int32_tc_int32_tc_floatc_int32_tc_float16_t(
  const device float16_t* queries [[buffer(0)]],
  const device float16_t* keys [[buffer(1)]],
  const device float16_t* values [[buffer(2)]],
  const constant int32_t* block_indices [[buffer(3)]],
  const constant int32_t* query_ends [[buffer(4)]],
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
    int query_end = int(query_ends[batch_idx * Q_LEN + query_idx]);

    const device T* qptr =
        queries + ((batch_head_idx * Q_LEN + query_idx) * D_SIZE) +
        int(simd_lid) * qk_per_thread;
    device T* optr =
        out + ((batch_head_idx * Q_LEN + query_idx) * D_SIZE) +
        int(simd_gid) * v_per_thread;

    U s = U(scale[0]);
    for (int i = 0; i < qk_per_thread; ++i) {
        q[i] = s * static_cast<U>(qptr[i]);
    }
    for (int i = 0; i < v_per_thread; ++i) {
        o[i] = 0;
    }

    int blocks_offset = (batch_idx * Q_LEN + query_idx) * TOPK_BLOCKS;
    U max_score = -3.4028234663852886e38f;
    U sum_exp_score = 0;

    // The selected blocks are sorted by token position before launch. Each
    // SIMD group owns an interleaved subset and the final reduction combines
    // their independent online-softmax states.
    for (int selected_idx = int(simd_gid); selected_idx < SELECTED_LENGTH;
         selected_idx += BN) {
        int block_slot = selected_idx / BLOCK_SIZE;
        int block_offset = selected_idx - block_slot * BLOCK_SIZE;
        int block_idx = int(block_indices[blocks_offset + block_slot]);
        int key_pos = block_idx * BLOCK_SIZE + block_offset;
        bool valid = block_idx >= 0 && key_pos < key_length && key_pos < query_end;

        U score = -3.4028234663852886e38f;
        if (valid) {
            const device T* kptr =
                keys + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                         key_pos) * D_SIZE) +
                int(simd_lid) * qk_per_thread;
            score = 0;
            for (int j = 0; j < qk_per_thread; ++j) {
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
                           key_pos) * D_SIZE) +
                int(simd_lid) * v_per_thread;
            for (int j = 0; j < v_per_thread; ++j) {
                o[j] = o[j] * factor + exp_score * static_cast<U>(vptr[j]);
            }
        } else {
            for (int j = 0; j < v_per_thread; ++j) {
                o[j] *= factor;
            }
        }
    }

    // QSA also attends the incomplete block immediately after the selected
    // complete blocks. It contains at most BLOCK_SIZE - 1 tokens.
    int tail_start = (query_end / BLOCK_SIZE) * BLOCK_SIZE;
    int tail_pos = tail_start + int(simd_gid);
    bool valid_tail = tail_pos < query_end && tail_pos < key_length;
    if (valid_tail) {
        const device T* kptr =
            keys + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                     tail_pos) * D_SIZE) +
            int(simd_lid) * qk_per_thread;
        U score = 0;
        for (int j = 0; j < qk_per_thread; ++j) {
            score += q[j] * static_cast<U>(kptr[j]);
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        const device T* vptr =
            values + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                       tail_pos) * D_SIZE) +
            int(simd_lid) * v_per_thread;
        for (int j = 0; j < v_per_thread; ++j) {
            o[j] = o[j] * factor + exp_score * static_cast<U>(vptr[j]);
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

    for (int i = 0; i < v_per_thread; ++i) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = total_sum == 0 ? U(0) : (o[i] / total_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < v_per_thread; ++i) {
            optr[i] = static_cast<T>(o[i]);
        }
    }

}

template [[host_name("custom_kernel_qwen4_exp_qsa_sparse_attention_fp16_d64_s8_b8_k1_qh2_kh1__float16_t_64_4_2_1_2_8_1_8_float16_t_float16_t_float16_t_int32_tc_int32_tc_floatc_int32_tc_float16_t")]] [[kernel]] decltype(custom_kernel_qwen4_exp_qsa_sparse_attention_fp16_d64_s8_b8_k1_qh2_kh1__float16_t_64_4_2_1_2_8_1_8_float16_t_float16_t_float16_t_int32_tc_int32_tc_floatc_int32_tc_float16_t<float16_t, 64, 4, 2, 1, 2, 8, 1, 8>) custom_kernel_qwen4_exp_qsa_sparse_attention_fp16_d64_s8_b8_k1_qh2_kh1__float16_t_64_4_2_1_2_8_1_8_float16_t_float16_t_float16_t_int32_tc_int32_tc_floatc_int32_tc_float16_t<float16_t, 64, 4, 2, 1, 2, 8, 1, 8>;
