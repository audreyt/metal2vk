// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/sparse_attention.py:131.
// Representative call: queries/keys/values (1,2,4,64)/(1,1,8,64)/(1,1,8,64) float16, indices (1,4,2) int32, scale (1) float32, q_size/k_size (1) int32; T=float16,QK_DIM=64,V_DIM=64,TOPK=2,NUM_Q_HEADS=2,NUM_KV_HEADS=1,GQA_FACTOR=2; grid=(1024,8,1), threadgroup=(1024,1,1); out (1,2,4,64).
#include <metal_simdgroup>
using namespace metal;
template <typename T, int QK_DIM, int V_DIM, int TOPK, int NUM_Q_HEADS, int NUM_KV_HEADS, int GQA_FACTOR>
[[kernel]] void custom_kernel_indexed_sparse_attention_fp16_q64_v64_k2_qh2_kh1__float16_t_64_64_2_2_1_2_float16_t_float16_t_float16_t_int32_t_floatc_int32_tc_int32_tc_float16_t(
  const device float16_t* queries [[buffer(0)]],
  const device float16_t* keys [[buffer(1)]],
  const device float16_t* values [[buffer(2)]],
  const device int32_t* indices [[buffer(3)]],
  const constant float* scale [[buffer(4)]],
  const constant int32_t* q_size [[buffer(5)]],
  const constant int32_t* k_size [[buffer(6)]],
  device float16_t* out [[buffer(7)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    uint row_idx = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    constexpr int SIMD_GROUPS = 32;
    constexpr int SIMD_WIDTH = 32;
    constexpr int qk_per_thread = QK_DIM / SIMD_WIDTH;
    constexpr int v_per_thread = V_DIM / SIMD_WIDTH;

    typedef float U;
    thread U q[qk_per_thread];
    thread U o[v_per_thread];
    threadgroup U outputs[SIMD_GROUPS * SIMD_WIDTH];
    threadgroup U max_scores[SIMD_GROUPS];
    threadgroup U sum_exp_scores[SIMD_GROUPS];

    int query_length = int(q_size[0]);
    int key_length = int(k_size[0]);
    int query_idx = int(row_idx % query_length);
    int batch_head_idx = int(row_idx / query_length);
    int batch_idx = batch_head_idx / NUM_Q_HEADS;
    int q_head_idx = batch_head_idx - batch_idx * NUM_Q_HEADS;
    int kv_head_idx = q_head_idx / GQA_FACTOR;

    const device T* qptr =
        queries + ((batch_head_idx * query_length + query_idx) * QK_DIM) +
        int(simd_lid) * qk_per_thread;
    device T* optr =
        out + ((batch_head_idx * query_length + query_idx) * V_DIM) +
        int(simd_gid) * v_per_thread;

    U s = U(scale[0]);
    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = s * static_cast<U>(qptr[i]);
    }
    for (int i = 0; i < v_per_thread; i++) {
        o[i] = 0;
    }

    int indices_offset = (batch_idx * query_length + query_idx) * TOPK;
    U max_score = -3.4028234663852886e38f;
    U sum_exp_score = 0;

    for (int selected_idx = int(simd_gid); selected_idx < TOPK;
         selected_idx += SIMD_GROUPS) {
        int key_pos = int(indices[indices_offset + selected_idx]);
        bool valid = key_pos >= 0 && key_pos < key_length;

        U score = -3.4028234663852886e38f;
        if (valid) {
            const device T* kptr =
                keys + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                         key_pos) *
                        QK_DIM) +
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
                          V_DIM) +
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
        outputs[simd_lid * SIMD_WIDTH + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(
            outputs[simd_gid * SIMD_WIDTH + simd_lid] * factor
        );
        o[i] = total_sum == 0 ? U(0) : (o[i] / total_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < v_per_thread; i++) {
            optr[i] = static_cast<T>(o[i]);
        }
    }

}

template [[host_name("custom_kernel_indexed_sparse_attention_fp16_q64_v64_k2_qh2_kh1__float16_t_64_64_2_2_1_2_float16_t_float16_t_float16_t_int32_t_floatc_int32_tc_int32_tc_float16_t")]] [[kernel]] decltype(custom_kernel_indexed_sparse_attention_fp16_q64_v64_k2_qh2_kh1__float16_t_64_64_2_2_1_2_float16_t_float16_t_float16_t_int32_t_floatc_int32_tc_int32_tc_float16_t<float16_t, 64, 64, 2, 2, 1, 2>) custom_kernel_indexed_sparse_attention_fp16_q64_v64_k2_qh2_kh1__float16_t_64_64_2_2_1_2_float16_t_float16_t_float16_t_int32_t_floatc_int32_tc_int32_tc_float16_t<float16_t, 64, 64, 2, 2, 1, 2>;
