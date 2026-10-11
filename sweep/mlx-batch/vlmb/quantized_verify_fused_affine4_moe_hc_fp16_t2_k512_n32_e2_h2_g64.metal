// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/quantized_verifier.py:477.
// Representative call: fused affine MoE+HC: x (1,2,2,512) float16, indices (1,2,2) int32, route_weights (1,2,2) float32, w uint32 (32,64), scales/biases (32,8) float16, shared (1,2,32), residual (1,2,2,32) float16, post/comb (1,2,2) float32; T=float16,BITS=4,GROUP_SIZE=64,K_SIZE=512,N_SIZE=32,TOP_K=2,HC=2; grid=(32,16,2), threadgroup=(32,4,1); out (1,2,2,32).

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

constant constexpr int PACK_FACTOR =
    (4 == 3 || 4 == 5) ? 8 : (4 == 6 ? 4 : 32 / 4);
constant constexpr int BYTES_PER_PACK =
    ((4 & (4 - 1)) == 0) ? 4 : (4 == 5 ? 5 : 3);
constant constexpr int PACKS_PER_THREAD = 4 == 2 ? 1 : 2;
constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
constant constexpr int SCALE_STEP_PER_THREAD =
    64 / VALUES_PER_THREAD;
constant constexpr bool HAS_AFFINE_BIAS = true;

template <typename T>
inline float load_quantized_vector(
    const device T* x,
    thread float* x_thread) {
  float sum = 0.0f;
  if constexpr (4 == 3 || 4 == 5) {
    for (int i = 0; i < VALUES_PER_THREAD; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
    }
  } else if constexpr (4 == 8) {
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
      sum += x[i];
    }
  } else {
    for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    }
  }
  for (int i = 0; i < VALUES_PER_THREAD; ++i) {
    int shift = (i * 4) % 8;
    x_thread[i] = x[i] / float(1 << shift);
  }
  return sum;
}

inline float quantized_dot(
    const device uint8_t* w,
    const thread float* x,
    float scale,
    float bias,
    float sum) {
  float accum = 0.0f;
  constexpr uint MASK = (1u << 4) - 1;
  if constexpr (4 == 2 || 4 == 4) {
    for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
      float partial = 0.0f;
      for (int element = 0; element < 4; ++element) {
        int value_index = i + element;
        int bit = value_index * 4;
        int byte = bit / 8;
        int shift = bit % 8;
        partial += x[value_index] * (w[byte] & ((MASK << shift) & 0xff));
      }
      accum += partial;
    }
  } else {
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
      int bit = i * 4;
      int byte = bit / 8;
      int shift = bit % 8;
      accum += x[i] * (w[byte] & ((MASK << shift) & 0xff));
      if (shift + 4 > 8) {
        uint high_mask = MASK >> (8 - shift);
        accum += (w[byte + 1] & high_mask) * (x[i] * 256.0f);
      }
    }
  }
  return scale * accum + sum * bias;
}

template <typename T>
inline float quantized_scale(T value) {
  return float(value);
}
template <typename T, int BITS, int GROUP_SIZE, int K_SIZE, int N_SIZE, int TOP_K, int HC>
[[kernel]] void custom_kernel_quantized_verify_fused_affine4_moe_hc_fp16_t2_k512_n32_e2_h2_g64__float16_t_4_64_512_32_2_2_float16_t_int32_tc_floatc_uint32_t_float16_t_float16_t_float16_t_float16_t_floatc_floatc_float16_t(
  const device float16_t* x [[buffer(0)]],
  const constant int32_t* indices [[buffer(1)]],
  const constant float* route_weights [[buffer(2)]],
  const device uint32_t* w [[buffer(3)]],
  const device float16_t* scales [[buffer(4)]],
  const device float16_t* biases [[buffer(5)]],
  const device float16_t* shared [[buffer(6)]],
  const device float16_t* residual [[buffer(7)]],
  const constant float* post [[buffer(8)]],
  const constant float* comb [[buffer(9)]],
  device float16_t* out [[buffer(10)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

uint n_tile = threadgroup_position_in_grid.y;
uint token = threadgroup_position_in_grid.z;
uint simd_gid = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

constexpr int RESULTS_PER_SIMDGROUP = 2;
constexpr int NUM_SIMDGROUPS = 4;
constexpr int ROWS_PER_TG = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;
constexpr int W_ROW_BYTES = K_SIZE * BITS / 8;
constexpr int W_EXPERT_BYTES = N_SIZE * W_ROW_BYTES;
constexpr int GROUPS = K_SIZE / GROUP_SIZE;
constexpr int S_EXPERT_SIZE = N_SIZE * GROUPS;
int out_row = int(n_tile) * ROWS_PER_TG +
    int(simd_gid) * RESULTS_PER_SIMDGROUP;

T routed_sum[RESULTS_PER_SIMDGROUP] = {T(0)};
for (int route = 0; route < TOP_K; ++route) {
  int route_offset = int(token) * TOP_K + route;
  int expert = int(indices[route_offset]);
  const device uint8_t* ws = (const device uint8_t*)w +
      expert * W_EXPERT_BYTES + out_row * W_ROW_BYTES +
      int(lane) * PACKS_PER_THREAD * BYTES_PER_PACK;
  const device auto* sc = scales + expert * S_EXPERT_SIZE +
      out_row * GROUPS + int(lane) / SCALE_STEP_PER_THREAD;
  const device T* bs = biases;
  if constexpr (HAS_AFFINE_BIAS) {
    bs += expert * S_EXPERT_SIZE + out_row * GROUPS +
        int(lane) / SCALE_STEP_PER_THREAD;
  }
  const device T* xk = x + route_offset * K_SIZE +
      int(lane) * VALUES_PER_THREAD;

  float result[RESULTS_PER_SIMDGROUP] = {0.0f};
  float x_thread[VALUES_PER_THREAD];
  for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
    float sum = load_quantized_vector<T>(xk, x_thread);
    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      const device uint8_t* wr = ws + row * W_ROW_BYTES;
      const device auto* sr = sc + row * GROUPS;
      const device T* br = bs + (HAS_AFFINE_BIAS ? row * GROUPS : 0);
      float bias = HAS_AFFINE_BIAS ? float(br[0]) : 0.0f;
      result[row] += quantized_dot(
          wr, x_thread, quantized_scale(sr[0]), bias, sum);
    }
    ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    sc += BLOCK_SIZE / GROUP_SIZE;
    if constexpr (HAS_AFFINE_BIAS) {
      bs += BLOCK_SIZE / GROUP_SIZE;
    }
    xk += BLOCK_SIZE;
  }

  T route_weight = T(route_weights[route_offset]);
  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    float value = simd_sum(result[row]);
    if (lane == 0) {
      T product = T(T(value) * route_weight);
      routed_sum[row] = T(routed_sum[row] + product);
    }
  }
}

threadgroup T collapsed_tile[ROWS_PER_TG];
for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
  int n = out_row + row;
  if (lane == 0) {
    collapsed_tile[int(simd_gid) * RESULTS_PER_SIMDGROUP + row] =
        T(routed_sum[row] + shared[int(token) * N_SIZE + n]);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (simd_gid == 0) {
  short qid = lane / 4;
  short matrix_row = (qid & 4) + ((lane / 2) % 4);
  short matrix_col = (qid & 2) * 2 + (lane % 2) * 2;
  float2 a_values = 0.0f;
  float2 b_values = 0.0f;
  if (matrix_row < HC) {
    for (short element = 0; element < 2; ++element) {
      short source = matrix_col + element;
      if (source < HC) {
        a_values[element] = comb[
            int(token) * HC * HC + source * HC + matrix_row];
      }
      b_values[element] = float(residual[
          int(token) * HC * N_SIZE + matrix_row * N_SIZE +
          int(n_tile) * ROWS_PER_TG + matrix_col + element]);
    }
  }

  simdgroup_matrix<float, 8, 8> a_matrix;
  simdgroup_matrix<float, 8, 8> b_matrix;
  simdgroup_matrix<float, 8, 8> c_matrix;
  simdgroup_matrix<float, 8, 8> d_matrix;
  reinterpret_cast<thread float2&>(a_matrix.thread_elements()) = a_values;
  reinterpret_cast<thread float2&>(b_matrix.thread_elements()) = b_values;
  reinterpret_cast<thread float2&>(c_matrix.thread_elements()) = float2(0.0f);
  simdgroup_multiply_accumulate(d_matrix, a_matrix, b_matrix, c_matrix);

  float2 values =
      reinterpret_cast<thread float2&>(d_matrix.thread_elements());
  if (matrix_row < HC) {
    for (short element = 0; element < 2; ++element) {
      int column = int(n_tile) * ROWS_PER_TG + matrix_col + element;
      volatile float product = post[int(token) * HC + matrix_row] *
          float(collapsed_tile[matrix_col + element]);
      values[element] = product + values[element];
      out[int(token) * HC * N_SIZE + matrix_row * N_SIZE + column] =
          T(values[element]);
    }
  }
}

}

template [[host_name("custom_kernel_quantized_verify_fused_affine4_moe_hc_fp16_t2_k512_n32_e2_h2_g64__float16_t_4_64_512_32_2_2_float16_t_int32_tc_floatc_uint32_t_float16_t_float16_t_float16_t_float16_t_floatc_floatc_float16_t")]] [[kernel]] decltype(custom_kernel_quantized_verify_fused_affine4_moe_hc_fp16_t2_k512_n32_e2_h2_g64__float16_t_4_64_512_32_2_2_float16_t_int32_tc_floatc_uint32_t_float16_t_float16_t_float16_t_float16_t_floatc_floatc_float16_t<float16_t, 4, 64, 512, 32, 2, 2>) custom_kernel_quantized_verify_fused_affine4_moe_hc_fp16_t2_k512_n32_e2_h2_g64__float16_t_4_64_512_32_2_2_float16_t_int32_tc_floatc_uint32_t_float16_t_float16_t_float16_t_float16_t_floatc_floatc_float16_t<float16_t, 4, 64, 512, 32, 2, 2>;
