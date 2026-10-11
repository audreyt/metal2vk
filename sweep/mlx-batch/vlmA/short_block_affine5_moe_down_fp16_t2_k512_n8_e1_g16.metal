// Origin: mlx-vlm@8f5dc3d, mlx-vlm/mlx_vlm/models/fast_ops.py:809.
// Representative call: affine 5-bit, fp16, same guard sizes as switch_gate_up, routed x (1,2,1,512), shared_y (1,2,8); grid=(32, 4, 2), threadgroup=(32, 4, 1).

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

constant constexpr int SIMD_SIZE = 32;
constant constexpr int PACK_FACTOR = 8;
constant constexpr int BYTES_PER_PACK = 5;
constant constexpr int PACKS_PER_THREAD = 2;
constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constant constexpr int SCALE_STEP_PER_THREAD =
    16 / VALUES_PER_THREAD;
constant constexpr int RESULTS_PER_SIMDGROUP = 2;
constant constexpr int NUM_SIMDGROUPS = 4;
constant constexpr int ROWS_PER_TG =
    RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

template <typename T>
inline float load_affine5_vector_exact(
    const device T* x,
    thread float* x_thread) {
  float sum = 0.0f;
  for (int i = 0; i < VALUES_PER_THREAD; i += 8) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
        x[i + 6] + x[i + 7];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 32.0f;
    x_thread[i + 2] = x[i + 2] / 4.0f;
    x_thread[i + 3] = x[i + 3] / 128.0f;
    x_thread[i + 4] = x[i + 4] / 16.0f;
    x_thread[i + 5] = x[i + 5] / 2.0f;
    x_thread[i + 6] = x[i + 6] / 64.0f;
    x_thread[i + 7] = x[i + 7] / 8.0f;
  }
  return sum;
}

inline float affine5_qdot_exact(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0.0f;
  for (int i = 0; i < VALUES_PER_THREAD / 8; ++i) {
    const thread float* xt = x_thread + 8 * i;
    const device uint8_t* wb = w + 5 * i;
    accum += (wb[0] & 0x1f) * xt[0];
    accum += (wb[0] & 0xe0) * xt[1];
    accum += (wb[1] & 0x03) * (xt[1] * 256.0f);
    accum += (wb[1] & 0x7c) * xt[2];
    accum += (wb[1] & 0x80) * xt[3];
    accum += (wb[2] & 0x0f) * (xt[3] * 256.0f);
    accum += (wb[2] & 0xf0) * xt[4];
    accum += (wb[3] & 0x01) * (xt[4] * 256.0f);
    accum += (wb[3] & 0x3e) * xt[5];
    accum += (wb[3] & 0xc0) * xt[6];
    accum += (wb[4] & 0x07) * (xt[6] * 256.0f);
    accum += (wb[4] & 0xf8) * xt[7];
  }
  return scale * accum + sum * bias;
}
template <typename T, int VERIFY_T, int K_SIZE, int N_SIZE, int TOP_K, int GROUP_SIZE>
[[kernel]] void custom_kernel_short_block_affine5_moe_down_fp16_t2_k512_n8_e1_g16__float16_t_2_512_8_1_16_float16_t_int32_tc_float16_tc_uint8_t_float16_t_float16_t_float16_t_float16_t(
  const device float16_t* x [[buffer(0)]],
  const constant int32_t* indices [[buffer(1)]],
  const constant float16_t* route_weights [[buffer(2)]],
  const device uint8_t* w [[buffer(3)]],
  const device float16_t* scales [[buffer(4)]],
  const device float16_t* biases [[buffer(5)]],
  const device float16_t* shared_y [[buffer(6)]],
  device float16_t* y [[buffer(7)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

uint n_tile = threadgroup_position_in_grid.y;
uint token = threadgroup_position_in_grid.z;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;

int out_row = int(n_tile) * ROWS_PER_TG +
    int(simd_gid) * RESULTS_PER_SIMDGROUP;
constexpr int W_ROW_BYTES = K_SIZE * 5 / 8;
constexpr int W_EXPERT_BYTES = N_SIZE * W_ROW_BYTES;
constexpr int GROUPS = K_SIZE / GROUP_SIZE;
constexpr int S_EXPERT_SIZE = N_SIZE * GROUPS;

T routed_sum[RESULTS_PER_SIMDGROUP] = {T(0)};
for (int route = 0; route < TOP_K; ++route) {
  int route_offset = int(token) * TOP_K + route;
  int expert = int(indices[route_offset]);
  const device uint8_t* ws = (const device uint8_t*)w +
      expert * W_EXPERT_BYTES + out_row * W_ROW_BYTES +
      int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
  const device T* sc = scales + expert * S_EXPERT_SIZE +
      out_row * GROUPS + int(simd_lid) / SCALE_STEP_PER_THREAD;
  const device T* bs = biases + expert * S_EXPERT_SIZE +
      out_row * GROUPS + int(simd_lid) / SCALE_STEP_PER_THREAD;
  const device T* xk = x + route_offset * K_SIZE +
      int(simd_lid) * VALUES_PER_THREAD;

  float result[RESULTS_PER_SIMDGROUP] = {0.0f};
  float x_thread[VALUES_PER_THREAD];
  for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
    float sum = load_affine5_vector_exact<T>(xk, x_thread);
    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      const device uint8_t* wr = ws + row * W_ROW_BYTES;
      const device T* sr = sc + row * GROUPS;
      const device T* br = bs + row * GROUPS;
      result[row] += affine5_qdot_exact(
          wr, x_thread, float(sr[0]), float(br[0]), sum);
    }
    ws += BLOCK_SIZE * 5 / 8;
    sc += BLOCK_SIZE / GROUP_SIZE;
    bs += BLOCK_SIZE / GROUP_SIZE;
    xk += BLOCK_SIZE;
  }

  T route_weight = T(route_weights[route_offset]);
  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    float value = simd_sum(result[row]);
    if (simd_lid == 0) {
      T product = T(T(value) * route_weight);
      routed_sum[row] = T(routed_sum[row] + product);
    }
  }
}

for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
  int n = out_row + row;
  if (simd_lid == 0 && n < N_SIZE) {
    y[int(token) * N_SIZE + n] =
        T(routed_sum[row] + shared_y[int(token) * N_SIZE + n]);
  }
}

}

template [[host_name("custom_kernel_short_block_affine5_moe_down_fp16_t2_k512_n8_e1_g16__float16_t_2_512_8_1_16_float16_t_int32_tc_float16_tc_uint8_t_float16_t_float16_t_float16_t_float16_t")]] [[kernel]] decltype(custom_kernel_short_block_affine5_moe_down_fp16_t2_k512_n8_e1_g16__float16_t_2_512_8_1_16_float16_t_int32_tc_float16_tc_uint8_t_float16_t_float16_t_float16_t_float16_t<float16_t, 2, 512, 8, 1, 16>) custom_kernel_short_block_affine5_moe_down_fp16_t2_k512_n8_e1_g16__float16_t_2_512_8_1_16_float16_t_int32_tc_float16_tc_uint8_t_float16_t_float16_t_float16_t_float16_t<float16_t, 2, 512, 8, 1, 16>;
