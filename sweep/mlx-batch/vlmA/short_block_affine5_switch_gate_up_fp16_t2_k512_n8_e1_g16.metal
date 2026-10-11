// Origin: mlx-vlm@8f5dc3d, mlx-vlm/mlx_vlm/models/fast_ops.py:777.
// Representative call: affine 5-bit, fp16 x (1,2,512), n_size 8, top_k 1, group_size 16 (smallest sizes passing k%512==0, n%8==0, g%16==0), 2 experts of 8 rows; scales/biases (2, 8*32); grid=(32, 2, 2), threadgroup=(32, 2, 1).

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
constant constexpr int RESULTS_PER_SIMDGROUP = 4;
constant constexpr int NUM_SIMDGROUPS = 2;
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
[[kernel]] void custom_kernel_short_block_affine5_switch_gate_up_fp16_t2_k512_n8_e1_g16__float16_t_2_512_8_1_16_float16_t_int32_tc_uint8_t_float16_t_float16_t_uint8_t_float16_t_float16_t_float16_t_float16_t(
  const device float16_t* x [[buffer(0)]],
  const constant int32_t* indices [[buffer(1)]],
  const device uint8_t* up_w [[buffer(2)]],
  const device float16_t* up_scales [[buffer(3)]],
  const device float16_t* up_biases [[buffer(4)]],
  const device uint8_t* gate_w [[buffer(5)]],
  const device float16_t* gate_scales [[buffer(6)]],
  const device float16_t* gate_biases [[buffer(7)]],
  device float16_t* up_y [[buffer(8)]],
  device float16_t* gate_y [[buffer(9)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

uint n_tile = threadgroup_position_in_grid.y;
uint route = threadgroup_position_in_grid.z;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;

int out_row = int(n_tile) * ROWS_PER_TG +
    int(simd_gid) * RESULTS_PER_SIMDGROUP;
int token = int(route) / TOP_K;
int batch = token / VERIFY_T;
int time = token - batch * VERIFY_T;
int batch_route = batch * VERIFY_T * TOP_K;
int expert = int(indices[route]);
int paired_route = -1;
if ((time & 1) == 0 && time + 1 < VERIFY_T) {
  int next_route = batch_route + (time + 1) * TOP_K;
  for (int other = 0; other < TOP_K; ++other) {
    if (int(indices[next_route + other]) == expert) {
      paired_route = next_route + other;
      break;
    }
  }
} else if ((time & 1) != 0) {
  int previous_route = batch_route + (time - 1) * TOP_K;
  for (int other = 0; other < TOP_K; ++other) {
    if (int(indices[previous_route + other]) == expert) {
      return;
    }
  }
}
int pair_count = paired_route >= 0 ? 2 : 1;
constexpr int W_ROW_BYTES = K_SIZE * 5 / 8;
constexpr int W_EXPERT_BYTES = N_SIZE * W_ROW_BYTES;
constexpr int GROUPS = K_SIZE / GROUP_SIZE;
constexpr int S_EXPERT_SIZE = N_SIZE * GROUPS;

const device uint8_t* up_ws = (const device uint8_t*)up_w +
    expert * W_EXPERT_BYTES + out_row * W_ROW_BYTES +
    int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
const device T* up_sc = up_scales + expert * S_EXPERT_SIZE +
    out_row * GROUPS + int(simd_lid) / SCALE_STEP_PER_THREAD;
const device T* up_bs = up_biases + expert * S_EXPERT_SIZE +
    out_row * GROUPS + int(simd_lid) / SCALE_STEP_PER_THREAD;
const device uint8_t* gate_ws = (const device uint8_t*)gate_w +
    expert * W_EXPERT_BYTES + out_row * W_ROW_BYTES +
    int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
const device T* gate_sc = gate_scales + expert * S_EXPERT_SIZE +
    out_row * GROUPS + int(simd_lid) / SCALE_STEP_PER_THREAD;
const device T* gate_bs = gate_biases + expert * S_EXPERT_SIZE +
    out_row * GROUPS + int(simd_lid) / SCALE_STEP_PER_THREAD;
const device T* xk[2];
xk[0] = x + token * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;
int paired_token = paired_route >= 0 ? paired_route / TOP_K : token;
xk[1] = x + paired_token * K_SIZE +
    int(simd_lid) * VALUES_PER_THREAD;

float up_result[2][RESULTS_PER_SIMDGROUP] = {0.0f};
float gate_result[2][RESULTS_PER_SIMDGROUP] = {0.0f};
float x_thread[2][VALUES_PER_THREAD];

for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
  float sums[2];
  for (int pair = 0; pair < pair_count; ++pair) {
    sums[pair] = load_affine5_vector_exact<T>(xk[pair], x_thread[pair]);
  }
  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    const device uint8_t* uw = up_ws + row * W_ROW_BYTES;
    const device T* us = up_sc + row * GROUPS;
    const device T* ub = up_bs + row * GROUPS;
    const device uint8_t* gw = gate_ws + row * W_ROW_BYTES;
    const device T* gs = gate_sc + row * GROUPS;
    const device T* gb = gate_bs + row * GROUPS;
    for (int pair = 0; pair < pair_count; ++pair) {
      up_result[pair][row] += affine5_qdot_exact(
          uw, x_thread[pair], float(us[0]), float(ub[0]), sums[pair]);
      gate_result[pair][row] += affine5_qdot_exact(
          gw, x_thread[pair], float(gs[0]), float(gb[0]), sums[pair]);
    }
  }
  up_ws += BLOCK_SIZE * 5 / 8;
  up_sc += BLOCK_SIZE / GROUP_SIZE;
  up_bs += BLOCK_SIZE / GROUP_SIZE;
  gate_ws += BLOCK_SIZE * 5 / 8;
  gate_sc += BLOCK_SIZE / GROUP_SIZE;
  gate_bs += BLOCK_SIZE / GROUP_SIZE;
  xk[0] += BLOCK_SIZE;
  xk[1] += BLOCK_SIZE;
}

for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
  int n = out_row + row;
  for (int pair = 0; pair < pair_count; ++pair) {
    float up_value = simd_sum(up_result[pair][row]);
    float gate_value = simd_sum(gate_result[pair][row]);
    if (simd_lid == 0 && n < N_SIZE) {
      int output_route = pair == 0 ? int(route) : paired_route;
      up_y[output_route * N_SIZE + n] = T(up_value);
      gate_y[output_route * N_SIZE + n] = T(gate_value);
    }
  }
}

}

template [[host_name("custom_kernel_short_block_affine5_switch_gate_up_fp16_t2_k512_n8_e1_g16__float16_t_2_512_8_1_16_float16_t_int32_tc_uint8_t_float16_t_float16_t_uint8_t_float16_t_float16_t_float16_t_float16_t")]] [[kernel]] decltype(custom_kernel_short_block_affine5_switch_gate_up_fp16_t2_k512_n8_e1_g16__float16_t_2_512_8_1_16_float16_t_int32_tc_uint8_t_float16_t_float16_t_uint8_t_float16_t_float16_t_float16_t_float16_t<float16_t, 2, 512, 8, 1, 16>) custom_kernel_short_block_affine5_switch_gate_up_fp16_t2_k512_n8_e1_g16__float16_t_2_512_8_1_16_float16_t_int32_tc_uint8_t_float16_t_float16_t_uint8_t_float16_t_float16_t_float16_t_float16_t<float16_t, 2, 512, 8, 1, 16>;
