// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/quantized_verifier.py:604.
// Representative call: MoE+HC epilogue: routed (1,2,2,64) float16, route_weights (1,2,2) float32, shared (1,2,64) float16, residual (1,2,2,64) float16, post/comb (1,2,2) float32; T=float16,ROWS=2,HC=2,D=64,SIMDS=8,TOP_K=2; grid=(64,8,1), threadgroup=(32,8,1); out (1,2,2,64).

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;
template <typename T, int ROWS, int HC, int D, int SIMDS, int TOP_K>
[[kernel]] void custom_kernel_quantized_verify_moe_hc_expand_fp16_r2_h2_d64_e2__float16_t_2_2_64_8_2_float16_t_floatc_float16_t_float16_t_floatc_floatc_float16_t(
  const device float16_t* routed [[buffer(0)]],
  const constant float* route_weights [[buffer(1)]],
  const device float16_t* shared [[buffer(2)]],
  const device float16_t* residual [[buffer(3)]],
  const constant float* post [[buffer(4)]],
  const constant float* comb [[buffer(5)]],
  device float16_t* out [[buffer(6)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

uint group = threadgroup_position_in_grid.x;
uint lane = thread_index_in_simdgroup;
uint simd_gid = simdgroup_index_in_threadgroup;
constexpr uint TILES = D / 8;
uint flat_tile = group * SIMDS + simd_gid;
uint row = flat_tile / TILES;
uint tile = flat_tile - row * TILES;
if (row >= ROWS) {
  return;
}

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
          row * HC * HC + source * HC + matrix_row];
    }
    b_values[element] = float(residual[
        row * HC * D + matrix_row * D + tile * 8 + matrix_col + element]);
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
    uint column = tile * 8 + matrix_col + element;
    T routed_sum = T(0);
    for (int route = 0; route < TOP_K; ++route) {
      int route_offset = (row * TOP_K + route) * D + column;
      T route_weight = T(route_weights[row * TOP_K + route]);
      T routed_product = T(routed[route_offset] * route_weight);
      routed_sum = T(routed_sum + routed_product);
    }
    T collapsed = T(routed_sum + shared[row * D + column]);
    volatile float product =
        post[row * HC + matrix_row] * float(collapsed);
    values[element] = product + values[element];
    out[row * HC * D + matrix_row * D + column] = T(values[element]);
  }
}

}

template [[host_name("custom_kernel_quantized_verify_moe_hc_expand_fp16_r2_h2_d64_e2__float16_t_2_2_64_8_2_float16_t_floatc_float16_t_float16_t_floatc_floatc_float16_t")]] [[kernel]] decltype(custom_kernel_quantized_verify_moe_hc_expand_fp16_r2_h2_d64_e2__float16_t_2_2_64_8_2_float16_t_floatc_float16_t_float16_t_floatc_floatc_float16_t<float16_t, 2, 2, 64, 8, 2>) custom_kernel_quantized_verify_moe_hc_expand_fp16_r2_h2_d64_e2__float16_t_2_2_64_8_2_float16_t_floatc_float16_t_float16_t_floatc_floatc_float16_t<float16_t, 2, 2, 64, 8, 2>;
