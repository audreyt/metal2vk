// Origin: mlx-vlm@8f5dc3d, mlx-vlm/mlx_vlm/models/fast_ops.py:756.
// Representative call: fp16 x (1,2,8), residual (1,2,4,8), rows 2, simds 8 as in the caller; fp16 per the caller dtype gate; grid=(32, 8, 1), threadgroup=(32, 8, 1).

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;
template <typename T, int ROWS, int HC, int D, int SIMDS>
[[kernel]] void custom_kernel_short_block_hc_expand_fp16_r2_h4_d8__float16_t_2_4_8_8_float16_t_float16_t_float_float_float16_t(
  const device float16_t* x [[buffer(0)]],
  const device float16_t* residual [[buffer(1)]],
  const device float* post [[buffer(2)]],
  const device float* comb [[buffer(3)]],
  device float16_t* out [[buffer(4)]],
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
    volatile float product =
        post[row * HC + matrix_row] * float(x[row * D + column]);
    values[element] = product + values[element];
    out[row * HC * D + matrix_row * D + column] = T(values[element]);
  }
}

}

template [[host_name("custom_kernel_short_block_hc_expand_fp16_r2_h4_d8__float16_t_2_4_8_8_float16_t_float16_t_float_float_float16_t")]] [[kernel]] decltype(custom_kernel_short_block_hc_expand_fp16_r2_h4_d8__float16_t_2_4_8_8_float16_t_float16_t_float_float_float16_t<float16_t, 2, 4, 8, 8>) custom_kernel_short_block_hc_expand_fp16_r2_h4_d8__float16_t_2_4_8_8_float16_t_float16_t_float_float_float16_t<float16_t, 2, 4, 8, 8>;
