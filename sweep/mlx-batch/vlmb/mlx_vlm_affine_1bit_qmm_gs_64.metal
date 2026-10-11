// Origin: mlx-vlm@8f5dc3d, mlx_vlm/quantization/one_bit.py:240.
// Representative call: x (16,512) float16, weight uint32 (64,16), scales/biases (64,8) float16; T=float16, group_size=64; grid=(128,1,1), threadgroup=(128,1,1); out (1024) float16.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;
template <typename T>
[[kernel]] void custom_kernel_mlx_vlm_affine_1bit_qmm_gs_64__float16_t_float16_t_uint32_t_float16_t_float16_t_float16_t(
  const device float16_t* x [[buffer(0)]],
  const constant int* x_shape [[buffer(1)]],
  const device uint32_t* weight [[buffer(2)]],
  const constant int* weight_shape [[buffer(3)]],
  const device float16_t* scales [[buffer(4)]],
  const constant int* scales_shape [[buffer(5)]],
  const device float16_t* biases [[buffer(6)]],
  const constant int* biases_shape [[buffer(7)]],
  device float16_t* out [[buffer(8)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    constexpr short BM = 32;
    constexpr short BK = 32;
    constexpr short BN = 32;
    constexpr short TILE_STRIDE = BK + 16 / sizeof(T);

    const int M = x_shape[0];
    const int K = x_shape[1];
    const int N = weight_shape[0];
    const uint lane = thread_index_in_simdgroup;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint thread_index = thread_index_in_threadgroup;
    const int tile_row = threadgroup_position_in_grid.y * BM;
    const int tile_column = threadgroup_position_in_grid.x * BN;
    const int load_row = thread_index / 4;
    const int load_column = (thread_index % 4) * 8;
    const int simd_row = simd_group / 2;
    const int simd_column = simd_group % 2;

    const short quad = lane / 4;
    const short fragment_row = (quad & 4) + ((lane / 2) % 4);
    const short fragment_column = (quad & 2) * 2 + (lane % 2) * 2;

    threadgroup T x_tile[BM * TILE_STRIDE];
    threadgroup T weight_tile[BN * TILE_STRIDE];

    metal::simdgroup_matrix<float, 8, 8> accumulators[2][2];
    #pragma clang loop unroll(full)
    for (short row = 0; row < 2; ++row) {
        #pragma clang loop unroll(full)
        for (short column = 0; column < 2; ++column) {
            accumulators[row][column].thread_elements()[0] = 0.0f;
            accumulators[row][column].thread_elements()[1] = 0.0f;
        }
    }

    for (int k_start = 0; k_start < K; k_start += BK) {
        int input_row = tile_row + load_row;
        int input_column = k_start + load_column;
        int weight_row = tile_column + load_row;
        int packed_column = k_start / 32;
        uint packed = weight_row < N
            ? weight[weight_row * weight_shape[1] + packed_column]
            : 0;
        uchar packed_byte = uchar(packed >> ((thread_index % 4) * 8));
        T scale = weight_row < N
            ? scales[weight_row * scales_shape[1] + k_start / 64]
            : T(0);
        T bias = weight_row < N
            ? biases[weight_row * biases_shape[1] + k_start / 64]
            : T(0);

        #pragma clang loop unroll(full)
        for (short element = 0; element < 8; ++element) {
            x_tile[load_row * TILE_STRIDE + load_column + element] =
                input_row < M ? x[input_row * K + input_column + element] : T(0);
            weight_tile[load_row * TILE_STRIDE + load_column + element] =
                bias + ((packed_byte & (uchar(1) << element)) ? scale : T(0));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (short k_fragment = 0; k_fragment < BK; k_fragment += 8) {
            metal::simdgroup_matrix<float, 8, 8> a[2];
            metal::simdgroup_matrix<float, 8, 8> b[2];
            #pragma clang loop unroll(full)
            for (short tile = 0; tile < 2; ++tile) {
                int a_row = simd_row * 16 + tile * 8 + fragment_row;
                int b_row = simd_column * 16 + tile * 8 + fragment_column;
                #pragma clang loop unroll(full)
                for (short element = 0; element < 2; ++element) {
                    a[tile].thread_elements()[element] = static_cast<float>(
                        x_tile[a_row * TILE_STRIDE + k_fragment +
                               fragment_column + element]);
                    b[tile].thread_elements()[element] = static_cast<float>(
                        weight_tile[(b_row + element) * TILE_STRIDE +
                                    k_fragment + fragment_row]);
                }
            }

            #pragma clang loop unroll(full)
            for (short row = 0; row < 2; ++row) {
                #pragma clang loop unroll(full)
                for (short column = 0; column < 2; ++column) {
                    metal::simdgroup_matrix<float, 8, 8> result;
                    simdgroup_multiply_accumulate(
                        result,
                        a[row],
                        b[column],
                        accumulators[row][column]);
                    accumulators[row][column] = result;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (short row = 0; row < 2; ++row) {
        #pragma clang loop unroll(full)
        for (short column = 0; column < 2; ++column) {
            int output_row = tile_row + simd_row * 16 + row * 8 + fragment_row;
            int output_column =
                tile_column + simd_column * 16 + column * 8 + fragment_column;
            #pragma clang loop unroll(full)
            for (short element = 0; element < 2; ++element) {
                if (output_row < M && output_column + element < N) {
                    out[output_row * N + output_column + element] = static_cast<T>(
                        accumulators[row][column].thread_elements()[element]);
                }
            }
        }
    }

}

template [[host_name("custom_kernel_mlx_vlm_affine_1bit_qmm_gs_64__float16_t_float16_t_uint32_t_float16_t_float16_t_float16_t")]] [[kernel]] decltype(custom_kernel_mlx_vlm_affine_1bit_qmm_gs_64__float16_t_float16_t_uint32_t_float16_t_float16_t_float16_t<float16_t>) custom_kernel_mlx_vlm_affine_1bit_qmm_gs_64__float16_t_float16_t_uint32_t_float16_t_float16_t_float16_t<float16_t>;
