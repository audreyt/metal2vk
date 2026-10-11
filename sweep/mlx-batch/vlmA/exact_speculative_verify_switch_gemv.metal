// Origin: mlx-vlm@8f5dc3d, mlx-vlm/mlx_vlm/models/exact_speculative_verify.py:86.
// Representative call: float32, 2 experts of (8,8); per-token uint32 indices (2,), BROADCAST_EXPERTS=false as when token shapes differ; grid=(32, 2, 8), threadgroup=(32, 1, 8).
#include <metal_simdgroup>
using namespace metal;
template <typename T, int K, int O, int R, int S, bool BROADCAST_EXPERTS>
[[kernel]] void custom_kernel_exact_speculative_verify_switch_gemv__float_8_8_2_1_false_float_float_uint32_tc_float(
  const device float* x [[buffer(0)]],
  const device float* weight [[buffer(1)]],
  const constant uint32_t* expert_indices [[buffer(2)]],
  device float* out [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

        uint lane = thread_position_in_grid.x;
        uint out_block = thread_position_in_grid.y;
        uint row = thread_position_in_grid.z;

        constexpr int TM = 4;
        constexpr int TN = 4;
        constexpr int SN = 32;
        constexpr int blockN = SN * TN;

        if (row >= R) {
            return;
        }

        int out_row = int(out_block * TM);
        if (out_row >= O) {
            return;
        }

        uint input_row = BROADCAST_EXPERTS ? row / S : row;
        uint expert = expert_indices[row];
        const device T* in_vec = x + input_row * K;
        const device T* mat = weight + expert * O * K + out_row * K;

        float result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
        int col = int(lane * TN);
        int n_iter = K / blockN;
        int leftover = K - blockN * n_iter;

        for (int iter = 0; iter < n_iter; ++iter) {
            float v[TN];
            for (int tn = 0; tn < TN; ++tn) {
                v[tn] = static_cast<float>(in_vec[col + tn]);
            }

            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    result[tm] +=
                        static_cast<float>(mat[tm * K + col + tn]) * v[tn];
                }
            }

            col += blockN;
        }

        if (leftover > 0) {
            float v[TN];
            for (int tn = 0; tn < TN; ++tn) {
                v[tn] =
                    (col + tn < K) ? static_cast<float>(in_vec[col + tn]) : 0.0f;
            }

            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    T m =
                        (col + tn < K) ? mat[tm * K + col + tn] : T(0);
                    result[tm] += static_cast<float>(m) * v[tn];
                }
            }
        }

        for (int tm = 0; tm < TM; ++tm) {
            for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                result[tm] += simd_shuffle_down(result[tm], sn);
            }
        }

        if (lane == 0) {
            for (int tm = 0; tm < TM; ++tm) {
                out[row * O + out_row + tm] = static_cast<T>(result[tm]);
            }
        }
    
}

template [[host_name("custom_kernel_exact_speculative_verify_switch_gemv__float_8_8_2_1_false_float_float_uint32_tc_float")]] [[kernel]] decltype(custom_kernel_exact_speculative_verify_switch_gemv__float_8_8_2_1_false_float_float_uint32_tc_float<float, 8, 8, 2, 1, false>) custom_kernel_exact_speculative_verify_switch_gemv__float_8_8_2_1_false_float_float_uint32_tc_float<float, 8, 8, 2, 1, false>;
