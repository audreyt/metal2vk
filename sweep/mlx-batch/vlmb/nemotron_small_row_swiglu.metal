// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/nemotron_labs_diffusion/language.py:435.
// Representative call: x (8,64) float16, gate/up weights (64,64) float16; T=float16,K=64,O=64,R=8; grid=(32,16,8), threadgroup=(32,1,8); out (8,64).
#include <metal_simdgroup>
using namespace metal;
template <typename T, int K, int O, int R>
[[kernel]] void custom_kernel_nemotron_small_row_swiglu__float16_t_64_64_8_float16_t_float16_t_float16_t_float16_t(
  const device float16_t* x [[buffer(0)]],
  const device float16_t* gate_weight [[buffer(1)]],
  const device float16_t* up_weight [[buffer(2)]],
  device float16_t* out [[buffer(3)]],
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

            const device T* in_vec = x + row * K;
            const device T* gate_mat = gate_weight + out_row * K;
            const device T* up_mat = up_weight + out_row * K;

            float gate_result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
            float up_result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
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
                        float value = v[tn];
                        gate_result[tm] +=
                            static_cast<float>(gate_mat[tm * K + col + tn]) * value;
                        up_result[tm] +=
                            static_cast<float>(up_mat[tm * K + col + tn]) * value;
                    }
                }

                col += blockN;
            }

            if (leftover > 0) {
                float v[TN];
                for (int tn = 0; tn < TN; ++tn) {
                    v[tn] = (col + tn < K) ? static_cast<float>(in_vec[col + tn]) : 0.0f;
                }

                for (int tm = 0; tm < TM; ++tm) {
                    for (int tn = 0; tn < TN; ++tn) {
                        float value = v[tn];
                        T gate_weight_value =
                            (col + tn < K) ? gate_mat[tm * K + col + tn] : T(0);
                        T up_weight_value =
                            (col + tn < K) ? up_mat[tm * K + col + tn] : T(0);
                        gate_result[tm] += static_cast<float>(gate_weight_value) * value;
                        up_result[tm] += static_cast<float>(up_weight_value) * value;
                    }
                }
            }

            for (int tm = 0; tm < TM; ++tm) {
                for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                    gate_result[tm] += simd_shuffle_down(gate_result[tm], sn);
                    up_result[tm] += simd_shuffle_down(up_result[tm], sn);
                }
            }

            if (lane == 0) {
                for (int tm = 0; tm < TM; ++tm) {
                    float gate = static_cast<float>(static_cast<T>(gate_result[tm]));
                    float up = static_cast<float>(static_cast<T>(up_result[tm]));
                    float activated = gate / (1.0f + exp(-gate));
                    out[row * O + out_row + tm] = static_cast<T>(activated * up);
                }
            }
        
}

template [[host_name("custom_kernel_nemotron_small_row_swiglu__float16_t_64_64_8_float16_t_float16_t_float16_t_float16_t")]] [[kernel]] decltype(custom_kernel_nemotron_small_row_swiglu__float16_t_64_64_8_float16_t_float16_t_float16_t_float16_t<float16_t, 64, 64, 8>) custom_kernel_nemotron_small_row_swiglu__float16_t_64_64_8_float16_t_float16_t_float16_t_float16_t<float16_t, 64, 64, 8>;
