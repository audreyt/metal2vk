// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/quantized_verifier.py:1974.
// Representative call: nvfp4 argmax: x (1,2,64) float16, w uint32 (64,8), scales uint8 (64,4); T=float16,VERIFY_T=2,K_SIZE=64,N_SIZE=64,NUM_TILES=8; grid=(32,16,1), threadgroup=(32,2,1); tile_values (1,2,8) float16, tile_indices (1,2,8) int32.

    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int VALUES_PER_THREAD = 8;
    constant constexpr int BLOCK_SIZE = SIMD_SIZE * VALUES_PER_THREAD;
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int ROWS_PER_THREADGROUP =
        RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

    inline float decode_e2m1(uint value) {
        half decoded = as_type<half>(ushort((value & 7) << 9));
        decoded *= 16384.0h;
        return float(value & 8 ? -decoded : decoded);
    }

    inline float decode_e4m3(uint8_t value) {
        half decoded = as_type<half>(ushort((value & 127) << 7));
        decoded *= 256.0h;
        return float(value & 128 ? -decoded : decoded);
    }

    inline float nvfp4_dot(
        const device uint8_t* weight,
        const thread float* values,
        float scale) {
        const device uint16_t* packed =
            reinterpret_cast<const device uint16_t*>(weight);
        float result = 0.0f;
        for (int index = 0; index < 2; ++index) {
            uint word = packed[index];
            result +=
                values[4 * index] * decode_e2m1(word) +
                values[4 * index + 1] * decode_e2m1(word >> 4) +
                values[4 * index + 2] * decode_e2m1(word >> 8) +
                values[4 * index + 3] * decode_e2m1(word >> 12);
        }
        return scale * result;
    }
template <typename T, int VERIFY_T, int K_SIZE, int N_SIZE, int NUM_TILES>
[[kernel]] void custom_kernel_quantized_verify_nvfp4_argmax_t2_k64_n64_fp16__float16_t_2_64_64_8_float16_t_uint32_t_uint8_t_float16_t_int32_t(
  const device float16_t* x [[buffer(0)]],
  const device uint32_t* w [[buffer(1)]],
  const device uint8_t* scales [[buffer(2)]],
  device float16_t* tile_values [[buffer(3)]],
  device int32_t* tile_indices [[buffer(4)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    uint output_tile = threadgroup_position_in_grid.y;
    uint batch_index = threadgroup_position_in_grid.z;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint output_row =
        output_tile * ROWS_PER_THREADGROUP +
        simd_group * RESULTS_PER_SIMDGROUP;

    threadgroup float group_values[VERIFY_T][NUM_SIMDGROUPS];
    threadgroup int group_indices[VERIFY_T][NUM_SIMDGROUPS];
    float results[VERIFY_T][RESULTS_PER_SIMDGROUP];
    float values[VERIFY_T][VALUES_PER_THREAD];
    for (int token = 0; token < VERIFY_T; ++token) {
        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            results[token][row] = 0.0f;
        }
    }

    const device uint8_t* weight_bytes =
        reinterpret_cast<const device uint8_t*>(w);
    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
        int lane_start = k + int(lane) * VALUES_PER_THREAD;
        if (lane_start < K_SIZE) {
            for (int token = 0; token < VERIFY_T; ++token) {
                const device T* input =
                    x + (int(batch_index) * VERIFY_T + token) * K_SIZE + lane_start;
                for (int index = 0; index < VALUES_PER_THREAD; ++index) {
                    values[token][index] = float(input[index]);
                }
            }

            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                int n = int(output_row) + row;
                const device uint8_t* weight =
                    weight_bytes + n * (K_SIZE / 2) + lane_start / 2;
                float scale = decode_e4m3(
                    scales[n * (K_SIZE / 16) + lane_start / 16]
                );
                for (int token = 0; token < VERIFY_T; ++token) {
                    results[token][row] += nvfp4_dot(weight, values[token], scale);
                }
            }
        }
    }

    for (int token = 0; token < VERIFY_T; ++token) {
        float best_value = -3.4028234663852886e38f;
        int best_index = 0;
        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int n = int(output_row) + row;
            float value = float(T(simd_sum(results[token][row])));
            if (value > best_value) {
                best_value = value;
                best_index = n;
            }
        }
        if (lane == 0) {
            group_values[token][simd_group] = best_value;
            group_indices[token][simd_group] = best_index;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && lane == 0) {
        for (int token = 0; token < VERIFY_T; ++token) {
            float best_value = group_values[token][0];
            int best_index = group_indices[token][0];
            for (int group = 1; group < NUM_SIMDGROUPS; ++group) {
                float value = group_values[token][group];
                if (value > best_value) {
                    best_value = value;
                    best_index = group_indices[token][group];
                }
            }
            int offset =
                (int(batch_index) * VERIFY_T + token) * NUM_TILES +
                int(output_tile);
            tile_values[offset] = T(best_value);
            tile_indices[offset] = best_index;
        }
    }

}

template [[host_name("custom_kernel_quantized_verify_nvfp4_argmax_t2_k64_n64_fp16__float16_t_2_64_64_8_float16_t_uint32_t_uint8_t_float16_t_int32_t")]] [[kernel]] decltype(custom_kernel_quantized_verify_nvfp4_argmax_t2_k64_n64_fp16__float16_t_2_64_64_8_float16_t_uint32_t_uint8_t_float16_t_int32_t<float16_t, 2, 64, 64, 8>) custom_kernel_quantized_verify_nvfp4_argmax_t2_k64_n64_fp16__float16_t_2_64_64_8_float16_t_uint32_t_uint8_t_float16_t_int32_t<float16_t, 2, 64, 64, 8>;
