// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/quantized_verifier.py:1591.
// Representative call: streamed masked: x (1,8,512) float16, mask (8,2) int32; grid=(32,64,1).

    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int BITS = 4;
    constant constexpr int GS = 64;
    constant constexpr int PACK_FACTOR = (BITS == 5 ? 8 : 32 / BITS);
    constant constexpr int BYTES_PER_PACK = (BITS == 5 ? 5 : 32 / 8);
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
    constant constexpr int RESULTS_PER_SIMDGROUP = 1;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

    template <typename T>
    inline float load_vector_exact(const device T* x, thread float* x_thread) {
      float sum = 0.0f;
      if (BITS == 4) {
        for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1];
          x_thread[i + 2] = x[i + 2];
          x_thread[i + 3] = x[i + 3];
        }
      } else if (BITS == 5) {
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
      } else if (BITS == 8) {
        for (int i = 0; i < VALUES_PER_THREAD; i++) {
          sum += x[i];
          x_thread[i] = x[i];
        }
      }
      return sum;
    }

    inline float qdot_exact(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      float accum = 0.0f;
      if (BITS == 4) {
        const device uint16_t* ws = (const device uint16_t*)w;
        for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
          uint packed = ws[i];
          accum +=
              (x_thread[4 * i] * (packed & 0x000f) +
               x_thread[4 * i + 1] * ((packed >> 4) & 0x000f) +
               x_thread[4 * i + 2] * ((packed >> 8) & 0x000f) +
               x_thread[4 * i + 3] * ((packed >> 12) & 0x000f));
        }
      } else if (BITS == 5) {
        for (int i = 0; i < (VALUES_PER_THREAD / 8); i++) {
          const thread float* xt = x_thread + 8 * i;
          const device uint8_t* wb = w + 5 * i;

          accum += (wb[0] & 0x1f) * xt[0];
          accum += (wb[0] & 0xe0) * xt[1];
          accum += (wb[1] & 0x3) * (xt[1] * 256.0f);
          accum += (wb[1] & 0x7c) * xt[2];
          accum += (wb[1] & 0x80) * xt[3];
          accum += (wb[2] & 0xf) * (xt[3] * 256.0f);
          accum += (wb[2] & 0xf0) * xt[4];
          accum += (wb[3] & 0x1) * (xt[4] * 256.0f);
          accum += (wb[3] & 0x3e) * xt[5];
          accum += (wb[3] & 0xc0) * xt[6];
          accum += (wb[4] & 0x7) * (xt[6] * 256.0f);
          accum += (wb[4] & 0xf8) * xt[7];
        }
      } else if (BITS == 8) {
        for (int i = 0; i < VALUES_PER_THREAD; i++) {
          accum += x_thread[i] * w[i];
        }
      }
      return scale * accum + sum * bias;
    }

    inline float qdot_exact(
        const thread uint16_t* ws,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      float accum = 0.0f;
      if (BITS == 4) {
        for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
          uint packed = ws[i];
          accum +=
              (x_thread[4 * i] * (packed & 0x000f) +
               x_thread[4 * i + 1] * ((packed >> 4) & 0x000f) +
               x_thread[4 * i + 2] * ((packed >> 8) & 0x000f) +
               x_thread[4 * i + 3] * ((packed >> 12) & 0x000f));
        }
      }
      return scale * accum + sum * bias;
    }

template <typename T, int VERIFY_T, int K_SIZE, int N_SIZE, int NUM_TILES>
[[kernel]] void custom_kernel_quantized_verify_affine_masked_qargmax_streamed_b4_gs64_t8_k512_n64_fp16__float16_t_8_512_64_32_float16_t_uint32_t_float16_t_float16_t_int32_t_float16_t_int32_t(
  const device float16_t* x [[buffer(0)]],
  const device uint32_t* w [[buffer(1)]],
  const device float16_t* scales [[buffer(2)]],
  const device float16_t* biases [[buffer(3)]],
  const device int32_t* mask [[buffer(4)]],
  const constant int* mask_shape [[buffer(5)]],
  device float16_t* tile_values [[buffer(6)]],
  device int32_t* tile_indices [[buffer(7)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    uint n_tile = threadgroup_position_in_grid.y;
    uint b_idx = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    threadgroup float tile_best_values[VERIFY_T][NUM_SIMDGROUPS];
    threadgroup int tile_best_indices[VERIFY_T][NUM_SIMDGROUPS];

    const device uint8_t* ws =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc =
        scales + out_row * in_vec_size_g +
        int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs =
        biases + out_row * in_vec_size_g +
        int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xk =
        x + int(b_idx) * VERIFY_T * K_SIZE +
        int(simd_lid) * VALUES_PER_THREAD;

    float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        result[t][row] = 0.0f;
      }
    }

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint16_t* source =
            (const device uint16_t*)(ws + row * in_vec_size_w);
        uint16_t packed[VALUES_PER_THREAD / 4];
        for (int index = 0; index < VALUES_PER_THREAD / 4; ++index) {
          packed[index] = source[index];
        }
        float scale = float(sc[row * in_vec_size_g]);
        float bias = float(bs[row * in_vec_size_g]);
        for (int t = 0; t < VERIFY_T; ++t) {
          float x_thread[VALUES_PER_THREAD];
          float sum = load_vector_exact<T>(
              xk + t * K_SIZE, x_thread);
          result[t][row] += qdot_exact(
              packed, x_thread, scale, bias, sum);
        }
      }

      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xk += BLOCK_SIZE;
    }

    for (int t = 0; t < VERIFY_T; ++t) {
      float best_value = -3.4028234663852886e38f;
      int best_index = 0;
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        int n = out_row + row;
        float rounded = float(T(simd_sum(result[t][row])));
        if (
          n < N_SIZE &&
          ((as_type<uint>(mask[
                (int(b_idx) * VERIFY_T + t) * mask_shape[1] + (n >> 5)]) >>
            (n & 31)) & 1u) != 0u &&
          rounded > best_value) {
          best_value = rounded;
          best_index = n;
        }
      }

      if (simd_lid == 0) {
        tile_best_values[t][simd_gid] = best_value;
        tile_best_indices[t][simd_gid] = best_index;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0 && simd_lid == 0) {
      for (int t = 0; t < VERIFY_T; ++t) {
        float best = tile_best_values[t][0];
        int best_idx = tile_best_indices[t][0];
        for (int i = 1; i < NUM_SIMDGROUPS; ++i) {
          float candidate = tile_best_values[t][i];
          int candidate_idx = tile_best_indices[t][i];
          if (candidate > best) {
            best = candidate;
            best_idx = candidate_idx;
          }
        }
        int offset =
            (int(b_idx) * VERIFY_T + t) * NUM_TILES + int(n_tile);
        tile_values[offset] = T(best);
        tile_indices[offset] = best_idx;
      }
    }

}

template [[host_name("custom_kernel_quantized_verify_affine_masked_qargmax_streamed_b4_gs64_t8_k512_n64_fp16__float16_t_8_512_64_32_float16_t_uint32_t_float16_t_float16_t_int32_t_float16_t_int32_t")]] [[kernel]] decltype(custom_kernel_quantized_verify_affine_masked_qargmax_streamed_b4_gs64_t8_k512_n64_fp16__float16_t_8_512_64_32_float16_t_uint32_t_float16_t_float16_t_int32_t_float16_t_int32_t<float16_t, 8, 512, 64, 32>) custom_kernel_quantized_verify_affine_masked_qargmax_streamed_b4_gs64_t8_k512_n64_fp16__float16_t_8_512_64_32_float16_t_uint32_t_float16_t_float16_t_int32_t_float16_t_int32_t<float16_t, 8, 512, 64, 32>;
