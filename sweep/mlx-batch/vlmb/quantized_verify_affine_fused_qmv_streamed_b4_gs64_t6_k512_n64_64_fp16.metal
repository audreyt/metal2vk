// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/quantized_verifier.py:1631.
// Representative call: fused two linears streamed: x (1,6,512) float16; VERIFY_T=6,N_SIZE=128; grid=(32,128,1), threadgroup=(32,2,1); y (1,6,128).

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

template <typename T, int VERIFY_T, int K_SIZE, int N_SIZE>
[[kernel]] void custom_kernel_quantized_verify_affine_fused_qmv_streamed_b4_gs64_t6_k512_n64_64_fp16__float16_t_6_512_128_float16_t_uint32_t_float16_t_float16_t_uint32_t_float16_t_float16_t_float16_t(
  const device float16_t* x [[buffer(0)]],
  const device uint32_t* w0 [[buffer(1)]],
  const device float16_t* scales0 [[buffer(2)]],
  const device float16_t* biases0 [[buffer(3)]],
  const device uint32_t* w1 [[buffer(4)]],
  const device float16_t* scales1 [[buffer(5)]],
  const device float16_t* biases1 [[buffer(6)]],
  device float16_t* y [[buffer(7)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    uint n_tile = threadgroup_position_in_grid.y;
    uint b_idx = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int global_out = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int local_out = global_out;
    const device uint8_t* selected_w = (const device uint8_t*)w0;
    const device T* selected_scales = scales0;
    const device T* selected_biases = biases0;
    if (global_out >= 64) {
      local_out = global_out - 64;
      selected_w = (const device uint8_t*)w1;
      selected_scales = scales1;
      selected_biases = biases1;
    }
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    const device uint8_t* ws =
        selected_w + local_out * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc =
        selected_scales + local_out * in_vec_size_g +
        int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs =
        selected_biases + local_out * in_vec_size_g +
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

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = global_out + row;
      for (int t = 0; t < VERIFY_T; ++t) {
        float value = simd_sum(result[t][row]);
        if (simd_lid == 0) {
          y[(int(b_idx) * VERIFY_T + t) * N_SIZE + n] = T(value);
        }
      }
    }

}

template [[host_name("custom_kernel_quantized_verify_affine_fused_qmv_streamed_b4_gs64_t6_k512_n64_64_fp16__float16_t_6_512_128_float16_t_uint32_t_float16_t_float16_t_uint32_t_float16_t_float16_t_float16_t")]] [[kernel]] decltype(custom_kernel_quantized_verify_affine_fused_qmv_streamed_b4_gs64_t6_k512_n64_64_fp16__float16_t_6_512_128_float16_t_uint32_t_float16_t_float16_t_uint32_t_float16_t_float16_t_float16_t<float16_t, 6, 512, 128>) custom_kernel_quantized_verify_affine_fused_qmv_streamed_b4_gs64_t6_k512_n64_64_fp16__float16_t_6_512_128_float16_t_uint32_t_float16_t_float16_t_uint32_t_float16_t_float16_t_float16_t<float16_t, 6, 512, 128>;
