// Origin: mlx-vlm@8f5dc3d, mlx_vlm/quantization/one_bit.py:224.
// Representative call: x (1,512) float16, weight uint32 (64,16), scales/biases (64,8) float16; T=float16, group_size=64, 4 outputs per simdgroup, row-aligned; grid=(512,1,1), threadgroup=(64,1,1); out (1,64) float16.
template <typename T>
[[kernel]] void custom_kernel_mlx_vlm_affine_1bit_qmv_gs_64_rows_4_aligned_1__float16_t_float16_t_uint32_t_float16_t_float16_t_float16_t(
  const device float16_t* x [[buffer(0)]],
  const constant int* x_shape [[buffer(1)]],
  const device uint32_t* weight [[buffer(2)]],
  const constant int* weight_shape [[buffer(3)]],
  const device float16_t* scales [[buffer(4)]],
  const constant int* scales_shape [[buffer(5)]],
  const device float16_t* biases [[buffer(6)]],
  device float16_t* out [[buffer(7)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint input_row = threadgroup_position_in_grid.y;
    uint input_dims = x_shape[1];
    uint output_dims = weight_shape[0];
    uint groups = scales_shape[1];
    uint output_start = threadgroup_position_in_grid.x * 8 +
        simd_group * 4;

    float accumulators[4] = {0.0f};
    constexpr uint VALUES_PER_THREAD = 16;
    constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * 32;

    // Derived from PrismML's qmv_fast SIMD layout. Each lane owns one packed
    // 16-bit half-word and 16 contiguous activations, lowering register
    // pressure while four output rows reuse the activation block.
    for (uint block_start = lane * VALUES_PER_THREAD;
         block_start < input_dims;
         block_start += BLOCK_SIZE) {
        float x_thread[VALUES_PER_THREAD];
        float total_sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < VALUES_PER_THREAD; ++i) {
            float value = static_cast<float>(
                x[input_row * input_dims + block_start + i]);
            x_thread[i] = value;
            total_sum += value;
        }

        uint group = block_start / 64;
        uint packed_column = block_start >> 5;
        uint packed_shift = block_start & 31;
        for (uint row = 0; row < 4; ++row) {
            uint output_row = output_start + row;
            if (true) {
                ushort packed = ushort(
                    weight[output_row * weight_shape[1] + packed_column] >>
                    packed_shift);
                float selected_sum = 0.0f;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < VALUES_PER_THREAD; ++i) {
                    selected_sum +=
                        (packed & (ushort(1) << i)) ? x_thread[i] : 0.0f;
                }

                uint parameter_index = output_row * groups + group;
                accumulators[row] += selected_sum *
                    static_cast<float>(scales[parameter_index]);
                accumulators[row] += total_sum *
                    static_cast<float>(biases[parameter_index]);
            }
        }
    }

    for (uint row = 0; row < 4; ++row) {
        accumulators[row] = simd_sum(accumulators[row]);
        uint output_row = output_start + row;
        if (lane == 0 && output_row < output_dims) {
            out[input_row * output_dims + output_row] =
                static_cast<T>(accumulators[row]);
        }
    }

}

template [[host_name("custom_kernel_mlx_vlm_affine_1bit_qmv_gs_64_rows_4_aligned_1__float16_t_float16_t_uint32_t_float16_t_float16_t_float16_t")]] [[kernel]] decltype(custom_kernel_mlx_vlm_affine_1bit_qmv_gs_64_rows_4_aligned_1__float16_t_float16_t_uint32_t_float16_t_float16_t_float16_t<float16_t>) custom_kernel_mlx_vlm_affine_1bit_qmv_gs_64_rows_4_aligned_1__float16_t_float16_t_uint32_t_float16_t_float16_t_float16_t<float16_t>;
