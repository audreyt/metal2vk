// Origin: mlx-audio@17001a6, mlx_audio/sts/models/mossformer2_se/flash_attention_kernels.py:40.
// Representative call: q/k float32 (1,1,2,4), scale float32 (1), out float32 (1,1,2,2), T=float.
// Grid=(2,2,1), threadgroup=(2,2,1).
template <typename T>
[[kernel]] void custom_kernel_qk_relu_squared__float_float_float_floatc_float(
  const device float* q [[buffer(0)]],
  const constant int* q_shape [[buffer(1)]],
  const device float* k [[buffer(2)]],
  const constant int* k_shape [[buffer(3)]],
  const constant float* scale [[buffer(4)]],
  device float* out [[buffer(5)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        uint3 tid = thread_position_in_grid;
        uint batch = tid.z;
        uint row = tid.y;
        uint col = tid.x;

        if (row >= q_shape[2] || col >= k_shape[2]) return;

        uint groups = q_shape[1];
        uint q_dim = q_shape[3];

        uint q_stride = q_shape[2] * q_shape[3];
        uint k_stride = k_shape[2] * k_shape[3];
        uint out_stride = q_shape[2] * k_shape[2];

        T scale_val = scale[0];

        for (uint g = 0; g < groups; g++) {
            T sum = T(0);
            uint q_offset = batch * groups * q_stride + g * q_stride + row * q_dim;
            uint k_offset = batch * groups * k_stride + g * k_stride + col * q_dim;

            for (uint d = 0; d < q_dim; d++) {
                sum += q[q_offset + d] * k[k_offset + d];
            }

            sum *= scale_val;
            T result = sum > T(0) ? sum * sum : T(0);
            uint out_idx = batch * groups * out_stride + g * out_stride + row * k_shape[2] + col;
            out[out_idx] = result;
        }
}

template [[host_name("custom_kernel_qk_relu_squared__float_float_float_floatc_float")]] [[kernel]] decltype(custom_kernel_qk_relu_squared__float_float_float_floatc_float<float>) custom_kernel_qk_relu_squared__float_float_float_floatc_float<float>;
