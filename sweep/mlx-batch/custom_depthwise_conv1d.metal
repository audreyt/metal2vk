// Origin: mlx-audio@17001a6, mlx_audio/sts/models/mossformer2_se/depthwise_conv1d_kernel.py:15.
// Representative call: inp float32 (1,4,2), weight float32 (2,3,1), params int32 (6), T=float.
// Grid=(2,4,1), threadgroup=(1,1,1); weight uses constant storage because it has 6 elements.
template <typename T>
[[kernel]] void custom_kernel_custom_depthwise_conv1d__float_float_floatc_int32_tc_float(
  const device float* inp [[buffer(0)]],
  const constant float* weight [[buffer(1)]],
  const constant int32_t* params [[buffer(2)]],
  device float* out [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        const int chan = int(thread_position_in_grid.x);
        const int time = int(thread_position_in_grid.y);
        const int batch = int(thread_position_in_grid.z);

        const int B = params[0];
        const int L_in = params[1];
        const int C = params[2];
        const int K = params[3];
        const int pad = params[4];
        const int L_out = params[5];

        if (batch >= B || time >= L_out || chan >= C) {
            return;
        }

        const int out_index = ((batch * L_out) + time) * C + chan;
        const int weight_base = chan * K;

        T acc = T(0);
        for (int k = 0; k < K; ++k) {
            const int in_time = time + k - pad;
            if (in_time >= 0 && in_time < L_in) {
                const int in_index = ((batch * L_in) + in_time) * C + chan;
                acc += inp[in_index] * weight[weight_base + k];
            }
        }

        out[out_index] = acc;
}

template [[host_name("custom_kernel_custom_depthwise_conv1d__float_float_floatc_int32_tc_float")]] [[kernel]] decltype(custom_kernel_custom_depthwise_conv1d__float_float_floatc_int32_tc_float<float>) custom_kernel_custom_depthwise_conv1d__float_float_floatc_int32_tc_float<float>;
