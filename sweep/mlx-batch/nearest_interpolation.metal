// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/kernels.py:92.
// Representative call: float32 input (16), int32 dims (6), float32 output (4); dims=(1,1,4,4,2,2).
// Grid=(2,2,1), threadgroup=(2,2,1).
[[kernel]] void custom_kernel_nearest_interpolation_float_int32_tc_float(
  const device float* input [[buffer(0)]],
  const constant int32_t* dims [[buffer(1)]],
  device float* output [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        uint x_out = thread_position_in_grid.x;
        uint y_out = thread_position_in_grid.y;
        uint bc_idx = thread_position_in_grid.z;

        int batch_size = dims[0];
        int channels = dims[1];
        int in_h = dims[2];
        int in_w = dims[3];
        int out_h = dims[4];
        int out_w = dims[5];

        if (x_out >= (uint)out_w || y_out >= (uint)out_h || bc_idx >= (uint)(batch_size * channels))
            return;

        int c = bc_idx % channels;
        int b = bc_idx / channels;

        float scale_h = float(in_h) / float(out_h);
        float scale_w = float(in_w) / float(out_w);

        int y_in = int(floor(float(y_out) * scale_h));
        int x_in = int(floor(float(x_out) * scale_w));

        y_in = max(0, min(y_in, in_h - 1));
        x_in = max(0, min(x_in, in_w - 1));

        int input_offset = ((b * channels + c) * in_h + y_in) * in_w + x_in;
        int output_offset = ((b * channels + c) * out_h + y_out) * out_w + x_out;

        output[output_offset] = input[input_offset];
}
