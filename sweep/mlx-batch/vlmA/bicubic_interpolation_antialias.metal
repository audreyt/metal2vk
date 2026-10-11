// Origin: mlx-vlm@8f5dc3d, mlx-vlm/mlx_vlm/models/kernels.py:401.
// Representative call: float32, 2x2 input upsampled to 4x4, batch*channels 1; dims/params arrays exactly as the caller builds them; input stays float32 so no dtype template is needed; grid=(4, 4, 1), threadgroup=(4, 4, 1).

        // Cubic convolution kernel. PyTorch's non-antialiased bicubic uses
        // a=-0.75 (ATen upsample_bicubic2d); its antialiased path matches
        // Pillow, which uses a=-0.5 (Keys' cubic).
        float cubic_kernel_a(float x, float a) {
            float absx = fabs(x);
            float absx2 = absx * absx;
            float absx3 = absx2 * absx;
            if (absx <= 1.0f) {
                return (a + 2.0f) * absx3 - (a + 3.0f) * absx2 + 1.0f;
            } else if (absx < 2.0f) {
                return a * absx3 - 5.0f * a * absx2 + 8.0f * a * absx - 4.0f * a;
            }
            return 0.0f;
        }

        float cubic_kernel(float x) {
            return cubic_kernel_a(x, -0.75f);
        }

        // Antialiased bicubic kernel - widens the support for downsampling and
        // uses Pillow's a=-0.5 to match PyTorch's antialiased path.
        float cubic_kernel_antialias(float x, float scale) {
            return cubic_kernel_a(x / scale, -0.5f);
        }
    [[kernel]] void custom_kernel_bicubic_interpolation_antialias_floatc_int32_tc_floatc_float(
  const constant float* input [[buffer(0)]],
  const constant int32_t* dims [[buffer(1)]],
  const constant float* params [[buffer(2)]],
  device float* output [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

        // Get thread position
        uint x_out = thread_position_in_grid.x;
        uint y_out = thread_position_in_grid.y;
        uint bc_idx = thread_position_in_grid.z;

        // Extract dimensions
        int batch_size = dims[0];
        int channels = dims[1];
        int in_h = dims[2];
        int in_w = dims[3];
        int out_h = dims[4];
        int out_w = dims[5];

        // Extract parameters
        float scale_h = params[0];
        float scale_w = params[1];
        bool align_corners = params[2] > 0.5f;
        bool use_antialias = params[3] > 0.5f;
        float filter_scale_h = params[4];
        float filter_scale_w = params[5];
        float support = params[6];

        // Check bounds
        if (x_out >= (uint)out_w || y_out >= (uint)out_h || bc_idx >= (uint)(batch_size * channels))
            return;

        // Calculate batch and channel indices
        int c = bc_idx % channels;
        int b = bc_idx / channels;

        // Calculate input coordinates
        float x_in, y_in;

        if (align_corners && out_w > 1 && out_h > 1) {
            x_in = float(x_out) * (in_w - 1) / (out_w - 1);
            y_in = float(y_out) * (in_h - 1) / (out_h - 1);
        } else {
            // PyTorch's default coordinate mapping
            x_in = ((float(x_out) + 0.5f) / float(out_w)) * float(in_w) - 0.5f;
            y_in = ((float(y_out) + 0.5f) / float(out_h)) * float(in_h) - 0.5f;
        }

        // Calculate the support region based on antialiasing
        float support_h = use_antialias ? support * filter_scale_h : support;
        float support_w = use_antialias ? support * filter_scale_w : support;

        // Calculate the range of input pixels to sample
        int y_start = int(floor(y_in - support_h)) + 1;
        int y_end = int(floor(y_in + support_h)) + 1;
        int x_start = int(floor(x_in - support_w)) + 1;
        int x_end = int(floor(x_in + support_w)) + 1;

        // Clamp to valid range
        y_start = max(0, y_start);
        y_end = min(in_h, y_end);
        x_start = max(0, x_start);
        x_end = min(in_w, x_end);

        // Perform bicubic interpolation with antialiasing
        float result = 0.0f;
        float weight_sum = 0.0f;

        for (int y_pos = y_start; y_pos < y_end; y_pos++) {
            float dy = float(y_pos) - y_in;
            float wy = use_antialias ?
                cubic_kernel_antialias(dy, filter_scale_h) :
                cubic_kernel(dy);

            for (int x_pos = x_start; x_pos < x_end; x_pos++) {
                float dx = float(x_pos) - x_in;
                float wx = use_antialias ?
                    cubic_kernel_antialias(dx, filter_scale_w) :
                    cubic_kernel(dx);

                float weight = wy * wx;

                // Calculate input tensor offset
                int input_offset = ((b * channels + c) * in_h + y_pos) * in_w + x_pos;

                // Add weighted contribution
                result += input[input_offset] * weight;
                weight_sum += weight;
            }
        }

        // Normalize by weight sum
        if (weight_sum > 1e-8f) {
            result /= weight_sum;
        }

        // Calculate output tensor offset
        int output_offset = ((b * channels + c) * out_h + y_out) * out_w + x_out;

        // Assign the result to output
        output[output_offset] = result;
    
}
