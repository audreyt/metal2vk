// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/kernels.py:546.
// Representative call: x float32 (1,2,2,2), grid float32 (1,2,2,2), out float32 (1,2,2,2), T=float.
// Grid=(8,1,1), threadgroup=(8,1,1).
template <typename T>
[[kernel]] void custom_kernel_grid_sample__float_float_float_float(
  const device float* x [[buffer(0)]],
  const constant int* x_shape [[buffer(1)]],
  const device float* grid [[buffer(2)]],
  const constant int* grid_shape [[buffer(3)]],
  device float* out [[buffer(4)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        uint elem = thread_position_in_grid.x;
        int H = x_shape[1];
        int W = x_shape[2];
        int C = x_shape[3];
        int gH = grid_shape[1];
        int gW = grid_shape[2];

        int w_stride = C;
        int h_stride = W * w_stride;
        int b_stride = H * h_stride;

        uint grid_idx = elem / C * 2;
        float ix = ((grid[grid_idx] + 1) * W - 1) / 2;
        float iy = ((grid[grid_idx + 1] + 1) * H - 1) / 2;

        int ix_nw = floor(ix);
        int iy_nw = floor(iy);

        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;

        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;

        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        T nw = (ix_se - ix) * (iy_se - iy);
        T ne = (ix - ix_sw) * (iy_sw - iy);
        T sw = (ix_ne - ix) * (iy - iy_ne);
        T se = (ix - ix_nw) * (iy - iy_nw);

        int batch_idx = elem / C / gH / gW * b_stride;
        int channel_idx = elem % C;
        int base_idx = batch_idx + channel_idx;

        T I_nw = x[base_idx + iy_nw * h_stride + ix_nw * w_stride];
        T I_ne = x[base_idx + iy_ne * h_stride + ix_ne * w_stride];
        T I_sw = x[base_idx + iy_sw * h_stride + ix_sw * w_stride];
        T I_se = x[base_idx + iy_se * h_stride + ix_se * w_stride];

        I_nw = iy_nw >= 0 && iy_nw <= H - 1 && ix_nw >= 0 && ix_nw <= W - 1 ? I_nw : 0;
        I_ne = iy_ne >= 0 && iy_ne <= H - 1 && ix_ne >= 0 && ix_ne <= W - 1 ? I_ne : 0;
        I_sw = iy_sw >= 0 && iy_sw <= H - 1 && ix_sw >= 0 && ix_sw <= W - 1 ? I_sw : 0;
        I_se = iy_se >= 0 && iy_se <= H - 1 && ix_se >= 0 && ix_se <= W - 1 ? I_se : 0;

        out[elem] = nw * I_nw + ne * I_ne + sw * I_sw + se * I_se;
}

template [[host_name("custom_kernel_grid_sample__float_float_float_float")]] [[kernel]] decltype(custom_kernel_grid_sample__float_float_float_float<float>) custom_kernel_grid_sample__float_float_float_float<float>;
