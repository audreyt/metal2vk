// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/inkling/language.py:136.
// Representative call: rel float32 (1,4,1,2), proj float32 (2,4), shape carrier (1,1,8), output (1,1,4,8).
// T=float, D_REL=2, REL_EXTENT=4, SLIDING=4; grid=(8,4,1), threadgroup=(8,8,1).
template <typename T, int D_REL, int REL_EXTENT, int SLIDING>
[[kernel]] void custom_kernel_inkling_banded_mask_v2__float_2_4_4_float_float_float_float(
  const device float* rel [[buffer(0)]],
  const constant int* rel_shape [[buffer(1)]],
  const constant int64_t* rel_strides [[buffer(2)]],
  const device float* proj [[buffer(3)]],
  const constant int64_t* proj_strides [[buffer(4)]],
  const device float* kshape [[buffer(5)]],
  const constant int* kshape_shape [[buffer(6)]],
  device float* out [[buffer(7)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    const int B  = rel_shape[0];
    const int LQ = rel_shape[1];
    const int H  = rel_shape[2];
    const int S  = kshape_shape[2];
    uint j  = thread_position_in_grid.x;
    uint i  = thread_position_in_grid.y;
    uint bh = thread_position_in_grid.z;
    if ((int)i >= LQ || (int)j >= S || (int)bh >= B * H) return;
    uint b = bh / H, h = bh % H;
    int dist = ((int)i + (S - LQ)) - (int)j;
    T val;
    if (dist < 0) {
        val = (T)(-1e30f);
    } else if (SLIDING > 0 && dist >= (int)SLIDING) {
        val = (T)(-1e30f);
    } else if (dist < (int)REL_EXTENT) {
        float acc = 0.0f;
        const size_t rbase = (size_t)b * rel_strides[0]
            + (size_t)i * rel_strides[1] + (size_t)h * rel_strides[2];
        for (uint d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + (size_t)d * rel_strides[3]]
                 * (float)proj[(size_t)d * proj_strides[0]
                               + (size_t)dist * proj_strides[1]];
        val = (T)acc;
    } else {
        val = (T)0;
    }
    out[(((size_t)b * H + h) * LQ + i) * S + j] = val;
}

template [[host_name("custom_kernel_inkling_banded_mask_v2__float_2_4_4_float_float_float_float")]] [[kernel]] decltype(custom_kernel_inkling_banded_mask_v2__float_2_4_4_float_float_float_float<float, 2, 4, 4>) custom_kernel_inkling_banded_mask_v2__float_2_4_4_float_float_float_float<float, 2, 4, 4>;
