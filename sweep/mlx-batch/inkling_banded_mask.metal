// Origin: mlx-vlm@8f5dc3d, mlx_vlm/models/inkling/language.py:70.
// Representative call: rel/proj float32, B=1,H=1,LQ=4,S=8,Q_OFF=4,D_REL=2,REL_EXTENT=4,SLIDING=4.
// Grid=(8,4,1), threadgroup=(8,8,1).
template <typename T, int B, int H, int LQ, int S, int Q_OFF, int D_REL, int REL_EXTENT, int SLIDING>
[[kernel]] void custom_kernel_inkling_banded_mask__float_1_1_4_8_4_2_4_4_float_float_float(
  const device float* rel [[buffer(0)]],
  const device float* proj [[buffer(1)]],
  device float* out [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint j  = thread_position_in_grid.x;
    uint i  = thread_position_in_grid.y;
    uint bh = thread_position_in_grid.z;
    if (i >= LQ || j >= S || bh >= B * H) return;
    uint b = bh / H, h = bh % H;
    int dist = (int(i) + int(Q_OFF)) - int(j);
    T val;
    if (dist < 0) {
        val = (T)(-1e30f);
    } else if (SLIDING > 0 && dist >= (int)SLIDING) {
        val = (T)(-1e30f);
    } else if (dist < (int)REL_EXTENT) {
        float acc = 0.0f;
        uint rbase = ((b * LQ + i) * H + h) * D_REL;
        uint pcol = (uint)dist;
        for (uint d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + d] * (float)proj[d * REL_EXTENT + pcol];
        val = (T)acc;
    } else {
        val = (T)0;
    }
    out[((b * H + h) * LQ + i) * S + j] = val;
}

template [[host_name("custom_kernel_inkling_banded_mask__float_1_1_4_8_4_2_4_4_float_float_float")]] [[kernel]] decltype(custom_kernel_inkling_banded_mask__float_1_1_4_8_4_2_4_4_float_float_float<float, 1, 1, 4, 8, 4, 2, 4, 4>) custom_kernel_inkling_banded_mask__float_1_1_4_8_4_2_4_4_float_float_float<float, 1, 1, 4, 8, 4, 2, 4, 4>;
