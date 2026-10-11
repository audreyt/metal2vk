// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:2889.
// Representative call: no-rotation quantize: rotated (1,64) float16, midpoints (15) float32; Dim=64,PackedWidth=2; grid=(64,1,1), threadgroup=(64,1,1); out (1,2) uint32.
template <int Dim, int PackedWidth>
[[kernel]] void custom_kernel_turboquant_norot_quantize_4__64_2_float16_t_float_uint32_t(
  const device float16_t* rotated [[buffer(0)]],
  const device float* midpoints [[buffer(1)]],
  device uint32_t* out [[buffer(2)]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

        auto d = thread_position_in_threadgroup.x;
        auto bh = threadgroup_position_in_grid.x;

        // Read pre-rotated, pre-normalized value
        float val = (d < Dim) ? static_cast<float>(rotated[bh * Dim + d]) : 0.0f;

        // Comparison-based quantize
        threadgroup uint shared_idx[Dim];
        uint idx = 0;
        if (d < Dim) {
            for (int m = 0; m < 15; m++)
                idx += (val > midpoints[m]) ? 1u : 0u;
            shared_idx[d] = idx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pack indices: thread d builds word d in a private register
        // and writes it once (no shared-memory race, no atomics).
        // Walks every dim whose [i*bits, (i+1)*bits) range intersects
        // [32d, 32(d+1)), including a single spill from the previous
        // word for non-32-aligned bit-widths (3-bit, 5-bit, etc.).
        if (d < PackedWidth) {
            uint w_val = 0u;
            int word_start = (int)d * 32;
            int i_min = word_start / 4;
            int i_max = (word_start + 31) / 4;
            if (i_max >= (int)Dim) i_max = (int)Dim - 1;
            for (int i = i_min; i <= i_max; i++) {
                uint idx_val = shared_idx[i] & 15u;
                int bit_off = i * 4 - word_start;
                if (bit_off >= 0) {
                    w_val |= idx_val << bit_off;
                } else {
                    w_val |= idx_val >> (-bit_off);
                }
            }
            out[bh * PackedWidth + d] = w_val;
        }
    
}

template [[host_name("custom_kernel_turboquant_norot_quantize_4__64_2_float16_t_float_uint32_t")]] [[kernel]] decltype(custom_kernel_turboquant_norot_quantize_4__64_2_float16_t_float_uint32_t<64, 2>) custom_kernel_turboquant_norot_quantize_4__64_2_float16_t_float_uint32_t<64, 2>;
