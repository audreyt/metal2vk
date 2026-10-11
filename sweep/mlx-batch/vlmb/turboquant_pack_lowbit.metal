// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:205.
// Representative call: values uint32 (4,8); Bits=4,Length=8,PackedWidth=2; grid=(2,4,1), threadgroup=(2,1,1); out (4,2) uint32.
template <int Bits, int Length, int PackedWidth>
[[kernel]] void custom_kernel_turboquant_pack_lowbit__4_8_2_uint32_t_uint32_t(
  const device uint32_t* values [[buffer(0)]],
  const constant int* values_shape [[buffer(1)]],
  device uint32_t* out [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

        auto word = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= values_shape[0] || word >= PackedWidth) {
            return;
        }

        auto values_ptr = values + row * Length;
        uint packed_word = 0u;
        int start = max(0, (int(word) * 32 - (Bits - 1)) / Bits);
        int end = min(Length, ((int(word) + 1) * 32 + (Bits - 1)) / Bits);

        for (int idx = start; idx < end; ++idx) {
            int bit_offset = idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = values_ptr[idx] & ((1u << Bits) - 1u);
            if (word_idx == word) {
                packed_word |= value << offset;
            }
            if (word_idx + 1 == word) {
                int spill = offset + Bits - 32;
                if (spill > 0) {
                    packed_word |= value >> (Bits - spill);
                }
            }
        }

        out[row * PackedWidth + word] = packed_word;
    
}

template [[host_name("custom_kernel_turboquant_pack_lowbit__4_8_2_uint32_t_uint32_t")]] [[kernel]] decltype(custom_kernel_turboquant_pack_lowbit__4_8_2_uint32_t_uint32_t<4, 8, 2>) custom_kernel_turboquant_pack_lowbit__4_8_2_uint32_t_uint32_t<4, 8, 2>;
