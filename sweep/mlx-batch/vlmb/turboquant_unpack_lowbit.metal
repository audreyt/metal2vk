// Origin: mlx-vlm@8f5dc3d, mlx_vlm/turboquant.py:237.
// Representative call: packed uint32 (4,2); Bits=4,Length=8,PackedWidth=2; grid=(8,4,1), threadgroup=(32,1,1); out (4,8) uint32.
template <int Bits, int Length, int PackedWidth>
[[kernel]] void custom_kernel_turboquant_unpack_lowbit__4_8_2_uint32_t_uint32_t(
  const device uint32_t* packed [[buffer(0)]],
  const constant int* packed_shape [[buffer(1)]],
  device uint32_t* out [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

        auto idx = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= packed_shape[0] || idx >= Length) {
            return;
        }

        auto packed_ptr = packed + row * PackedWidth;
        int bit_offset = idx * Bits;
        int word_idx = bit_offset / 32;
        int offset = bit_offset % 32;
        uint value = packed_ptr[word_idx] >> offset;
        int spill = offset + Bits - 32;
        if (spill > 0) {
            value |= packed_ptr[word_idx + 1] << (Bits - spill);
        }
        out[row * Length + idx] = value & ((1u << Bits) - 1u);
    
}

template [[host_name("custom_kernel_turboquant_unpack_lowbit__4_8_2_uint32_t_uint32_t")]] [[kernel]] decltype(custom_kernel_turboquant_unpack_lowbit__4_8_2_uint32_t_uint32_t<4, 8, 2>) custom_kernel_turboquant_unpack_lowbit__4_8_2_uint32_t_uint32_t<4, 8, 2>;
