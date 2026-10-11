// Origin: mlx-vlm@8f5dc3d, mlx_vlm/structured.py:7 (production callsite).
// Representative call: logits float32 (1,16), mask int32 (1,1), output float32 (1,16).
// Template: T=float; grid=(16,1,1); threadgroup=(256,1,1).
template <typename T>
[[kernel]] void custom_kernel_mlx_vlm_llguidance_mask__float_float_int32_tc_float(
  const device float* logits [[buffer(0)]],
  const constant int* logits_shape [[buffer(1)]],
  const constant int32_t* mask [[buffer(2)]],
  const constant int* mask_shape [[buffer(3)]],
  device float* out [[buffer(4)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
        uint batch = thread_position_in_grid.y;
        uint token = thread_position_in_grid.x;
        uint word = token >> 5;
        uint bit = token & 31;
        bool allowed = word < mask_shape[1] &&
            ((as_type<uint>(mask[batch * mask_shape[1] + word]) >> bit) & 1u);
        uint offset = batch * logits_shape[1] + token;
        out[offset] = allowed ? logits[offset] : -metal::numeric_limits<T>::infinity();
}

template [[host_name("custom_kernel_mlx_vlm_llguidance_mask__float_float_int32_tc_float")]] [[kernel]] decltype(custom_kernel_mlx_vlm_llguidance_mask__float_float_int32_tc_float<float>) custom_kernel_mlx_vlm_llguidance_mask__float_float_int32_tc_float<float>;
