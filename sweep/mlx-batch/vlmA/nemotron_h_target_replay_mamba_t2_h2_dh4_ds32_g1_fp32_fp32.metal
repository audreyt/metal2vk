// Origin: mlx-vlm@8f5dc3d, mlx-vlm/mlx_vlm/models/nemotron_h/speculative_verifier.py:155.
// Representative call: float32, same shapes as the verify kernel without the C/D output path; grid=(32, 4, 2), threadgroup=(32, 4, 1).
template <typename T, typename U, int VERIFY_T, int H, int DH, int DS, int GROUPS, int HEADS_PER_GROUP>
[[kernel]] void custom_kernel_nemotron_h_target_replay_mamba_t2_h2_dh4_ds32_g1_fp32_fp32__float_float_2_2_4_32_1_2_float_floatc_float_floatc_float_float(
  const device float* X [[buffer(0)]],
  const constant float* A_log [[buffer(1)]],
  const device float* B [[buffer(2)]],
  const constant float* dt [[buffer(3)]],
  const device float* state_in [[buffer(4)]],
  device float* state_out [[buffer(5)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

    uint lane = thread_position_in_grid.x;
    uint d_idx = thread_position_in_grid.y;
    uint n = thread_position_in_grid.z;
    uint batch_idx = n / H;
    uint head_idx = n % H;
    uint group_idx = head_idx / HEADS_PER_GROUP;
    uint state_idx = ((batch_idx * H + head_idx) * DH + d_idx) * DS;
    constexpr int N_PER_THREAD = DS / 32;

    U local_state[N_PER_THREAD];
    for (int i = 0; i < N_PER_THREAD; ++i) {
        uint s_idx = N_PER_THREAD * lane + i;
        local_state[i] = state_in[state_idx + s_idx];
    }

    float A = -fast::exp(static_cast<float>(A_log[head_idx]));
    for (int token = 0; token < VERIFY_T; ++token) {
        uint x_idx = ((batch_idx * VERIFY_T + token) * H + head_idx) * DH + d_idx;
        uint b_idx = ((batch_idx * VERIFY_T + token) * GROUPS + group_idx) * DS;
        uint dt_idx = (batch_idx * VERIFY_T + token) * H + head_idx;
        float dt_value = static_cast<float>(dt[dt_idx]);
        float decay = fast::exp(A * dt_value);
        float x_value = static_cast<float>(X[x_idx]);

        for (int i = 0; i < N_PER_THREAD; ++i) {
            uint s_idx = N_PER_THREAD * lane + i;
            float next_state =
                decay * static_cast<float>(local_state[i]) +
                x_value * dt_value * static_cast<float>(B[b_idx + s_idx]);
            local_state[i] = static_cast<U>(next_state);
        }
    }

    for (int i = 0; i < N_PER_THREAD; ++i) {
        uint s_idx = N_PER_THREAD * lane + i;
        state_out[state_idx + s_idx] = local_state[i];
    }

}

template [[host_name("custom_kernel_nemotron_h_target_replay_mamba_t2_h2_dh4_ds32_g1_fp32_fp32__float_float_2_2_4_32_1_2_float_floatc_float_floatc_float_float")]] [[kernel]] decltype(custom_kernel_nemotron_h_target_replay_mamba_t2_h2_dh4_ds32_g1_fp32_fp32__float_float_2_2_4_32_1_2_float_floatc_float_floatc_float_float<float, float, 2, 2, 4, 32, 1, 2>) custom_kernel_nemotron_h_target_replay_mamba_t2_h2_dh4_ds32_g1_fp32_fp32__float_float_2_2_4_32_1_2_float_floatc_float_floatc_float_float<float, float, 2, 2, 4, 32, 1, 2>;
