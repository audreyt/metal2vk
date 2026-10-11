// Origin: mlx-vlm@8f5dc3d, mlx-vlm/mlx_vlm/models/fast_ops.py:735.
// Representative call: the caller hard-requires x.shape[2:] == (4, 4096) and mix weight (24, 16384); fp16 per the same dtype gate; grid=(1024, 1, 1), threadgroup=(1024, 1, 1).
template <typename T, typename W, int HC, int D, int ITERS, int HC_EPS_INT, int NORM_EPS_INT>
[[kernel]] void custom_kernel_short_block_hc_normalized_norm_fp16_fp16_h4_d4096_i2_he1_ne1000__float16_t_float16_t_4_4096_2_1_1000_float16_t_float16_t_floatc_float_float16_t_float16_t_float_float(
  const device float16_t* x_in [[buffer(0)]],
  const device float16_t* mix_weight [[buffer(1)]],
  const constant float* scale [[buffer(2)]],
  const device float* base [[buffer(3)]],
  const device float16_t* norm_weight [[buffer(4)]],
  device float16_t* normalized [[buffer(5)]],
  device float* post [[buffer(6)]],
  device float* comb [[buffer(7)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

uint row = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;

constexpr int MIX = (2 + HC) * HC;
constexpr int BASE_OFF = 2 * HC;
constexpr int K_SIZE = HC * D;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int MIX_SIMDS = 8;
constexpr int MIX_BLOCK = MIX_SIMDS * 32 * TN;
constexpr int MIX_BLOCKS = (MIX + TM - 1) / TM;
constexpr float HC_EPS = HC_EPS_INT * 1e-9;
constexpr float NORM_EPS = NORM_EPS_INT * 1e-9;
constexpr int D4 = D / 4;

const device T* x_row = x_in + row * K_SIZE;

// Match the standalone FP32 RMS reduction: 32 SIMD groups, four values per
// lane, then a fixed SIMD-group reduction over the 32 partials.
threadgroup float norm_partials[32];
threadgroup float inv_norm[1];
float norm_sum = 0.0f;
for (int k = int(tid) * TN; k < K_SIZE; k += 32 * 32 * TN) {
  float4 value = float4(*(const device vec<T, 4>*)(x_row + k));
  norm_sum += value.x * value.x;
  norm_sum += value.y * value.y;
  norm_sum += value.z * value.z;
  norm_sum += value.w * value.w;
}
norm_sum = simd_sum(norm_sum);
if (lane == 0) {
  norm_partials[sg] = norm_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
  norm_sum = simd_sum(norm_partials[lane]);
  if (lane == 0) {
    inv_norm[0] = metal::precise::rsqrt(norm_sum / K_SIZE + NORM_EPS);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Six four-output GEMVs use the same eight-SIMD reduction tree as the
// standalone decode-exact mix kernel. Four blocks run in phase zero and the
// remaining two in phase one, sharing this threadgroup and the normalized x.
threadgroup float mix_partials[4][MIX_SIMDS][TM];
threadgroup float mix_values[MIX];
uint mix_slot = sg / MIX_SIMDS;
uint mix_sg = sg % MIX_SIMDS;
for (int phase = 0; phase < 2; ++phase) {
  int out_block = phase * 4 + int(mix_slot);
  if (out_block < MIX_BLOCKS) {
    int out_row = out_block * TM;
    int bn = (int(mix_sg) * 32 + int(lane)) * TN;
    float result[TM] = {0.0f};
    for (int k = 0; k < K_SIZE; k += MIX_BLOCK) {
      float vectors[TN];
      float inv = inv_norm[0];
      const device T* source_x = x_row + k + bn;
      for (int tn = 0; tn < TN; ++tn) {
        vectors[tn] = float(source_x[tn]) * inv;
      }
      for (int tm = 0; tm < TM; ++tm) {
        const device W* source_w =
            mix_weight + (out_row + tm) * K_SIZE + k + bn;
        for (int tn = 0; tn < TN; ++tn) {
          result[tm] += source_w[tn] * vectors[tn];
        }
      }
    }
    for (int tm = 0; tm < TM; ++tm) {
      for (ushort offset = 16; offset >= 1; offset >>= 1) {
        result[tm] += simd_shuffle_down(result[tm], offset);
      }
      if (lane == 0) {
        mix_partials[mix_slot][mix_sg][tm] = result[tm];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (mix_sg == 0 && lane == 0 && out_block < MIX_BLOCKS) {
    int out_row = out_block * TM;
    for (int tm = 0; tm < TM; ++tm) {
      float value = mix_partials[mix_slot][0][tm];
      for (int group = 1; group < MIX_SIMDS; ++group) {
        value += mix_partials[mix_slot][group][tm];
      }
      if (out_row + tm < MIX) {
        mix_values[out_row + tm] = value;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

device float* post_out = post + row * HC;
device float* comb_out = comb + row * HC * HC;
threadgroup float pre_shared[HC];
threadgroup float collapsed_sums[32];
threadgroup float collapsed_inv[1];

if (sg == 0) {
  const float pre_scale = scale[0];
  const float post_scale = scale[1];
  const float comb_scale = scale[2];
  const float active = lane < (uint)HC ? 1.0f : 0.0f;
  const uint llane = metal::min(lane, (uint)(HC - 1));

  float pre_z = mix_values[llane] * pre_scale + base[llane];
  float post_z = mix_values[HC + llane] * post_scale + base[HC + llane];
  float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + HC_EPS;
  float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
  if (lane < (uint)HC) {
    pre_shared[lane] = pre_v;
    post_out[lane] = post_v;
  }

  float4 value =
      (*(threadgroup float4*)(mix_values + BASE_OFF + llane * HC) *
           comb_scale +
       *(const device float4*)(base + BASE_OFF + llane * HC)) * active;
  float row_max = metal::max(
      metal::max(value.x, value.y), metal::max(value.z, value.w));
  float4 exponent = metal::fast::exp(value - row_max) * active;
  float4 result = exponent *
          (1.0f /
           (exponent.x + exponent.y + exponent.z + exponent.w + HC_EPS)) +
      HC_EPS * active;
  float4 column_inv = 1.0f /
      (float4(
           simd_sum(result.x), simd_sum(result.y),
           simd_sum(result.z), simd_sum(result.w)) +
       HC_EPS);
  result *= column_inv;
  for (int iter = 1; iter < ITERS; ++iter) {
    result *=
        (1.0f / (result.x + result.y + result.z + result.w + HC_EPS)) *
        active;
    column_inv = 1.0f /
        (float4(
             simd_sum(result.x), simd_sum(result.y),
             simd_sum(result.z), simd_sum(result.w)) +
         HC_EPS);
    result *= column_inv;
  }
  if (lane < (uint)HC) {
    *(device float4*)(comb_out + lane * HC) = result;
  }
  collapsed_sums[lane] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

using T4 = vec<T, 4>;
const device T4* x0 = (const device T4*)(x_row);
const device T4* x1 = (const device T4*)(x_row + D);
const device T4* x2 = (const device T4*)(x_row + 2 * D);
const device T4* x3 = (const device T4*)(x_row + 3 * D);
T4 rounded = T4(0);
float collapsed_sum = 0.0f;
if (tid < D4) {
  float4 collapsed = fma(
      float4(pre_shared[0]), float4(x0[tid]),
      fma(
          float4(pre_shared[1]), float4(x1[tid]),
          fma(
              float4(pre_shared[2]), float4(x2[tid]),
              float4(pre_shared[3]) * float4(x3[tid]))));
  rounded = T4(collapsed);
  float4 rounded_float = float4(rounded);
  collapsed_sum += rounded_float.x * rounded_float.x;
  collapsed_sum += rounded_float.y * rounded_float.y;
  collapsed_sum += rounded_float.z * rounded_float.z;
  collapsed_sum += rounded_float.w * rounded_float.w;
}
collapsed_sum = simd_sum(collapsed_sum);
if (lane == 0) {
  collapsed_sums[sg] = collapsed_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
  collapsed_sum = simd_sum(collapsed_sums[lane]);
  if (lane == 0) {
    collapsed_inv[0] =
        metal::precise::rsqrt(collapsed_sum / D + NORM_EPS);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid < D4) {
  float inv = collapsed_inv[0];
  const device T4* weights = (const device T4*)norm_weight;
  T4 scaled = T4(float4(rounded) * inv);
  T4 weight = weights[tid];
  ((device T4*)normalized)[row * D4 + tid] = T4(
      weight.x * scaled.x,
      weight.y * scaled.y,
      weight.z * scaled.z,
      weight.w * scaled.w);
}

}

template [[host_name("custom_kernel_short_block_hc_normalized_norm_fp16_fp16_h4_d4096_i2_he1_ne1000__float16_t_float16_t_4_4096_2_1_1000_float16_t_float16_t_floatc_float_float16_t_float16_t_float_float")]] [[kernel]] decltype(custom_kernel_short_block_hc_normalized_norm_fp16_fp16_h4_d4096_i2_he1_ne1000__float16_t_float16_t_4_4096_2_1_1000_float16_t_float16_t_floatc_float_float16_t_float16_t_float_float<float16_t, float16_t, 4, 4096, 2, 1, 1000>) custom_kernel_short_block_hc_normalized_norm_fp16_fp16_h4_d4096_i2_he1_ne1000__float16_t_float16_t_4_4096_2_1_1000_float16_t_float16_t_floatc_float_float16_t_float16_t_float_float<float16_t, float16_t, 4, 4096, 2, 1, 1000>;
