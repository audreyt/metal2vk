// Origin: mlx-vlm@8f5dc3d, mlx-vlm/mlx_vlm/models/fast_ops.py:707.
// Representative call: fp16 x (1,2,4,8), hc_mult 4, width 8, 2 sinkhorn iters, hc_eps 1e-09, norm_eps 1e-06; fp16 because the caller gates dtype to bfloat16/float16 (float16 chosen over bfloat16); grid=(64, 1, 1), threadgroup=(32, 1, 1).
template <typename T, int HC, int D, int ITERS, int HC_EPS_INT, int NORM_EPS_INT>
[[kernel]] void custom_kernel_short_block_hc_norm_fp16_h4_d8_i2_he1_ne1000__float16_t_4_8_2_1_1000_float16_t_float_floatc_float_float16_t_float16_t_float_float(
  const device float16_t* x_in [[buffer(0)]],
  const device float* mixes [[buffer(1)]],
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
constexpr float HC_EPS = HC_EPS_INT * 1e-9;
constexpr float NORM_EPS = NORM_EPS_INT * 1e-9;
constexpr int D4 = D / 4;

const device float* mix = mixes + row * MIX;
device float* post_out = post + row * HC;
device float* comb_out = comb + row * HC * HC;
threadgroup float pre_shared[HC];
threadgroup float local_inv_mean[1];
threadgroup float local_sums[32];

if (sg == 0) {
  const float pre_scale = scale[0];
  const float post_scale = scale[1];
  const float comb_scale = scale[2];
  const float active = lane < (uint)HC ? 1.0f : 0.0f;
  const uint llane = metal::min(lane, (uint)(HC - 1));

  float pre_z = mix[llane] * pre_scale + base[llane];
  float post_z = mix[HC + llane] * post_scale + base[HC + llane];
  float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + HC_EPS;
  float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
  if (lane < (uint)HC) {
    pre_shared[lane] = pre_v;
    post_out[lane] = post_v;
  }

  float4 value =
      (*(const device float4*)(mix + BASE_OFF + llane * HC) * comb_scale +
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
}

if (sg == 0) {
  local_sums[lane] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const device T* x_row = (const device T*)x_in + row * HC * D;
using T4 = vec<T, 4>;
const device T4* x0 = (const device T4*)(x_row);
const device T4* x1 = (const device T4*)(x_row + D);
const device T4* x2 = (const device T4*)(x_row + 2 * D);
const device T4* x3 = (const device T4*)(x_row + 3 * D);

T4 rounded = T4(0);
float accum = 0.0f;
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
  accum += rounded_float.x * rounded_float.x;
  accum += rounded_float.y * rounded_float.y;
  accum += rounded_float.z * rounded_float.z;
  accum += rounded_float.w * rounded_float.w;
}

accum = simd_sum(accum);
if (lane == 0) {
  local_sums[sg] = accum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
  accum = simd_sum(local_sums[lane]);
  if (lane == 0) {
    local_inv_mean[0] = metal::precise::rsqrt(accum / D + NORM_EPS);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid < D4) {
  float inv = local_inv_mean[0];
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

template [[host_name("custom_kernel_short_block_hc_norm_fp16_h4_d8_i2_he1_ne1000__float16_t_4_8_2_1_1000_float16_t_float_floatc_float_float16_t_float16_t_float_float")]] [[kernel]] decltype(custom_kernel_short_block_hc_norm_fp16_h4_d8_i2_he1_ne1000__float16_t_4_8_2_1_1000_float16_t_float_floatc_float_float16_t_float16_t_float_float<float16_t, 4, 8, 2, 1, 1000>) custom_kernel_short_block_hc_norm_fp16_h4_d8_i2_he1_ne1000__float16_t_4_8_2_1_1000_float16_t_float_floatc_float_float16_t_float16_t_float_float<float16_t, 4, 8, 2, 1, 1000>;
