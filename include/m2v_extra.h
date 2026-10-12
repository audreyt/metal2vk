// m2v_extra.h: Metal features the bulk sweep (sweep/results/FEATURES.md) found missing. Included at the end of metal_stdlib.
#pragma once

// OpenCL C++ reserves `local` and `global` as address-space keywords, while Metal code uses them as variable names (uzu's
// threadgroup_reduce.h does). The macros apply to every file included after the shim, so C++ for OpenCL code that spells the address
// spaces without the leading underscores does not compile through this shim; Metal never spells them that way.
#define local m2v_local
#define global m2v_global

namespace metal {

// ---- barriers: mem_flags combine with | ----
constexpr mem_flags operator|(mem_flags a, mem_flags b) { return (mem_flags)((int)a | (int)b); }
constexpr mem_flags operator&(mem_flags a, mem_flags b) { return (mem_flags)((int)a & (int)b); }

// ---- thread fences with a scope (Metal 3.2) ----
enum thread_scope { thread_scope_thread = 0, thread_scope_simdgroup = 1, thread_scope_threadgroup = 2, thread_scope_device = 3 };
METAL_FUNC void atomic_thread_fence(mem_flags f, memory_order, thread_scope = thread_scope_device) {
  barrier(((int)f & mem_device) ? CLK_GLOBAL_MEM_FENCE : CLK_LOCAL_MEM_FENCE);
}

// ---- integer bit functions (Metal integer functions with no OpenCL spelling) ----
template <typename T> METAL_FUNC T extract_bits(T v, uint offset, uint bits) {
  if (bits == 0u) return T(0);
  const uint w = sizeof(T) * 8u;
  const T shifted = (T)(v >> offset);
  if (__is_signed(T)) {  // arithmetic shift sign-extends from the top of the field
    const uint s = w - bits;
    return (T)((T)(shifted << s) >> s);
  }
  return bits >= w ? shifted : (T)(shifted & (T)(((T)1 << bits) - (T)1));
}
template <typename T> METAL_FUNC T insert_bits(T base, T insert, uint offset, uint bits) {
  if (bits == 0u) return base;
  const T mask = bits >= sizeof(T) * 8u ? (T)~(T)0 : (T)(((T)1 << bits) - (T)1);
  return (T)((base & (T)~(mask << offset)) | ((insert & mask) << offset));
}
METAL_FUNC uint reverse_bits(uint x) {
  x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
  x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
  x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
  x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
  return (x >> 16) | (x << 16);
}

// ---- scalar math ----
template <typename T> METAL_FUNC T sign(T x) { return ::sign(x); }
namespace precise {
METAL_FUNC float divide(float a, float b) { return a / b; }
}
namespace fast {
METAL_FUNC float divide(float a, float b) { return native_divide(a, b); }
}
template <> struct numeric_limits<bool> { static constexpr bool max() { return true; } static constexpr bool min() { return false; } };

// ---- subgroup shuffles up/down: lowered to the plain subgroup shuffle (clspv has no OpGroupNonUniformShuffleDown/Up lowering) ----
// Out-of-range lanes keep their own value (Metal leaves them undefined).
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_down(T x, ushort d) {
  const uint l = get_sub_group_local_id();
  return sub_group_shuffle(x, l + d < (uint)get_sub_group_size() ? l + d : l);
}
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_up(T x, ushort d) {
  const uint l = get_sub_group_local_id();
  return sub_group_shuffle(x, l >= d ? l - d : l);
}
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_down(V x, ushort d) {
  V r;
  for (int i = 0; i < (int)__builtin_vectorelements(V); i++) r[i] = simd_shuffle_down(x[i], d);
  return r;
}
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_up(V x, ushort d) {
  V r;
  for (int i = 0; i < (int)__builtin_vectorelements(V); i++) r[i] = simd_shuffle_up(x[i], d);
  return r;
}

// ---- simd reductions and scans for every arithmetic type, bfloat, and vectors ----
// M2V_MLX_TREE (mlx-core sweep): the TU carries MLX's own bf16_math.h, which defines these bfloat
// overloads itself - suppress the shim's to keep one definition.
#ifdef M2V_MLX_TREE
#define M2V_SIMD_RED_BF(NAME, BUILTIN)
#else
#define M2V_SIMD_RED_BF(NAME, BUILTIN) METAL_FUNC bfloat NAME(bfloat x) { return bfloat(BUILTIN((float)x)); }
#endif
#define M2V_SIMD_RED(NAME, BUILTIN)                                                                                    \
  template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> NAME(T x) {                                      \
    static_assert(sizeof(T) <= 4, "simd reductions and scans on 64-bit types are not supported: the value would be truncated to 32 bits"); \
    using W = conditional_t<__is_integral(T), conditional_t<__is_signed(T), int, uint>, float>;                        \
    return (T)BUILTIN((W)x);                                                                                            \
  }                                                                                                                    \
  M2V_SIMD_RED_BF(NAME, BUILTIN)                                                                                       \
  template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V) && !is_same_v<V, bfloat>, V> NAME(V x) {            \
    V r;                                                                                                               \
    for (int i = 0; i < (int)__builtin_vectorelements(V); i++) r[i] = NAME(x[i]);                                      \
    return r;                                                                                                          \
  }
M2V_SIMD_RED(simd_sum, sub_group_reduce_add)
M2V_SIMD_RED(simd_max, sub_group_reduce_max)
M2V_SIMD_RED(simd_min, sub_group_reduce_min)
M2V_SIMD_RED(simd_prefix_inclusive_sum, sub_group_scan_inclusive_add)
M2V_SIMD_RED(simd_prefix_exclusive_sum, sub_group_scan_exclusive_add)

// ---- quad (4-lane) operations ----
template <typename T> METAL_FUNC T quad_sum(T x) {
  T y = x + simd_shuffle_xor(x, (ushort)1);
  return y + simd_shuffle_xor(y, (ushort)2);
}
template <typename T> METAL_FUNC T quad_max(T x) {
  T y = max(x, simd_shuffle_xor(x, (ushort)1));
  return max(y, simd_shuffle_xor(y, (ushort)2));
}
template <typename T> METAL_FUNC T quad_min(T x) {
  T y = min(x, simd_shuffle_xor(x, (ushort)1));
  return min(y, simd_shuffle_xor(y, (ushort)2));
}

// ---- simdgroup_load / simdgroup_store (Metal 3): the Apple 8x8 fragment layout over any device or threadgroup pointer ----
template <typename T, typename P> METAL_FUNC void simdgroup_load(simdgroup_matrix<T, 8, 8>& dst, P src, size_t ld) {
  const uint lane = get_sub_group_local_id();
  const int r = m2v_row(lane), c = m2v_col(lane);
  dst.el[0] = T(src[(size_t)r * ld + c]);
  dst.el[1] = T(src[(size_t)r * ld + c + 1]);
}
// origin is (column, row) offsets in elements; transpose loads the transposed tile (dst(r, c) = src(c, r)).
template <typename T, typename P>
METAL_FUNC void simdgroup_load(simdgroup_matrix<T, 8, 8>& dst, P src, size_t ld, ulong2 origin, bool transpose) {
  const uint lane = get_sub_group_local_id();
  const int r = m2v_row(lane), c = m2v_col(lane);
  if (transpose) {
    dst.el[0] = T(src[(origin.y + c) * ld + origin.x + r]);
    dst.el[1] = T(src[(origin.y + c + 1) * ld + origin.x + r]);
  } else {
    dst.el[0] = T(src[(origin.y + r) * ld + origin.x + c]);
    dst.el[1] = T(src[(origin.y + r) * ld + origin.x + c + 1]);
  }
}
template <typename T, typename P> METAL_FUNC void simdgroup_store(const simdgroup_matrix<T, 8, 8>& src, P dst, size_t ld) {
  const uint lane = get_sub_group_local_id();
  const int r = m2v_row(lane), c = m2v_col(lane);
  dst[(size_t)r * ld + c] = src.el[0];
  dst[(size_t)r * ld + c + 1] = src.el[1];
}
template <typename T, typename P>
METAL_FUNC void simdgroup_store(const simdgroup_matrix<T, 8, 8>& src, P dst, size_t ld, ulong2 origin, bool transpose) {
  const uint lane = get_sub_group_local_id();
  const int r = m2v_row(lane), c = m2v_col(lane);
  if (transpose) {
    dst[(origin.y + c) * ld + origin.x + r] = src.el[0];
    dst[(origin.y + c + 1) * ld + origin.x + r] = src.el[1];
  } else {
    dst[(origin.y + r) * ld + origin.x + c] = src.el[0];
    dst[(origin.y + r) * ld + origin.x + c + 1] = src.el[1];
  }
}

// ---- fast:: and precise:: math on vectors ----
#define M2V_FAST_VEC(NAME, SCALAR)                                                                                     \
  namespace fast {                                                                                                     \
  template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> NAME(V x) {                                     \
    V r;                                                                                                               \
    for (int i = 0; i < (int)__builtin_vectorelements(V); i++) r[i] = SCALAR(x[i]);                                    \
    return r;                                                                                                          \
  }                                                                                                                    \
  }
M2V_FAST_VEC(exp, native_exp)
M2V_FAST_VEC(exp2, native_exp2)
M2V_FAST_VEC(log, native_log)
M2V_FAST_VEC(log2, native_log2)
M2V_FAST_VEC(sqrt, native_sqrt)
M2V_FAST_VEC(rsqrt, native_rsqrt)
M2V_FAST_VEC(tanh, ::tanh)
M2V_FAST_VEC(sin, native_sin)
M2V_FAST_VEC(cos, native_cos)

#ifdef M2V_BFLOAT_STRUCT
// ---- bfloat vectors (storage + conversion; arithmetic goes through float) ----
// Fields are raw i16 at the IR level (see the vec_sel<bfloat, N> note in metal_stdlib); the element interface keeps
// bfloat semantics with round-to-nearest-even narrowing on every write.
#define M2V_BFVEC_COMMON(NAME, N)                                                                                      \
  static constexpr int m2v_n = N;                                                                                      \
  METAL_FUNC NAME() = default;                                                                                         \
  METAL_FUNC bfloat operator[](int i) const { bfloat r; r.bits = (&x)[i]; return r; }                                  \
  struct ref {                                                                                                         \
    ushort* slot;                                                                                                      \
    ushort bits;                                                                                                       \
    METAL_FUNC ref(ushort* s, ushort b) : slot(s), bits(b) {}                                                          \
    METAL_FUNC ref(float f) : slot(nullptr), bits(bfloat(f).bits) {}                                                   \
    METAL_FUNC operator bfloat() const { bfloat r; r.bits = slot ? *slot : bits; return r; }                           \
    METAL_FUNC operator float() const { return as_float((uint)(slot ? *slot : bits) << 16); }                          \
    METAL_FUNC ref& operator=(bfloat b) { if (slot) *slot = b.bits; else bits = b.bits; return *this; }                \
    METAL_FUNC ref& operator=(float f) { return *this = bfloat(f); }                                                   \
  };                                                                                                                   \
  METAL_FUNC ref operator[](int i) { return ref{&x + i, 0}; }                                                             \
  template <typename E> METAL_FUNC NAME(E v) { for (int i = 0; i < N; i++) (&x)[i] = bfloat((float)v[i]).bits; }
struct bfloat2 { ushort x, y;
  M2V_BFVEC_COMMON(bfloat2, 2)
  METAL_FUNC bfloat2(bfloat a, bfloat b) : x(a.bits), y(b.bits) {} };
struct bfloat3 { ushort x, y, z;
  M2V_BFVEC_COMMON(bfloat3, 3)
  METAL_FUNC bfloat3(bfloat a, bfloat b, bfloat c) : x(a.bits), y(b.bits), z(c.bits) {} };
struct bfloat4 { ushort x, y, z, w;
  M2V_BFVEC_COMMON(bfloat4, 4)
  METAL_FUNC bfloat4(bfloat a, bfloat b, bfloat c, bfloat d) : x(a.bits), y(b.bits), z(c.bits), w(d.bits) {} };

// ---- metal::array ----
template <typename T, size_t N> struct array {
  T _data[N];
  METAL_FUNC constexpr size_t size() const { return N; }
  METAL_FUNC T& operator[](size_t i) { return _data[i]; }
  METAL_FUNC const T& operator[](size_t i) const { return _data[i]; }
  METAL_FUNC T* data() { return _data; }
};

// ---- Apple built-ins that standard-library headers wrap (glm_glue.metal re-declares bfloat math through them) ----
#define __METAL_MAYBE_FAST_MATH__ 0
#define __METAL_FAST_MATH__ 0
#define __METAL_PRECISE_MATH__ 0
#define M2V_BUILTIN1(N, F) template <typename T> METAL_FUNC T __metal_##N(T x, int) { return F(x); }
M2V_BUILTIN1(fabs, ::fabs) M2V_BUILTIN1(exp, ::exp) M2V_BUILTIN1(exp2, ::exp2) M2V_BUILTIN1(log, ::log) M2V_BUILTIN1(log2, ::log2)
M2V_BUILTIN1(sqrt, ::sqrt) M2V_BUILTIN1(rsqrt, ::rsqrt) M2V_BUILTIN1(tanh, ::tanh) M2V_BUILTIN1(sin, ::sin) M2V_BUILTIN1(cos, ::cos)

// ---- The rest of Apple's internals MLX's bf16_math.h and steel nax.h call: the trailing int is Metal's
// fast-math flag (one precision here) and arity follows the internal spellings. The simd forms the OpenCL
// headers lack are butterflies / Hillis-Steele over sub_group_shuffle, exact for a power-of-two simdgroup.
#define M2V_BUILTIN2(N, F) template <typename T> METAL_FUNC T __metal_##N(T x, T y, int) { return F(x, y); }
M2V_BUILTIN1(acos, ::acos) M2V_BUILTIN1(acosh, ::acosh) M2V_BUILTIN1(asin, ::asin) M2V_BUILTIN1(asinh, ::asinh)
M2V_BUILTIN1(atan, ::atan) M2V_BUILTIN1(atanh, ::atanh) M2V_BUILTIN1(ceil, ::ceil) M2V_BUILTIN1(cosh, ::cosh) M2V_BUILTIN1(sinh, ::sinh) M2V_BUILTIN1(tan, ::tan)
M2V_BUILTIN1(exp10, ::exp10) M2V_BUILTIN1(floor, ::floor) M2V_BUILTIN1(log10, ::log10) M2V_BUILTIN1(rint, ::rint)
M2V_BUILTIN1(round, ::round) M2V_BUILTIN1(trunc, ::trunc) M2V_BUILTIN1(cospi, ::cospi) M2V_BUILTIN1(sinpi, ::sinpi)
M2V_BUILTIN1(tanpi, ::tanpi) M2V_BUILTIN2(atan2, ::atan2) M2V_BUILTIN2(fmax, ::fmax) M2V_BUILTIN2(fmin, ::fmin)
M2V_BUILTIN2(fmod, ::fmod) M2V_BUILTIN2(pow, ::pow) M2V_BUILTIN2(powr, ::powr)
template <typename T> METAL_FUNC T __metal_fract(T x, int) { T ip; return ::fract(x, &ip); }
template <typename T> METAL_FUNC T __metal_fma(T x, T y, T z) { return ::fma(x, y, z); }
template <typename T> METAL_FUNC T __metal_nextafter(T x, T y) { return ::nextafter(x, y); }
template <typename T> METAL_FUNC T __metal_divide(T x, T y, int) { return x / y; }
template <typename T> METAL_FUNC T __metal_ldexp(T x, int k, int) { return ::ldexp(x, k); }
template <typename T, typename E> METAL_FUNC T __metal_frexp(T x, E* e) { return ::frexp(x, e); }
template <typename T> METAL_FUNC T __metal_fmax3(T x, T y, T z, int) { return ::fmax(::fmax(x, y), z); }
template <typename T> METAL_FUNC T __metal_fmin3(T x, T y, T z, int) { return ::fmin(::fmin(x, y), z); }
template <typename T> METAL_FUNC T __metal_fmedian3(T x, T y, T z, int) { return __metal_fmax(__metal_fmin(x, y, 0), __metal_fmin(__metal_fmax(x, y, 0), z, 0), 0); }
METAL_FUNC uint __metal_get_thread_index_in_simdgroup() { return get_sub_group_local_id(); }
METAL_FUNC uint __metal_get_simdgroup_size(ushort) { return get_sub_group_size(); }
template <typename T> METAL_FUNC T __metal_simd_broadcast(T x, ushort l) { return simd_broadcast(x, l); }
template <typename T> METAL_FUNC T __metal_simd_shuffle(T x, ushort l) { return simd_shuffle(x, l); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_xor(T x, ushort m) { return simd_shuffle_xor(x, m); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_up(T x, ushort d) { return simd_shuffle_up(x, d); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_down(T x, ushort d) { return simd_shuffle_down(x, d); }
template <typename T> METAL_FUNC T __metal_simd_sum(T x) { return simd_sum(x); }
template <typename T> METAL_FUNC T __metal_simd_max(T x) { return simd_max(x); }
template <typename T> METAL_FUNC T __metal_simd_min(T x) { return simd_min(x); }
template <typename T> METAL_FUNC T __metal_simd_prefix_inclusive_sum(T x) { return simd_prefix_inclusive_sum(x); }
template <typename T> METAL_FUNC T __metal_simd_prefix_exclusive_sum(T x) { return simd_prefix_exclusive_sum(x); }
template <typename T> METAL_FUNC T __metal_simd_product(T x) { for (uint o = (uint)get_sub_group_size() >> 1; o; o >>= 1) x *= sub_group_shuffle_xor(x, o); return x; }
template <typename T> METAL_FUNC T __metal_simd_xor(T x) { using I = conditional_t<sizeof(T) <= 2, short, conditional_t<sizeof(T) == 8, long, int>>; I v = __builtin_astype(x, I); for (uint o = (uint)get_sub_group_size() >> 1; o; o >>= 1) v ^= sub_group_shuffle_xor(v, o); return __builtin_astype(v, T); }
template <typename T> METAL_FUNC T __metal_simd_prefix_inclusive_product(T x) { const uint l = get_sub_group_local_id(); for (uint o = 1u; o < (uint)get_sub_group_size(); o <<= 1) { T y = l >= o ? sub_group_shuffle(x, l - o) : (T)1; x *= y; } return x; }
template <typename T> METAL_FUNC T __metal_simd_prefix_exclusive_product(T x) { const uint l = get_sub_group_local_id(); T i = __metal_simd_prefix_inclusive_product(x); return l ? sub_group_shuffle(i, l - 1) : (T)1; }
// ---- simd_shuffle_and_fill / rotate: no OpenCL builtin; exact for a power-of-two simdgroup over sub_group_shuffle
// (every out-of-block lane read is discarded by the guard). Metal: fill puts `filling` where the shift reads past the
// block edge; rotate wraps around the block edge.
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_and_fill_up(T x, T f, ushort d, ushort m) { const uint l = get_sub_group_local_id(); return (l % m) >= d ? sub_group_shuffle(x, l - d) : f; }
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_and_fill_up(T x, T f, ushort d) { return simd_shuffle_and_fill_up(x, f, d, (ushort)get_sub_group_size()); }
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_and_fill_down(T x, T f, ushort d, ushort m) { const uint l = get_sub_group_local_id(); return (m - (l % m)) > d ? sub_group_shuffle(x, l + d) : f; }
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_and_fill_down(T x, T f, ushort d) { return simd_shuffle_and_fill_down(x, f, d, (ushort)get_sub_group_size()); }
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_rotate_up(T x, ushort d, ushort m) { const uint l = get_sub_group_local_id(); return (l % m) >= d ? sub_group_shuffle(x, l - d) : sub_group_shuffle(x, l - d + m); }
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_rotate_up(T x, ushort d) { return simd_shuffle_rotate_up(x, d, (ushort)get_sub_group_size()); }
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_rotate_down(T x, ushort d, ushort m) { const uint l = get_sub_group_local_id(); return (l % m) + d < m ? sub_group_shuffle(x, l + d) : sub_group_shuffle(x, l + d - m); }
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_rotate_down(T x, ushort d) { return simd_shuffle_rotate_down(x, d, (ushort)get_sub_group_size()); }
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_and_fill_up(V x, V f, ushort d, ushort m) { V r; for (int i = 0; i < (int)__builtin_vectorelements(V); i++) r[i] = simd_shuffle_and_fill_up(x[i], f[i], d, m); return r; }
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_and_fill_up(V x, V f, ushort d) { return simd_shuffle_and_fill_up(x, f, d, (ushort)get_sub_group_size()); }
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_and_fill_down(V x, V f, ushort d, ushort m) { V r; for (int i = 0; i < (int)__builtin_vectorelements(V); i++) r[i] = simd_shuffle_and_fill_down(x[i], f[i], d, m); return r; }
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_and_fill_down(V x, V f, ushort d) { return simd_shuffle_and_fill_down(x, f, d, (ushort)get_sub_group_size()); }
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_rotate_up(V x, ushort d, ushort m) { V r; for (int i = 0; i < (int)__builtin_vectorelements(V); i++) r[i] = simd_shuffle_rotate_up(x[i], d, m); return r; }
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_rotate_up(V x, ushort d) { return simd_shuffle_rotate_up(x, d, (ushort)get_sub_group_size()); }
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_rotate_down(V x, ushort d, ushort m) { V r; for (int i = 0; i < (int)__builtin_vectorelements(V); i++) r[i] = simd_shuffle_rotate_down(x[i], d, m); return r; }
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_rotate_down(V x, ushort d) { return simd_shuffle_rotate_down(x, d, (ushort)get_sub_group_size()); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_and_fill_up(T x, T f, ushort d, ushort m) { return simd_shuffle_and_fill_up(x, f, d, m); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_and_fill_up(T x, T f, ushort d) { return simd_shuffle_and_fill_up(x, f, d); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_and_fill_down(T x, T f, ushort d, ushort m) { return simd_shuffle_and_fill_down(x, f, d, m); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_and_fill_down(T x, T f, ushort d) { return simd_shuffle_and_fill_down(x, f, d); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_rotate_up(T x, ushort d, ushort m) { return simd_shuffle_rotate_up(x, d, m); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_rotate_up(T x, ushort d) { return simd_shuffle_rotate_up(x, d); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_rotate_down(T x, ushort d, ushort m) { return simd_shuffle_rotate_down(x, d, m); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_rotate_down(T x, ushort d) { return simd_shuffle_rotate_down(x, d); }

} // namespace metal

// packed_* vector constructors
#define packed_char4(...) m2v::mk<packed_char4>(__VA_ARGS__)
#define packed_uchar4(...) m2v::mk<packed_uchar4>(__VA_ARGS__)
#define packed_short4(...) m2v::mk<packed_short4>(__VA_ARGS__)
#define packed_int4(...) m2v::mk<packed_int4>(__VA_ARGS__)
#define packed_uint4(...) m2v::mk<packed_uint4>(__VA_ARGS__)
#define packed_float2(...) m2v::mk<packed_float2>(__VA_ARGS__)
#define packed_float4(...) m2v::mk<packed_float4>(__VA_ARGS__)
#define packed_half2(...) m2v::mk<packed_half2>(__VA_ARGS__)
#define packed_half4(...) m2v::mk<packed_half4>(__VA_ARGS__)

#else
// ---- bfloat vectors (storage + conversion; arithmetic goes through float) ----
#define M2V_BFVEC_COMMON(NAME, N)                                                                                      \
  static constexpr int m2v_n = N;                                                                                      \
  METAL_FUNC NAME() = default;                                                                                         \
  METAL_FUNC bfloat& operator[](int i) { return (&x)[i]; }                                                             \
  METAL_FUNC const bfloat& operator[](int i) const { return (&x)[i]; }                                                 \
  template <typename E> METAL_FUNC NAME(E v) { for (int i = 0; i < N; i++) (&x)[i] = bfloat((float)v[i]); }
struct bfloat2 { bfloat x, y;
  M2V_BFVEC_COMMON(bfloat2, 2)
  METAL_FUNC bfloat2(bfloat a, bfloat b) : x(a), y(b) {} };
struct bfloat3 { bfloat x, y, z;
  M2V_BFVEC_COMMON(bfloat3, 3)
  METAL_FUNC bfloat3(bfloat a, bfloat b, bfloat c) : x(a), y(b), z(c) {} };
struct bfloat4 { bfloat x, y, z, w;
  M2V_BFVEC_COMMON(bfloat4, 4)
  METAL_FUNC bfloat4(bfloat a, bfloat b, bfloat c, bfloat d) : x(a), y(b), z(c), w(d) {} };

// ---- store_hadamard_vector: keep uzu's helper callable when the source vector type must be deduced ----
// uzu writes the source parameter as `vec<SourceT, 4>`. The shim spells vec<T, N> through vec_sel<...>::type, which is
// a non-deduced context, so template argument deduction cannot recover SourceT from a builtin vector argument and the
// helper is discarded as a candidate. Take the source under its own deduced type and keep uzu's write: convert to the
// destination element type, then store through the packed reinterpret.
template <typename T, typename E>
METAL_FUNC void store_hadamard_vector(device T* destination, const E source) {
  *reinterpret_cast<device packed_vec<T, 4>*>(destination) = vec<T, 4>(source);
}


// ---- bfloat compound assignments (struct fallback only: on the __bf16 typedef the native operators already have
// Metal's widen-operate-narrow semantics, and free overloads on a builtin type are ill-formed) ----
#ifdef M2V_BFLOAT_STRUCT
METAL_FUNC bfloat& operator+=(bfloat& a, const bfloat& b) { a = bfloat(float(a) + float(b)); return a; }
METAL_FUNC bfloat& operator-=(bfloat& a, const bfloat& b) { a = bfloat(float(a) - float(b)); return a; }
METAL_FUNC bfloat& operator*=(bfloat& a, const bfloat& b) { a = bfloat(float(a) * float(b)); return a; }
METAL_FUNC bfloat& operator/=(bfloat& a, const bfloat& b) { a = bfloat(float(a) / float(b)); return a; }
#endif

// ---- metal::array ----
template <typename T, size_t N> struct array {
  T _data[N];
  METAL_FUNC constexpr size_t size() const { return N; }
  METAL_FUNC T& operator[](size_t i) { return _data[i]; }
  METAL_FUNC const T& operator[](size_t i) const { return _data[i]; }
  METAL_FUNC T* data() { return _data; }
};

// ---- Apple built-ins that standard-library headers wrap (glm_glue.metal re-declares bfloat math through them) ----
#define __METAL_MAYBE_FAST_MATH__ 0
#define __METAL_FAST_MATH__ 0
#define __METAL_PRECISE_MATH__ 0
#define M2V_BUILTIN1(N, F) template <typename T> METAL_FUNC T __metal_##N(T x, int) { return F(x); }
M2V_BUILTIN1(fabs, ::fabs) M2V_BUILTIN1(exp, ::exp) M2V_BUILTIN1(exp2, ::exp2) M2V_BUILTIN1(log, ::log) M2V_BUILTIN1(log2, ::log2)
M2V_BUILTIN1(sqrt, ::sqrt) M2V_BUILTIN1(rsqrt, ::rsqrt) M2V_BUILTIN1(tanh, ::tanh) M2V_BUILTIN1(sin, ::sin) M2V_BUILTIN1(cos, ::cos)

// ---- The rest of Apple's internals MLX's bf16_math.h and steel nax.h call: the trailing int is Metal's
// fast-math flag (one precision here) and arity follows the internal spellings. The simd forms the OpenCL
// headers lack are butterflies / Hillis-Steele over sub_group_shuffle, exact for a power-of-two simdgroup.
#define M2V_BUILTIN2(N, F) template <typename T> METAL_FUNC T __metal_##N(T x, T y, int) { return F(x, y); }
M2V_BUILTIN1(acos, ::acos) M2V_BUILTIN1(acosh, ::acosh) M2V_BUILTIN1(asin, ::asin) M2V_BUILTIN1(asinh, ::asinh)
M2V_BUILTIN1(atan, ::atan) M2V_BUILTIN1(atanh, ::atanh) M2V_BUILTIN1(ceil, ::ceil) M2V_BUILTIN1(cosh, ::cosh) M2V_BUILTIN1(sinh, ::sinh) M2V_BUILTIN1(tan, ::tan)
M2V_BUILTIN1(exp10, ::exp10) M2V_BUILTIN1(floor, ::floor) M2V_BUILTIN1(log10, ::log10) M2V_BUILTIN1(rint, ::rint)
M2V_BUILTIN1(round, ::round) M2V_BUILTIN1(trunc, ::trunc) M2V_BUILTIN1(cospi, ::cospi) M2V_BUILTIN1(sinpi, ::sinpi)
M2V_BUILTIN1(tanpi, ::tanpi) M2V_BUILTIN2(atan2, ::atan2) M2V_BUILTIN2(fmax, ::fmax) M2V_BUILTIN2(fmin, ::fmin)
M2V_BUILTIN2(fmod, ::fmod) M2V_BUILTIN2(pow, ::pow) M2V_BUILTIN2(powr, ::powr)
template <typename T> METAL_FUNC T __metal_fract(T x, int) { T ip; return ::fract(x, &ip); }
template <typename T> METAL_FUNC T __metal_fma(T x, T y, T z) { return ::fma(x, y, z); }
template <typename T> METAL_FUNC T __metal_nextafter(T x, T y) { return ::nextafter(x, y); }
template <typename T> METAL_FUNC T __metal_divide(T x, T y, int) { return x / y; }
template <typename T> METAL_FUNC T __metal_ldexp(T x, int k, int) { return ::ldexp(x, k); }
template <typename T, typename E> METAL_FUNC T __metal_frexp(T x, E* e) { return ::frexp(x, e); }
template <typename T> METAL_FUNC T __metal_fmax3(T x, T y, T z, int) { return ::fmax(::fmax(x, y), z); }
template <typename T> METAL_FUNC T __metal_fmin3(T x, T y, T z, int) { return ::fmin(::fmin(x, y), z); }
template <typename T> METAL_FUNC T __metal_fmedian3(T x, T y, T z, int) { return __metal_fmax(__metal_fmin(x, y, 0), __metal_fmin(__metal_fmax(x, y, 0), z, 0), 0); }
METAL_FUNC uint __metal_get_thread_index_in_simdgroup() { return get_sub_group_local_id(); }
METAL_FUNC uint __metal_get_simdgroup_size(ushort) { return get_sub_group_size(); }
template <typename T> METAL_FUNC T __metal_simd_broadcast(T x, ushort l) { return simd_broadcast(x, l); }
template <typename T> METAL_FUNC T __metal_simd_shuffle(T x, ushort l) { return simd_shuffle(x, l); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_xor(T x, ushort m) { return simd_shuffle_xor(x, m); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_up(T x, ushort d) { return simd_shuffle_up(x, d); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_down(T x, ushort d) { return simd_shuffle_down(x, d); }
template <typename T> METAL_FUNC T __metal_simd_sum(T x) { return simd_sum(x); }
template <typename T> METAL_FUNC T __metal_simd_max(T x) { return simd_max(x); }
template <typename T> METAL_FUNC T __metal_simd_min(T x) { return simd_min(x); }
template <typename T> METAL_FUNC T __metal_simd_prefix_inclusive_sum(T x) { return simd_prefix_inclusive_sum(x); }
template <typename T> METAL_FUNC T __metal_simd_prefix_exclusive_sum(T x) { return simd_prefix_exclusive_sum(x); }
template <typename T> METAL_FUNC T __metal_simd_product(T x) { for (uint o = (uint)get_sub_group_size() >> 1; o; o >>= 1) x *= sub_group_shuffle_xor(x, o); return x; }
template <typename T> METAL_FUNC T __metal_simd_xor(T x) { using I = conditional_t<sizeof(T) <= 2, short, conditional_t<sizeof(T) == 8, long, int>>; I v = __builtin_astype(x, I); for (uint o = (uint)get_sub_group_size() >> 1; o; o >>= 1) v ^= sub_group_shuffle_xor(v, o); return __builtin_astype(v, T); }
template <typename T> METAL_FUNC T __metal_simd_prefix_inclusive_product(T x) { const uint l = get_sub_group_local_id(); for (uint o = 1u; o < (uint)get_sub_group_size(); o <<= 1) { T y = l >= o ? sub_group_shuffle(x, l - o) : (T)1; x *= y; } return x; }
template <typename T> METAL_FUNC T __metal_simd_prefix_exclusive_product(T x) { const uint l = get_sub_group_local_id(); T i = __metal_simd_prefix_inclusive_product(x); return l ? sub_group_shuffle(i, l - 1) : (T)1; }
// ---- simd_shuffle_and_fill / rotate: no OpenCL builtin; exact for a power-of-two simdgroup over sub_group_shuffle
// (every out-of-block lane read is discarded by the guard). Metal: fill puts `filling` where the shift reads past the
// block edge; rotate wraps around the block edge.
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_and_fill_up(T x, T f, ushort d, ushort m) { const uint l = get_sub_group_local_id(); return (l % m) >= d ? sub_group_shuffle(x, l - d) : f; }
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_and_fill_up(T x, T f, ushort d) { return simd_shuffle_and_fill_up(x, f, d, (ushort)get_sub_group_size()); }
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_and_fill_down(T x, T f, ushort d, ushort m) { const uint l = get_sub_group_local_id(); return (m - (l % m)) > d ? sub_group_shuffle(x, l + d) : f; }
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_and_fill_down(T x, T f, ushort d) { return simd_shuffle_and_fill_down(x, f, d, (ushort)get_sub_group_size()); }
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_rotate_up(T x, ushort d, ushort m) { const uint l = get_sub_group_local_id(); return (l % m) >= d ? sub_group_shuffle(x, l - d) : sub_group_shuffle(x, l - d + m); }
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_rotate_up(T x, ushort d) { return simd_shuffle_rotate_up(x, d, (ushort)get_sub_group_size()); }
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_rotate_down(T x, ushort d, ushort m) { const uint l = get_sub_group_local_id(); return (l % m) + d < m ? sub_group_shuffle(x, l + d) : sub_group_shuffle(x, l + d - m); }
template <typename T> METAL_FUNC enable_if_t<__is_arithmetic(T), T> simd_shuffle_rotate_down(T x, ushort d) { return simd_shuffle_rotate_down(x, d, (ushort)get_sub_group_size()); }
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_and_fill_up(V x, V f, ushort d, ushort m) { V r; for (int i = 0; i < (int)__builtin_vectorelements(V); i++) r[i] = simd_shuffle_and_fill_up(x[i], f[i], d, m); return r; }
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_and_fill_up(V x, V f, ushort d) { return simd_shuffle_and_fill_up(x, f, d, (ushort)get_sub_group_size()); }
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_and_fill_down(V x, V f, ushort d, ushort m) { V r; for (int i = 0; i < (int)__builtin_vectorelements(V); i++) r[i] = simd_shuffle_and_fill_down(x[i], f[i], d, m); return r; }
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_and_fill_down(V x, V f, ushort d) { return simd_shuffle_and_fill_down(x, f, d, (ushort)get_sub_group_size()); }
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_rotate_up(V x, ushort d, ushort m) { V r; for (int i = 0; i < (int)__builtin_vectorelements(V); i++) r[i] = simd_shuffle_rotate_up(x[i], d, m); return r; }
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_rotate_up(V x, ushort d) { return simd_shuffle_rotate_up(x, d, (ushort)get_sub_group_size()); }
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_rotate_down(V x, ushort d, ushort m) { V r; for (int i = 0; i < (int)__builtin_vectorelements(V); i++) r[i] = simd_shuffle_rotate_down(x[i], d, m); return r; }
template <typename V> METAL_FUNC enable_if_t<!__is_arithmetic(V), V> simd_shuffle_rotate_down(V x, ushort d) { return simd_shuffle_rotate_down(x, d, (ushort)get_sub_group_size()); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_and_fill_up(T x, T f, ushort d, ushort m) { return simd_shuffle_and_fill_up(x, f, d, m); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_and_fill_up(T x, T f, ushort d) { return simd_shuffle_and_fill_up(x, f, d); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_and_fill_down(T x, T f, ushort d, ushort m) { return simd_shuffle_and_fill_down(x, f, d, m); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_and_fill_down(T x, T f, ushort d) { return simd_shuffle_and_fill_down(x, f, d); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_rotate_up(T x, ushort d, ushort m) { return simd_shuffle_rotate_up(x, d, m); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_rotate_up(T x, ushort d) { return simd_shuffle_rotate_up(x, d); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_rotate_down(T x, ushort d, ushort m) { return simd_shuffle_rotate_down(x, d, m); }
template <typename T> METAL_FUNC T __metal_simd_shuffle_rotate_down(T x, ushort d) { return simd_shuffle_rotate_down(x, d); }

} // namespace metal

// packed_* vector constructors
#define packed_char4(...) m2v::mk<packed_char4>(__VA_ARGS__)
#define packed_uchar4(...) m2v::mk<packed_uchar4>(__VA_ARGS__)
#define packed_short4(...) m2v::mk<packed_short4>(__VA_ARGS__)
#define packed_int4(...) m2v::mk<packed_int4>(__VA_ARGS__)
#define packed_uint4(...) m2v::mk<packed_uint4>(__VA_ARGS__)
#define packed_float2(...) m2v::mk<packed_float2>(__VA_ARGS__)
#define packed_float4(...) m2v::mk<packed_float4>(__VA_ARGS__)
#define packed_half2(...) m2v::mk<packed_half2>(__VA_ARGS__)
#define packed_half4(...) m2v::mk<packed_half4>(__VA_ARGS__)

#endif

// float2(bfloat2) and friends
namespace m2v {
template <typename V, typename B> METAL_FUNC metal::enable_if_t<(B::m2v_n > 0), V> mk1(B b) {
  V r;
  for (int i = 0; i < B::m2v_n; i++) r[i] = (elem_t<V>)(float)b[i];
  return r;
}
} // namespace m2v

typedef metal::bfloat bfloat16_t;  // the MLX / Zig-host spelling
typedef half float16_t;

namespace metal {
typedef simdgroup_matrix<half, 8, 8> simdgroup_half8x8;
typedef simdgroup_matrix<float, 8, 8> simdgroup_float8x8;
typedef simdgroup_matrix<bfloat, 8, 8> simdgroup_bfloat8x8;
} // namespace metal
