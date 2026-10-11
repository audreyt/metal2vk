// mlx_prelude.h: the MLX device-side prelude types. MLX's own kernel build prepends
// mlx/backend/metal/kernels/complex.h to every Metal source it compiles, and the steel/gemm headers
// reference complex64_t through that prelude, so a standalone kernel file that expands those headers
// never sees the definition. This is the pinned prelude ported for the shim; two diffs against the
// upstream header: the <metal_stdlib> include (this shim provides it) and the bf16.h include
// (bfloat16_t comes from m2v_extra.h). Included at the end of metal_stdlib.
#pragma once

namespace metal {

template <typename T> struct complex_t;

template <typename T>
constexpr constant bool is_complex_v = false;

template <typename T>
constexpr constant bool is_complex_v<complex_t<T>> = true;

// Metal accepts explicit bfloat casts that is_convertible_v reports as false.
template <typename From, typename To>
static constexpr constant bool is_lane_convertible_v =
    is_convertible_v<From, To> ||
    (is_same_v<To, bfloat16_t> && is_convertible_v<From, float>) ||
    (is_same_v<From, bfloat16_t> && is_convertible_v<float, To>);

template <typename T>
struct complex_t {
  using value_type = T;

  T real;
  T imag;

  // Constructors
  constexpr complex_t(T real, T imag) thread : real(real), imag(imag) {};
  constexpr complex_t() thread : real(0), imag(0) {};
  constexpr complex_t() threadgroup : real(0), imag(0) {};

  // Conversions from scalar types
  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(U x) thread : real(static_cast<T>(x)),
                                    imag(static_cast<T>(0)) {}

  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(U x) threadgroup : real(static_cast<T>(x)),
                                         imag(static_cast<T>(0)) {}

  template <
      typename U,
      typename = typename enable_if<
          !is_same_v<U, T> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(complex_t<U> x) constant : real(static_cast<T>(x.real)),
                                                 imag(static_cast<T>(x.imag)) {}

  // Conversions to and from two-lane vectors (the FFT lane representation)
  constexpr complex_t(vec<T, 2> v) thread : real(v.x), imag(v.y) {};
  constexpr complex_t(vec<T, 2> v) threadgroup : real(v.x), imag(v.y) {};
  constexpr complex_t(vec<T, 2> v) device : real(v.x), imag(v.y) {};
  constexpr complex_t(vec<T, 2> v) constant : real(v.x), imag(v.y) {};

  constexpr operator vec<T, 2>() const thread {
    return vec<T, 2>(real, imag);
  }

  constexpr operator vec<T, 2>() const threadgroup {
    return vec<T, 2>(real, imag);
  }

  constexpr operator vec<T, 2>() const device {
    return vec<T, 2>(real, imag);
  }

  constexpr operator vec<T, 2>() const constant {
    return vec<T, 2>(real, imag);
  }

  // Conversions to scalar types
  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<T, U>>::type>
  constexpr operator U() const thread {
    return static_cast<U>(real);
  }

  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<T, U>>::type>
  constexpr operator U() const threadgroup {
    return static_cast<U>(real);
  }

  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<T, U>>::type>
  constexpr operator U() const device {
    return static_cast<U>(real);
  }

  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<T, U>>::type>
  constexpr operator U() const constant {
    return static_cast<U>(real);
  }
};

using complex32_t = complex_t<half>;
using complex64_t = complex_t<float>;

static_assert(sizeof(complex32_t) == 2 * sizeof(half));
static_assert(sizeof(complex64_t) == 2 * sizeof(float));
static_assert(sizeof(complex_t<bfloat16_t>) == 2 * sizeof(bfloat16_t));

} // namespace metal
