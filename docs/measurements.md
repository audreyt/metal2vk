# Kernel measurements

Per-kernel results for uzu kernels, translated unmodified and run through the open source Vulkan driver. For the headline numbers
(how many kernels translate, how many were checked against a reference) see [what-this-enables.md](what-this-enables.md); for
what fails and why, [coverage.md](coverage.md).

Measured on an Apple M1 Max with the omarchy-mlx fork of the driver (Mesa 26.3.0-devel). "Hand kernel" is the Vulkan kernel in
[omarchy-mlx](https://github.com/joshuaswarren/omarchy-mlx), timed through mlx.

| kernel (uzu source, unmodified) | result | translated | hand kernel |
| --- | --- | --- | --- |
| Softmax f32, 4096 x 4096 (`simd_max`, `simd_sum`, threadgroup memory) | max abs error 2.4e-7 | 733 us | 773 us |
| Softmax f16, 4096 x 4096 | max abs error 2.4e-4 | 358 us | 403 us |
| Activation SILU f32 / f16, 8M elements | max rel error 3.5e-7 / 3.8e-4 | 340 us / 340 us | not comparable (mlx composes it from several ops) |
| 8x8 tile matmul from uzu's `SimdgroupMMA`, 1024^3 f32, one tile per subgroup, no reuse | max rel error 1.4e-6 | 0.81 TFLOP/s | 4.1 TFLOP/s (tiled GEMM) |
| the same tile kernel, shuffle emulation instead of cooperative matrix | max rel error 1.4e-6 | 0.03 TFLOP/s | |
| 4x4 register-tiled matmul from the same header, 1024^3 f32 | max rel error 1.1e-6 | 4.27 TFLOP/s | 4.07 TFLOP/s (tiled GEMM) |
| uzu `Gemm` (`gemm.metal`, SimdgroupMmaCore, f32, Tile64x64x32), 1024^3 | max rel error 1.2e-6 | 1.42 TFLOP/s (1.42 / 4.07 = 0.35 of the hand GEMM, the "about 0.3" of the README; at 256^3 0.36 TFLOP/s) | 4.07 TFLOP/s |

## Reading the table

- The 8x8 tile matmul is a deliberately naive kernel built from uzu's header (no operand reuse), so its gap to a tiled GEMM says
  little about the translation. uzu's tiled `Gemm` is the real comparison, and at about 0.3 of the hand-tuned GEMM it is why
  matrix multiplies keep using the hand-tuned kernels while metal2vk covers the long tail of smaller kernels.
- `Gemm` compiles and validates for the 8x32x32, 32x32x32 and 64x64x32 tilings. Its first failures in clspv all came from the
  front end's output (clang's InstCombine turning typed float addressing into byte addressing, and `noinline` closures); the typed
  route in [how-it-works.md](how-it-works.md) fixes that. The guards in `clspv-patches/apply-coopmat-lowering.py` remain, so an input
  that breaks those passes stops with the offending types instead of hanging.
- Null optional-buffer arguments crash clspv's producer, so the wrappers bind real typed buffers for them. clspv drops unused
  arguments; the test drivers bind from the reflection JSON that `compile.sh` writes next to each module.
- Cooperative matrix needs a driver that exposes `VK_KHR_cooperative_matrix`. Stock Mesa Honeykrisp does not yet; the omarchy-mlx
  fork does.

## More

- `bench/` holds the harness that produced these rows and the latest result files ([bench-status.md](bench-status.md)).
- [gemm-gap.md](gemm-gap.md) has the measurements for the `Gemm` gap and what they rule out.
- `census.py` parses every uzu `.metal` file through the shim.
