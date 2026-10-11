# Bench status

Bench harness for the translated (m2v-run) kernels against the hand kernels
(omarchy-mlx, timed through mlx) on one table. It also runs a tuning sweep.
Receipts from the latest G13C run are committed next to this file. They are
`bench/results-g13c-2026-10-09.json` and `bench/sweep-g13c-2026-10-09.json`.

## Reproduce (one command on the GPU host)

```
# on the GPU host, from a checkout of this branch:
PATH=$HOME/bin:$PATH <gpu-queue-wrapper> -m 25 -- bash bench/run-all.sh OUTDIR
```

`run-all.sh` compiles the registered cases (typed-GEP route below). It then
runs the table (correctness + timing per row, mlx hand side included) and the
sweep. It drops `results-<device>-<date>.json` and `sweep-<device>-<date>.json`
into `bench/`. Submits inside are sized to about 2 s (same rule as m2v-test.py).
Host paths live in an untracked `bench/env-local.sh`. The file holds the uzu
checkout, patched clspv, mlx python and fork ICD. The template is
`bench/env-local.sh.template`. The ICD can also be given per run via `M2V_ICD`.
The `--device` and `--icd` flags (in either `--name value` or `--name=value`
form) go to both the benchmark and `sweep.py`. The `--device g14c` flag selects
the output tag and the ICD environment. Without it the tag is derived from
vulkaninfo deviceName (g13c = M1 Max class, g14c = M2 Max class).

## Compile route: typed GEPs

clang's `-O2` InstCombine canonicalises float GEPs into i8 byte-offset form.
The clspv stage then emits hundreds of byte-wise loads; the coopmat lane found
that first, with independent confirmation below. Compiling with
`M2V_OPT="-O0 -Xclang -disable-O0-optnone"` keeps GEPs typed. The clspv
pipeline then does the optimising. Measured on the G13C device (M1 Max class),
same device, same ICD:

| kernel | `-O2` route | typed-GEP route |
| --- | --- | --- |
| tile matmul 1024³ f32 | 11264 us (181 GFLOP/s) | 2651 us (813 GFLOP/s) |
| activation SILU f32 8M | 397 us | 336 us |

The bench table below is all typed-GEP route. The sweep keeps the `-O2` route
variants (`base_o2route`) so the comparison is re-measurable.

## Table (G13C device, Apple M1 Max, Mesa 26.3.0-devel git-6543eeb7df fork ICD, 2026-10-09)

Hand = omarchy-mlx through mlx, wall time of a batch of independent evals, best
of 3. For activation the faster of mlx's fused `nn.silu` and unfused
`x*sigmoid(x)` is the comparator. Both are in the receipts. Hand-side f16 silu
varies run to run on this driver (211 us / 158.7 GB/s in the committed receipt,
148 us / 227 GB/s in a same-day rerun). The f16 explanation below rests on the
unfused-parity measurement, which is stable (315.5 vs 330.8 us).

| kernel | translated | hand | ratio | check |
| --- | --- | --- | --- | --- |
| activation f32, 8M | 343.5 us (195 GB/s) | 739.4 us (unfused) | 0.46x | err 3.5e-7 OK |
| activation f16, 8M | 330.8 us (101 GB/s) | 211.4 us (fused) | 1.56x | err 3.8e-4 OK |
| softmax f32 4096² | 742.3 us (181 GB/s) | 770.9 us | 0.96x | err 2.4e-7 OK |
| softmax f16 4096² | 358.1 us (187 GB/s) | 410.9 us | 0.87x | err 2.4e-4 OK |
| naive tile matmul 256³ | 38.5 us (871 GFLOP/s) | 12.3 us | 3.13x | err 6.7e-7 OK |
| naive tile matmul 1024³ | 2667.8 us (805 GFLOP/s) | 530.9 us | 5.03x | err 1.3e-6 OK |
| rt2x2 256³ | 32.6 us | 21.4 us | 1.52x | err 6.7e-7 OK |
| rt4x2 256³ | 45.3 us | 21.4 us | 2.12x | err 6.7e-7 OK |
| rt4x4 256³ | 65.1 us | 21.4 us | 3.04x | err 6.7e-7 OK |
| rt2x2 1024³ | 1323.4 us (1623 GFLOP/s) | 526.0 us | 2.52x | err 1.3e-6 OK |
| rt4x2 1024³ | 1076.3 us (1995 GFLOP/s) | 526.0 us | 2.05x | err 1.3e-6 OK |
| rt4x4 1024³ | 489.6 us (4386 GFLOP/s) | 526.0 us | 0.93x | err 1.3e-6 OK |

Uzu's full `Gemm` has no row. The bench has no driver for it. The clspv stage
still does not finish (parent slice's open item).

## M2 Max comparison (G14C, 2026-10-11)

This compares translated kernels with MLX's own kernels on the same GPU. It is
not a comparison with native Metal. Both runs used Mesa 26.3.0-devel
git-6543eeb7df7, kernel 7.1.12, and load average 1.30 at the start. The times
are the raw rows of pull request 70. The throughput columns are the matching
values from the same runs' JSON output, `bench/m2v-bench.py`. Every correctness
check was OK.

| kernel | run 1 translated / MLX | run 2 translated / MLX | translated throughput, run 1 / run 2 | spread |
| --- | --- | --- | --- | --- |
| activation f32, 8M | 282.0 / 637.4 us | 287.6 / 637.3 us | 237.9 / 233.3 GB/s | 5.6 us (2.0%) |
| activation f16, 8M | 268.5 / 133.8 us | 275.9 / 136.1 us | 125.0 / 121.6 GB/s | 7.3 us (2.7%) |
| softmax f32, 4096² | 459.5 / 613.5 us | 465.0 / 614.5 us | 292.1 / 288.6 GB/s | 5.6 us (1.2%) |
| softmax f16, 4096² | 253.3 / 277.8 us | 255.8 / 277.3 us | 264.9 / 262.4 GB/s | 2.4 us (1.0%) |
| tilematmul f32, 256³ | 30.0 / 41.0 us | 34.7 / 40.9 us | 1119.4 / 966.6 GFLOP/s | 4.7 us (15.8%) |
| tilematmul f32, 1024³ | 1383.5 / 351.4 us | 1382.8 / 352.0 us | 1552.2 / 1553.0 GFLOP/s | 0.8 us (0.1%) |
| rt2x2, 256³ | 25.9 / 41.1 us | 25.7 / 54.2 us | 1295.7 / 1304.0 GFLOP/s | 0.2 us (0.6%) |
| rt4x2, 256³ | 36.4 / 41.1 us | 36.4 / 54.2 us | 922.4 / 922.1 GFLOP/s | 0.0 us (0.0%) |
| rt4x4, 256³ | 50.4 / 41.1 us | 50.1 / 54.2 us | 666.3 / 669.2 GFLOP/s | 0.2 us (0.4%) |
| rt2x2, 1024³ | 885.6 / 351.5 us | 888.2 / 370.0 us | 2424.9 / 2417.8 GFLOP/s | 2.6 us (0.3%) |
| rt4x2, 1024³ | 730.1 / 351.5 us | 731.4 / 370.0 us | 2941.2 / 2936.2 GFLOP/s | 1.2 us (0.2%) |
| rt4x4, 1024³ | 450.6 / 351.5 us | 451.2 / 370.0 us | 4766.0 / 4759.0 GFLOP/s | 0.7 us (0.1%) |

Spread is the difference in translated time between runs.

## Ratios above 1.5x, each with its measurement

Activation f16 (1.56x vs fused mlx silu, 1.05x vs unfused). The Metal kernel
is `x*sigmoid(x)` element-wise. mlx's unfused two-op f16 runs at 106 GB/s, the
same as ours (101 GB/s), 330.8 vs 315.5 us. That is parity for the same op
structure. The gap is to mlx's single fused f16 kernel (158.7 GB/s). The
translated f16 path moves half the bytes of f32 in the same wall time (101 vs
195 GB/s). It is element-throughput-bound, not bandwidth-bound. The sweep
confirms widening per-thread work does not move it (ept2 352.6 us, ept4 338.9 vs
base 333.1). The lever is vectorised 16-bit access in the lowering. That is
pipeline work, not bench scope.

Naive tile matmul (3.13x at 256³, 5.03x at 1024³). The kernel is uzu's 8x8
tile driver with no operand reuse. One subgroup streams K/8 A-fragments and
K/8 B-fragments from DRAM per 8x8 output tile: 2 flops/byte. Measured bytes:
1024³ moves 1.07 GB in 2668 us = 402 GB/s. That is the M1 Max DRAM ceiling, so
the kernel is bandwidth-bound by construction. The rt rows are the controlled
experiment: same pipeline, same lowering, same device, only reuse changes. At
rt4x4 (16 flops/byte) the kernel reaches 4386 GFLOP/s and beats the hand GEMM
(0.93x). The gap is the kernel's operand traffic, not the Metal→Vulkan stack.

The rt shapes at 256³ (1.52x / 2.12x / 3.04x). Occupancy: one subgroup per
MRxNR block means rt4x4 launches (256/32)² = 64 threadgroups (2048 threads)
against the naive kernel's 1024. The GPU is underfilled and the 12-21 us hand
figure is dominated by submit overhead at this size. Measured: the same shapes
at 1024³ (grid 32x32 = 1024 threadgroups) flip to 2.52x / 2.05x / 0.93x.

## Sweep winners (bench/sweep-g13c-2026-10-09.json)

| kernel | winner | note |
| --- | --- | --- |
| activation f32 8M | base (typed-GEP) | wg512 within noise (336 vs 336); ept2/ept4 no better; `-O2` route 18% slower |
| activation f16 8M | base (typed-GEP) | ept2/ept4 no better (measured above) |
| softmax f32 4096² | base | workgroup and elements/thread are baked into softmax.metal; no wrapper knob |
| tile matmul 1024³ | chains4 (`-DM2V_TM_CHAINS=4`) | 2591 vs 2651 us, marginal; the real answer is rt4x4 |

Tuning knobs are compile-time defines in the wrappers. They are driven by
`M2V_DEFS` / `M2V_OPT` through compile.sh. Defaults reproduce the unmodified
uzu mapping exactly (`M2V_ACT_WG=256`, `M2V_ACT_EPT=1`, `M2V_TM_CHAINS=1`).
