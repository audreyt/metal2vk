# MLX small-M quantized kernels on the Vulkan route

Question: what does a verify step of T = 2..8 rows cost on a 4-bit linear layer, and does the cost grow with T?
The deployed Vulkan route (`mx.quantized_matmul` in omarchy-mlx) pays about five decode steps for a verify step.
MLX's own Metal kernels for small M reuse each decoded weight group across the rows.
This directory compiles those kernels with metal2vk and times them against the deployed route in one process.

## The kernels

All three come from `mlx/backend/metal/kernels/quantized.h`.
The MLX commit is the one in omarchy-mlx's `mlx.lock` (9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8).
A kernel body calls the MLX impl function.
`qmv_corpus.py` flattens the MLX headers into the `header` of an `mx.fast.metal_kernel` call.
The metal2vk route in omarchy-mlx then compiles and dispatches the kernel.

| id | MLX kernel | grid | what it does |
|---|---|---|---|
| `qmvf` | `affine_qmv_fast` | T x N/8 threadgroups of 64 | one threadgroup per row, weights read T times |
| `qmvw2` to `qmvw5` | `affine_qmv_wide` (NV = 2..5) | ceil(T/NV) x N/8 | one weight decode feeds NV rows |
| `qmmt` | `affine_qmm_t` | N/32 x ceil(T/32), 128 threads | 32 x 32 tile through simdgroup matrices |

T = 6, 7, 8 take two `qmvw` tiles (3, 4, 4 rows), the rule in `qmv_wide` in MLX's `quantized.cpp`.
MLX Metal uses `qmv_wide` for affine only on gen 15 and later. Here it is a candidate that Metal on older chips does not ship.

## Build (toolchain host, no GPU)

```sh
source $M2V_CT/env.sh                       # CLANG, OPT, CLSPV, SPIRV_VAL
export M2V_COMPILE=$PWD/tools/m2v-compile M2V_CACHE_DIR=$PWD/.m2v-cache
python3 bench/mlx-qmv/qmv_corpus.py build $MLX_SRC OUT
```

`$MLX_SRC` is the MLX checkout at the pin. `OUT` receives `header.metal`, one `.metal` per variant, and `manifest.json`.
It also receives `aot/<key>.spv` and `aot/<key>.json`.
The key is sha256 of the MSL, a NUL byte and the entry name. omarchy-mlx looks modules up by that key.
Every module passes `spirv-val --target-env vulkan1.3` inside `m2v-compile`.

## Run (lab Mac, under the GPU queue wrapper)

Copy `qbench.py`, `qmv_corpus.py` and `OUT/{header.metal,aot,manifest.json}` to one directory `D` on the Mac.
No compiler is needed there. Use a wheel that contains the metal2vk route (`m2v_route`).

```sh
cd D
MLX_OMARCHY_M2V_SUMMARY=1 PYTHON_OF_THE_WHEEL qbench.py --build . --out result.json 2> route.err | tee result.txt
grep -q 'translator=0 fallbacks=0 refusals=0' route.err    # else the translated rows are not valid
```

`qbench.py` sets `MLX_OMARCHY_M2V_AOT_DIR` to `D/aot`.
It routes `qmv_wide`, `qmv_fast` and `qmm_t` to metal2vk.
Defaults: K 5120 and N 10240, 12288, 17408, 248320 (the 27B layer shapes and the head).
Also T 1..8, 11 rounds, 16 dispatches per timed batch, group size 64, 4 bits, bf16.

## What the run measures

- One process, one device. Each round times one batch of every (arm, T) cell in a shuffled order, reversed on odd rounds.
  Round 0 is warm-up. The table shows the median of the other rounds, in ms per dispatch.
- Weights rotate through enough copies to pass 256 MiB, so the system cache never holds them.
- Each cell's output is compared with a float64 CPU reference on 96 sampled columns.
  Row error is the largest absolute error in a row, over the largest reference magnitude in that row, on those columns.
- After the table, each arm prints its cost at T = 8 over its cost at its smallest T (T = 2 for `qmvw`, else T = 1).
  Deployed T = 1 is the decode step. A ratio near 1 means flat in T. A ratio near 4 means the cost scales with T.
- An arm that raises on the device (for example, no matrix support for `qmmt`) is listed as not run and does not stop the others.
- Once per N and T, the deployed arm runs with `MLX_OMARCHY_TRACE_DISPATCH` set. The JSON holds the kernel ids and grids it dispatched.
  That shows which route the build chose for each T.

Limits: weights and activations are random. The rows show cost and numeric error. They say nothing about model quality.
Each timed batch includes host encode time. The kernels run for 0.1 ms or more, so the GPU should set the pace.
The deployed arm ("main wheel default route") is `mx.quantized_matmul` on the same wheel and driver, with the default environment.
A statement about a shipped release needs a separate run of that arm alone on the release wheel.
The release wheel has no metal2vk route, so it cannot run the translated arms.
`qbench.py` reads `/proc/self/maps` after the first dispatch and hashes every mapped Vulkan driver library.
The paths and the first 16 hex of each hash go to the JSON.
With `--driver-sha16 HASH` the run exits 4 unless that hash is the only one loaded.
The JSON records the `MLX_OMARCHY_*` variables.

## Test

```sh
python3 -m unittest bench/mlx-qmv/test_qmv_corpus.py
```
