# sweep/mlx-core: MLX's own library kernels through the m2v pipeline

A second corpus alongside `sweep/mlx-batch` (assembled call-site kernels). This one is
the `mlx/backend/metal/kernels/` source tree pinned to a specific commit, expanded and
compiled exactly as MLX builds it: the `instantiate_*` macros and the templated
`[[host_name("...")]] [[kernel]]` instantiations are the entries. The `sweep/m2v-sweep`
pattern, the `tools/m2v-compile` cache and the `sweep/m2v_sweep.py` entry discovery are
reused unchanged; this directory adds only the expansion step and the per-entry runner.

**Toolchain (this run, 2026-10-12):**
- clang-19 / opt-19, OpenCL C++ 2021 (the floor CI uses clang-23; on this machine the
  same clang-19 / clspv-bit2b baseline of the existing uzu + TensorFold corpus is
  `433 of 610`, so the numbers below are not directly comparable to CI's 610/610.)
- `~/scratch/sweep/clspv-bit2b` (LLVM 24.0.0git, the patched clspv mandated for this sweep).
- `~/scratch/sweep/mlx-batch-src/mlx` at `1f8e74e3f12f31365464a6867c6579f0e9b29d85`
  ("Hold GIL in AttachedData destructor", #4391).

**Fetch command** (the README assumes the worktree at
`~/src/metal2vk-mlxcore/` on branch `sweep/mlx-core`):

```sh
git clone https://github.com/joshuaswarren/metal2vk.git metal2vk-mlxcore
git -C metal2vk-mlxcore checkout sweep/mlx-core
git clone https://github.com/ml-explore/mlx.git scratch/sweep/mlx-batch-src/mlx
git -C   scratch/sweep/mlx-batch-src/mlx checkout 1f8e74e3f12f31365464a6867c6579f0e9b29d85
```

## MLX source license

The compiled sources are from [ml-explore/mlx](https://github.com/ml-explore/mlx),
MIT licensed:

> MIT License
>
> Copyright © 2023 Apple Inc.

The pinned tree is **read at run time only**; no MLX sources are committed to this
directory. Every entry's source line and line number is in `results.tsv`; the
upstream `.metal` files are not.

## Run

```sh
cd metal2vk-mlxcore
export CLSPV=~/scratch/sweep/clspv-bit2b CLANG=clang-19 OPT=opt-19
export MLX_SRC=~/scratch/sweep/mlx-batch-src/mlx
export M2V_DEFS="-DM2V_MLX_TREE"            # opt the shim's MLX prelude out (the real tree carries its own complex.h)
export M2V_CACHE_DIR=~/scratch/sweep/mlxcore-run/cache
nice -n 19 ionice -c 2 -n 7 python3 sweep/mlx-core/mlx_core_run.py \
    --mlx "$MLX_SRC" \
    --work ~/scratch/sweep/mlxcore-run/work \
    --out  ~/scratch/sweep/mlxcore-run/out \
    --jobs "$(nproc)" \
    --results sweep/mlx-core/results.tsv
```

`mlx_core_run.py` writes the manifest (one row per (work-file, host_name)) to
`work/manifest.json`; pass `--skip-expand` to reuse it on a re-run. The compile phase
is a thread pool of `tools/m2v-compile` subprocesses, one per entry, with the
content-addressed cache under `$M2V_CACHE_DIR`. The `m2v-compile` retry / crash /
refusal machinery is unchanged. The runner is a strict compile-only pass: every
produced module is re-validated with `spirv-val --target-env vulkan1.3`, and a
`gpu-job.sh` dispatch is **not** part of this corpus.

## Results (clang-19, this clspv-bit2b)

```
entries  17301
valid     1067   (compile ok AND spirv-val vulkan1.3)
failed   16234   (compile failed; spirv-val never rejects on this toolchain)
```

### By kernel family

| family                            | valid / entries |
|---|---:|
| `arange`                          | 11 / 11 |
| `arg_reduce`                      | 10 / 24 |
| `binary`                          | 0 / 4122 |
| `binary_two`                      | 0 / 225 |
| `conv`                            | 0 / 18 |
| `copy`                            | 923 / 2496 |
| `dot`                             | 3 / 3 |
| `fence`                           | 0 / 3 |
| `fft`                             | 0 / 101 |
| `fp_quantized`                    | 0 / 480 |
| `fp_quantized_nax`                | 0 / 126 |
| `gemv`                            | 0 / 256 |
| `gemv_masked`                     | 0 / 528 |
| `layer_norm`                      | 0 / 12 |
| `logsumexp`                       | 0 / 6 |
| `quantized`                       | 0 / 2052 |
| `quantized_nax`                   | 0 / 864 |
| `random`                          | 0 / 2 |
| `reduce`                          | 0 / 2396 |
| `rms_norm`                        | 0 / 12 |
| `rope`                            | 0 / 18 |
| `scaled_dot_product_attention`    | 0 / 57 |
| `scan`                            | 0 / 456 |
| `searchsorted`                    | 26 / 26 |
| `softmax`                         | 4 / 10 |
| `sort`                            | 12 / 264 |
| `steel_attn`                      | 0 / 78 |
| `steel_conv`                      | 0 / 162 |
| `steel_gemm`                      | 0 / 1404 |
| `ternary`                         | 78 / 212 |
| `unary`                           | 0 / 877 |

### By cause (failed entries)

| cause         | count |
|---|---:|
| `shim_gap`    | 15031 |
| `other`       | 1189 |
| `clspv_limit` | 14 |

### Top 3 failure causes, with one example each

1. **`shim_gap` — 15031 entries (92.6% of failures)**
   One chokepoint, not a per-feature family: every MLX translation unit that
   includes a math op header pulls in `kernels/logging.h`, which declares a
   `constant mlx::os_log logger(...)` at file scope. The Metal 3.2+ branch of
   MLX's `logging.h` aliases `mlx::os_log = metal::os_log`; the shim does not
   model `metal::os_log`, so the alias is undeclared and every constructor
   call fails to match.

   Example (`ss_LogAddExpcomplex64`, first error truncated to one line):

   ```
   .../binary_ops.h: error: no matching constructor for initialization
     of '__constant mlx::os_log'
   ```

   Unblocking path: a 5-10-line shim addition defining `namespace metal { struct os_log { ... } }`
   (or, equivalently, defining `__METAL_VERSION__` so MLX's else-branch struct is
   the one used and patching it to be call-compatible). This single change is
   expected to lift the `binary` / `reduce` / `quantized` / `steel_gemm` /
   `gemv` / `scan` / `fft` / `conv` / `rope` / `layer_norm` / `rms_norm` /
   `scaled_dot_product_attention` / `fp_quantized` / `quantized_nax` /
   `binary_two` / `unary` / `logsumexp` / `gemv_masked` / `fp_quantized_nax` /
   `random` / `fence` families off zero. It was **not** applied here: the fix
   lives in the MLX header itself (which we cannot modify from this corpus),
   and the shim half needs more than the "clearly under 30 lines" bar the
   assignment sets.

2. **`other` — 1189 entries (7.3%)**
   Two clusters: the m2v-compile-side `select_entry` failing on template
   instantiations whose generated `[[host_name("...")]]` collides with a
   second entry of the same name in the same file (32 rows in
   `steel_gemm_masked` — `steel_gemm_block_outmask_bfloat16_opmask_*`); and
   clang-frontend timeouts / segfaults on bfloat-vector-heavy translation
   units under the 12-job pool (192 rows, exit 139 / 134 from the parallel
   cold-cache pass — re-running the affected entries sequentially is the
   worked example the corpus's `M2V_CLSPV_TIMEOUT` knob would address).

   Example (one of the 192 cold-cache clang-frontend exits):

   ```
   steel_gemm_splitk_nn_bfloat16: clang-19: error: clang frontend command
     failed with exit code 139 (use -v to see invocation)
   ```

3. **`clspv_limit` — 14 entries (0.09%)**
   clspv's pointer-bitcast pass gives up on a shape it can't rewrite, the
   driver catches the `M2V-GIVEUP: gepfromgep-skip` marker and keeps no
   SPIR-V (by design — a miscompiled module is worse than a missing one).
   This is the same `simplify-pointer-bitcast` class that the uzu / TF
   corpora already name; on a clang-23 build with the merged clspv patches
   it would behave the same way.

   Example (`argmax_bfloat16`):

   ```
   clspv pass gave up on a pointer rewrite; the module would be miscompiled
     (no SPIR-V kept) | M2V-GIVEUP: gepfromgep-skip | ...
   ```

## What is not verified

- **No GPU dispatch.** The run produces and `spirv-val`s every module; it
  does not run any kernel. The companion `sweep/m2v_sweep_run.py` harness
  would do that on a GPU host, with synthetic data and a CPU reference
  where one exists. The `sweep/mlx-batch` README documents the same
  compile-only / no-GPU convention.
- **No parity harness.** No MLX reference outputs are compared.
- **The toolchain difference.** This is a clang-19 + this-clspv-bit2b
  measurement; CI's clang-23 build of the same sources may differ. The
  610-floor mechanism (`sweep/coverage-floor.txt` +
  `sweep/check_coverage.py`) and the CI gate are untouched. A baseline
  uzu + TF sweep on this branch holds the same 433 / 610 as pristine
  `origin/main` (no regressions from the shim / driver changes this
  branch introduces for MLX support).
