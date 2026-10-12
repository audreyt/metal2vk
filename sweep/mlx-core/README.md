# sweep/mlx-core: MLX's own library kernels through the m2v pipeline

A second corpus alongside `sweep/mlx-batch` (assembled call-site kernels). This one is
the `mlx/backend/metal/kernels/` source tree pinned to a specific commit, expanded and
compiled exactly as MLX builds it: the `instantiate_*` macros and the templated
`[[host_name("...")]] [[kernel]]` instantiations are the entries. The `sweep/m2v-sweep`
pattern, the `tools/m2v-compile` cache and the `sweep/m2v_sweep.py` entry discovery are
reused unchanged; this directory adds only the expansion step and the per-entry runner.

**Toolchain (this run, 2026-10-12):**
- clang-19 / opt-19, OpenCL C++ 2021 (the floor CI uses clang-23; the clang-19 / this-clspv
  baseline on the same machine is `433 of 610` on the uzu + TensorFold corpus, so numbers
  below are not directly comparable to CI's 610/610.)
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
directory (every entry's source line and line number is in `results.tsv`; the
upstream .metal files are not).

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
refusal machinery is unchanged.

The runner is a strict compile-only pass: every produced module is re-validated with
`spirv-val --target-env vulkan1.3`, and a `gpu-job.sh` dispatch is **not** part of
this corpus (the 610-floor CI also runs compile-only at sweep time; the GPU
verification is the `sweep/m2v_sweep_run.py` follow-up that the assignment does not
extend here).

## Results

`results.tsv` columns: `origin | name | family | compile | spirv_val | first_error |
m2v_giveup | cause`. `origin` is the upstream file:line of the instantiation (taken
from the line markers clang leaves in the expansion). `family` is the kernel family
for the group table below. `m2v_giveup` is the refused-construct on `refused` rows,
`no` otherwise (the convention `sweep/mlx-batch` uses).

### Totals

FILL after run.

### By kernel family

FILL after run.

### By cause

FILL after run.

### Top failure causes (with example errors)

FILL after run.

## What is not verified

- No GPU dispatch. The run produces and `spirv-val`s every module; it does not run
  any kernel. The companion `sweep/m2v_sweep_run.py` harness would do that on a
  GPU host, with synthetic data and a CPU reference where one exists.
- No parity harness (no MLX reference outputs).
- The numbers are a clang-19 + this-clspv-bit2b measurement; CI's clang-23 build of
  the same sources may differ. The floor mechanism (`sweep/coverage-floor.txt` +
  `sweep/check_coverage.py`) and the CI gate are untouched.
