# sweep/mlx-core: MLX's own library kernels through the m2v pipeline

A second corpus alongside `sweep/mlx-batch` (assembled call-site kernels). This one is
the `mlx/backend/metal/kernels/` source tree pinned to a specific commit, expanded and
compiled exactly as MLX builds it: the `instantiate_*` macros and the templated
`[[host_name("...")]] [[kernel]]` instantiations are the entries. The `sweep/m2v-sweep`
pattern, the `tools/m2v-compile` cache and the `sweep/m2v_sweep.py` entry discovery are
reused unchanged; this directory adds only the expansion step and the per-entry runner.

## Toolchain of this run (2026-10-12)

- clang-19 / opt-19, OpenCL C++ 2021. The 610-floor CI runs the same sweep with
  clang-23; the numbers below are a clang-19 measurement and are not directly
  comparable to the floor.
- clspv: LLVM 24.0.0git with the m2v patches (the `clspv-bit2b` build).
- MLX sources: `mlx/backend/metal/kernels` at
  `1f8e74e3f12f31365464a6867c6579f0e9b29d85` ("Hold GIL in AttachedData
  destructor", #4391).

Fetch command:

```sh
git clone https://github.com/joshuaswarren/metal2vk.git metal2vk-mlxcore
git -C metal2vk-mlxcore checkout sweep/mlx-core2
git clone https://github.com/ml-explore/mlx.git scratch/sweep/mlx-batch-src/mlx
git -C   scratch/sweep/mlx-batch-src/mlx checkout 1f8e74e3f12f31365464a6867c6579f0e9b29d85
```

## MLX source license

MLX is MIT licensed:

> MIT License
>
> Copyright © 2023 Apple Inc.

The pinned tree is read at run time only. No MLX sources are committed here; every
entry names its upstream file and line in `results.tsv`.

## Run

```sh
cd metal2vk-mlxcore
export CLSPV=~/scratch/sweep/clspv-bit2b CLANG=clang-19 OPT=opt-19
export MLX_SRC=~/scratch/sweep/mlx-batch-src/mlx
export M2V_DEFS="-DM2V_MLX_TREE"
export M2V_CACHE_DIR=~/scratch/sweep/mlxcore-run/cache
nice -n 19 ionice -c 2 -n 7 python3 sweep/mlx-core/mlx_core_run.py \
    --mlx "$MLX_SRC" \
    --work ~/scratch/sweep/mlxcore-run/work \
    --out  ~/scratch/sweep/mlxcore-run/out \
    --jobs "$(nproc)" \
    --results sweep/mlx-core/results.tsv
```

`M2V_DEFS="-DM2V_MLX_TREE"` opts the shim's injected MLX prelude out: the real tree
carries its own `kernels/complex.h` through `utils.h`, so injecting the shim's ported
copy would make every declaration ambiguous.

`mlx_core_run.py` writes the manifest (one row per (work file, kernel name)) to
`work/manifest.json`; pass `--skip-expand` to reuse it on a re-run. The compile phase
is a thread pool of `tools/m2v-compile` subprocesses, one per entry, with the
content-addressed cache under `$M2V_CACHE_DIR`. The retry, crash-retry and refusal
machinery of `m2v-compile` is unchanged. Every produced module is re-validated with
`spirv-val --target-env vulkan1.3` before the row is written.

## Results (clang-19, this clspv build)

```
entries  17301
valid     1067   (compile ok AND spirv-val vulkan1.3)
failed   16234
```

### By kernel family

| family                         | valid / entries |
|---|---:|
| arange                         | 11 / 11 |
| arg_reduce                     | 10 / 24 |
| binary                         | 0 / 4122 |
| binary_two                     | 0 / 225 |
| conv                           | 0 / 18 |
| copy                           | 923 / 2496 |
| dot                            | 3 / 3 |
| fence                          | 0 / 3 |
| fft                            | 0 / 101 |
| fp_quantized                   | 0 / 480 |
| fp_quantized_nax               | 0 / 126 |
| gemv                           | 0 / 256 |
| gemv_masked                    | 0 / 528 |
| layer_norm                     | 0 / 12 |
| logsumexp                      | 0 / 6 |
| quantized                      | 0 / 2052 |
| quantized_nax                  | 0 / 864 |
| random                         | 0 / 2 |
| reduce                         | 0 / 2396 |
| rms_norm                       | 0 / 12 |
| rope                           | 0 / 18 |
| scaled_dot_product_attention   | 0 / 57 |
| scan                           | 0 / 456 |
| searchsorted                   | 26 / 26 |
| softmax                        | 4 / 10 |
| sort                           | 12 / 264 |
| steel_attn                     | 0 / 78 |
| steel_conv                     | 0 / 162 |
| steel_gemm                     | 0 / 1404 |
| ternary                        | 78 / 212 |
| unary                          | 0 / 877 |

### By cause (failed entries)

| cause                      | count |
|---|---:|
| shim_gap                   | 15031 |
| duplicate_host_name        | 768 |
| clang_frontend_exit        | 192 |
| excess_elements_init       | 144 |
| ambiguous_conditional      | 65 |
| attribute_list_position    | 18 |
| clspv_limit                | 14 |
| vector_element_count       | 2 |

### The classes

1. **shim_gap (15 031, 92.6% of failures).** One chokepoint, not a per-feature
   family: every MLX translation unit that includes a math op header pulls in
   `kernels/logging.h`, which declares `constant mlx::os_log logger(...)` at file
   scope. MLX's Metal 3.2+ branch aliases `mlx::os_log = metal::os_log`; the shim
   does not model Metal's logging API, so the alias is undeclared and every
   constructor call fails to match:

   ```
   binary_ops.h: error: no matching constructor for initialization of
     '__constant mlx::os_log'
   ```

   Until the shim models that API surface, every family that includes the op
   headers stays at zero: binary, reduce, quantized, steel_gemm, gemv, scan, fft,
   conv, rope, layer_norm, rms_norm, scaled_dot_product_attention, fp_quantized,
   quantized_nax, binary_two, unary, logsumexp, gemv_masked, fp_quantized_nax,
   random, fence.

2. **duplicate_host_name (768).** `steel_gemm_masked`'s macro table instantiates
   two template kernels under one generated `host_name`, so `m2v-compile --name`
   refuses the ambiguous selection:

   ```
   error: name 'steel_gemm_block_outmask_bfloat16_opmask_bfloat16_nn_bfloat16_
     bfloat16_bm32_bn32_bk16_wm2_wn2_MN_naligned_K_naligned' matches 2 entries
     or instantiations
   ```

3. **clang_frontend_exit (192).** clang exited nonzero without a usable
   diagnostic on bfloat-vector-heavy translation units (exit 139/134) during the
   cold-cache parallel pass. `M2V_CLSPV_TIMEOUT` bounds the wait; re-running the
   affected entries sequentially is the workaround.

   ```
   clang-19: error: clang frontend command failed with exit code 139
   ```

4. **excess_elements_init (144), ambiguous_conditional (65),
   attribute_list_position (18), vector_element_count (2).** Smaller parse
   shapes: initializer lists with more elements than the shim's vector-wrapper
   types accept (`implicit_gemm_conv_2d_*`, `steel_gemm_splitk_*_complex64`),
   conditionals the shim's bfloat conversions leave ambiguous
   (`sort_mbsort_complex64_*`), an attribute spelled where the shim's
   attribute grammar does not accept it (`depthwise_conv_2d_*`,
   `winograd_conv_2d_*`), and a vector shuffle with too few elements
   (`rbits`, `rbitsc` in `random.metal`).

5. **clspv_limit (14).** clspv's pointer-bitcast pass gives up on a shape it
   cannot rewrite; the driver catches the `M2V-GIVEUP: gepfromgep-skip` marker
   and keeps no SPIR-V, so a module that hit the marker is never silently
   wrong. This is the same class the uzu and TensorFold corpora name.

## What is not verified

- No GPU dispatch. The run produces and `spirv-val`s every module; it does not
  run any kernel. The companion `sweep/m2v_sweep_run.py` harness does that on a
  GPU host with synthetic data.
- No parity harness. No MLX reference outputs are compared.
- A clang-23 / CI-clspv measurement of this corpus has not been made; the
  failure mix above is what this toolchain produces.
