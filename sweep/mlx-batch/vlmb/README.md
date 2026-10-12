# vlmB: mlx-vlm metal_kernel sweep, second alphabetical half

Family `vlmB` of the mlx-batch sweep: every `mx.fast.metal_kernel()` call site
in the pinned mlx-vlm tree (`mlx-vlm@8f5dc3d`) whose containing file sorts in
the second half of the alphabetical file list (26 files, even split; files
14-26 are ours). Every call site is assembled into the complete MSL MLX would
generate, then checked through `tools/m2v-compile` (clang, opt, patched clspv)
plus `spirv-val --target-env vulkan1.3`.

Coverage: 60 assembled kernels across 11 files. Two more files in our half have
no rows. `mlx_vlm/structured.py` holds the single kernel KernelBatch already
carried as `mlx_vlm_llguidance_mask`, so it is not redone here.
`mlx_vlm/tests/test_models.py` mentions metal_kernel only in prose and has no
call sites.
Results: `results-vlmb.tsv` (this directory).

## Commands

```sh
source ~/scratch/m2v-ct/env.sh          # clang, opt, patched clspv (MLX_BATCH_SRC_ROOT
                                        # points assemble.py at the pinned trees)
python3 vlmb/make_specs.py              # emit one spec JSON per kernel (vlmb/specs/)
for f in vlmb/specs/*.json; do python3 vlmb/assemble.py "$f"; done
M2V_TOOL=$PWD/../..'/tools/m2v-compile' \
  ionice -c3 nice -n 19 sh vlmb/compile_all.sh
python3 vlmb/make_results.py            # merge into ../results-vlmb.tsv
```

Each `assemble.py` run writes `<kernel name>.metal` (header comment carries the
origin and the representative call) and prints the file sha256 that lands in
the results TSV. Compiled artifacts land in `vlmb/out/` only for clean
compiles.

## How assembly works

`vlmb/assemble.py` executes the pinned upstream module with a recording fake
`mlx.core` (no MLX install needed), calls each kernel factory exactly as the
dispatcher would, and captures `name`, `input_names`, `output_names`, `header`
and `source` verbatim - including sources the call sites build at runtime
(f-strings, `str.replace`, `"\n".join` line builders, the expanded MLX steel
GEMM header stack for `nemotron_bm32_steel_linear_nt`). The signature writer is
a port of `mlx/backend/common/metal_kernel.cpp write_signature`: buffer order,
`_shape`/`_strides`/`_ndim` buffers when the source mentions them, `constant`
address space for inputs under 8 elements plus the `c`/`s` kernel-name
suffixes, the 20 metal attribute arguments, and the
`template [[host_name(...)]] [[kernel]] decltype(...)` instantiation.

Spec JSONs hold the representative shapes, dtypes, template values and launch
geometry, read off each dispatcher in the pinned tree. The per-input dtypes are
the ones the dispatchers produce (float16 activations, uint32 packed weights,
float32 scales/codebooks/epilogues, int32 indices/masks, bool token masks).

## Notes

- The steel GEMM kernel spec needs the pinned mlx source tree next to mlx-vlm
  under the pinned root: `assemble.py` builds a throwaway
  `include/mlx -> <pinned mlx>/mlx` shim so
  `_load_mlx_steel_gemm_header()` expands the real headers (about 55 KB) into
  the file.
- Result columns follow `results.tsv` from the first batch, plus explicit
  `compile` / `spirv_val` / `failure_cause` columns (m2v-compile gates
  spirv-val behind a clean compile, so a compile failure leaves the val column
  empty rather than "fail").
