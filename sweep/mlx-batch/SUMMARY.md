# MLX custom-kernel compile batch - combined summary

This directory holds statically assembled MSL for the pinned MLX custom-kernel inventory, the compile/validation runner, per-family results tables, and a top-level summary. Each `.metal` source names its upstream callsite and representative call contract in a header comment. The sources are compile-tested only; no GPU dispatch is performed.

## Top-level counts (this batch only)

| family | sources | valid (spirv-val vulkan1.3) | failed | results file |
|---|---:|---:|---:|---|
| top (mlx-lm, mlx-vlm helpers, mlx-audio) | 14 | 14 | 0 | `results.tsv` |
| vlmA (mlx-vlm first half) | 29 | 29 | 0 | `vlmA/results-vlmA.tsv` |
| vlmb (mlx-vlm second half) | 60 | 58 | 2 | `vlmb/results-vlmb.tsv` |
| others (mlx-lm / mlx-examples / mlx-audio) | not yet landed |  |  |  |
| **batch total (assembled)** | **103** | **101** | **2** |  |

Combined with the existing 32-kernel sibling corpus (TensorFold layout, 31 compile-valid, 1 needs M5 tensor ops): 101 + 31 = **132 valid**, 103 + 32 = **135 total**.

## Denominators

- 106 AST `metal_kernel()` callsites across the four pinned upstream trees (mlx-lm@2184db2: 11; mlx-examples@796f5b5: 2; mlx-vlm@8f5dc3d: 85; mlx-audio@17001a6: 8).
- Distinct assembled-source sha256 count across `sweep/mlx-batch/**.metal`: 103 (each file is a unique variant of its callsite; no two files share bytes).
- ~95 unique kernel bodies (inventory's context figure, after deduping identical copies shared across repos).
- Equation: 103 assembled + 3 not-assembled = 106 AST callsites (check). The 3 not-assembled are the 2 mlx-examples `lstm` callsites plus 1 in the missing "others" lane; the precise identification lands when the `others` branch merges.

## Failures and classification

| kernel | first error | cause class |
|---|---|---|
| `nemotron_bm32_steel_linear_nt` | `use of undeclared identifier 'complex64_t'` | missing prelude - the source expands the MLX `steel/gemm` headers, which depend on the MLX device-side `kernels/utils.h` prelude; the metal2vk shim does not yet inject it. Needs shim work, not a per-kernel fix. |
| `quantized_verify_nvfp4_argmax_t2_k64_n64_fp16` | `Expected both objects to be of Result Type: Select (OpSelect %_ptr_Workgroup__arr__arr_uint_uint_2_uint_2)` | clspv / spirv-val - the translated module uses `OpSelect` whose operands are workgroup-array pointers; spirv-val rejects the result type. This is a translator/clspv bug class, not an MSL-source defect. |

No shim or clspv pass change was made by this batch. The two failures are documented for the integrating lane; fixing them is out of scope for the compile sweep.

## Re-run

See the per-family table. To re-validate a family:

```sh
source /path/to/toolchain/env.sh
python3 sweep/mlx-batch/run_batch.py \
  --msl-dir sweep/mlx-batch/<family> \
  --tool tools/m2v-compile \
  --out "$M2V_OUT" \
  --cache "$M2V_CACHE_DIR" \
  --clspv "$CLSPV" \
  --spirv-val "$SPIRV_VAL" \
  --results sweep/mlx-batch/<family>/results-<family>.tsv
```

`run_batch.py` reads the `// Origin:` and `// Representative call:` header comments, compiles each `.metal` with `tools/m2v-compile`, validates the produced SPIR-V with `spirv-val --target-env vulkan1.3`, and writes a per-row result line.

## CI coverage floor

`sweep/check_coverage.py` is keyed to one sweep `RESULT.json` floor and does not support a second corpus directory without changing the `610` floor that the existing sweep enforces. The batch is therefore not added to the CI coverage floor. A second-corpus support change is out of scope for this task and the task forbids touching the floor.

## Source review

- Reviewer grep: `git grep -n -i -E '/home/|jw16|jwm1|macstudio|gpu-turn|—|–' -- sweep/mlx-batch` is clean.
- `vlmb/specs/ssm_kernel.json` originally carried an absolute `/home/...` path; rewritten to a repo-relative pin (`mlx-vlm@8f5dc3d:mlx_vlm/models/ssm.py`).
- Four `vlmb/*.metal` files had em dashes in comments; replaced with ` - ` to satisfy the repo rule. The 4 corresponding `source_sha256` rows in `vlmb/results-vlmb.tsv` were recomputed and patched.
- No GPU run was performed or claimed.
