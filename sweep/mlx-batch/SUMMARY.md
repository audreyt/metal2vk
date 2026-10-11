# MLX custom-kernel compile batch - combined summary

This directory holds statically assembled MSL for the pinned MLX custom-kernel inventory: the
compile/validation runner, four family directories (this top level, `vlmA/`, `vlmb/`, `others/`),
per-family results tables, a combined `results-all.tsv`, and this summary. Each `.metal` source
names its upstream callsite and representative call contract in a header comment. The sources are
compile-tested only; no GPU dispatch is performed.

## Top-level counts (this batch only)

| family | sources | valid (spirv-val vulkan1.3) | failed | results file |
|---|---:|---:|---:|---|
| top (mlx-lm, mlx-vlm helpers, mlx-audio) | 14 | 14 | 0 | `results.tsv` |
| vlmA (mlx-vlm first half) | 29 | 29 | 0 | `vlmA/results-vlmA.tsv` |
| vlmb (mlx-vlm second half) | 60 | 59 | 1 | `vlmb/results-vlmb.tsv` |
| others (mlx-lm / mlx-examples / mlx-audio) | 16 | 16 | 0 | `others/results-others.tsv` |
| **batch total (assembled)** | **119** | **118** | **1** | `results-all.tsv` |

Combined with the existing 32-kernel sibling corpus (TensorFold layout, 31 compile-valid, 1 needs M5 tensor ops): 118 + 31 = **149 valid**, 119 + 32 = **151 total**.

## Denominators

- 107 AST `metal_kernel()` callsites across the four pinned upstream trees (mlx-lm@2184db2: 11; mlx-examples@796f5b5: 3; mlx-vlm@8f5dc3d: 85; mlx-audio@17001a6: 8). This corrects the earlier draft's 106: `mlx-examples musicgen/encodec.py:14` is a real third mlx-examples callsite, not a duplicate of the census.
- The 119 rows cover 105 distinct callsites. The other 14 rows are extra variants out of the multi-variant generators: `mlx_lm/models/gated_delta.py:130` (4 bodies), `mlx_vlm/models/gated_delta.py:148` (8), `mlx_vlm/models/qwen3_5/gated_delta.py:109` (4), `qwen3_5/gated_delta.py:540` (2).
- Equation: 105 covered + 2 not assembled = 107 callsites (check). The 2 not assembled carry no new kernel body: `mlx-vlm bitnet/bitlinear.py:44` (the same `bitlinear_matmul` body as mlx-lm's, owned by the parent `results.tsv`) and `mlx-examples musicgen/encodec.py:14` (the same body as `encodec/encodec.py:14`, owned by `others`). The earlier draft's "103 assembled + 3 not-assembled = 106" guessed the gaps were the two mlx-examples lstm callsites; in fact both mlx-examples encodec sites are assembled and the real gaps are the duplicate-body sites.
- Result-row source sha256: 119 distinct (no two assembled rows share bytes). The `vlmb/repro/` isolation files are not result rows; `repro/full.metal` repeats the failing nvfp4 source verbatim by design.
- The wwdc25 notebook callsite is counted once (file-level; one `metal_kernel()` call in one code cell).

## Failures and classification

| kernel | status | cause |
|---|---|---|
| `nemotron_bm32_steel_linear_nt` | fixed on this branch | missing prelude. The source expands the MLX `steel/gemm` headers, which reference `complex64_t` from MLX's device-side prelude (`kernels/complex.h`), and a standalone kernel file never sees it. The shim now injects that prelude (`include/mlx_prelude.h`, ported from the pinned mlx header, plus `metal::is_convertible_v` in `include/metal_stdlib`). Row is `yes`. |
| `quantized_verify_nvfp4_argmax_t2_k64_n64_fp16` | open, clspv optimizer bug | the translated module uses `OpSelect` whose two operands are differently typed Workgroup pointers (an element pointer and the whole `[2 x [2 x uint]]` variable), which spirv-val rejects at vulkan1.3. Evidence: the incoming LLVM IR has no pointer selects at all; clspv at its default optimization emits the bad select; `-O0` validates clean; `--spv-version=1.6` does not help. Reduced repro and a 21-case isolation matrix: `vlmb/repro/` (`repro-results.txt`). Source-level escapes that validate: hoisting the guarded threadgroup reads out of the branch, or a private array. The real fix is a clspv pass patch (or a pipeline restructure pass); out of scope for this batch. The row stays failing with the cause recorded. |

## Integration provenance

- The `others` family was ported from the coreglass worktree branch `feat/mlx-batch-others` (ca76804), where it had been committed to the wrong repository; this repository owns `sweep/mlx-batch` and `tools/m2v-compile`.
- One row was dropped in the port: `mlx_lm/tuner/losses.py:168` (`kl_forward`) was already assembled in the parent `results.tsv`, and the no-duplicate-rows rule keeps the parent row (the vlmA-family convention). `others` carries 16 bodies over 13 callsites.
- KernelBatch's reduced repro for the nvfp4 failure (`feat/mlx-batch-vlmb` c8182cc) is merged under `vlmb/repro/`.
- `vlmb/specs/ssm_kernel.json`: the `module` field is now the runnable repo-relative pin (`mlx-vlm/mlx_vlm/models/ssm.py`). The earlier rewrite had put the provenance string (`repo@sha:path`) in the path field, which `assemble.py` cannot open; the provenance stays in the `origin` field.
- `vlmb/assemble.py` normalizes the wide dash at emit time, so regenerating the family reproduces every committed `.metal` byte for byte (60 of 60 verified on this branch).
- `vlmb/compile_all.sh` now captures the JSON line explicitly and its status pattern accepts `spirv-val failed`; the old `tail -1` capture recorded `no-json` whenever stderr trailed the JSON.
- `vlmb/README.md`: the results path points at `results-vlmb.tsv` in this directory.

## Re-run

Fresh end-to-end re-run of all four families on this merged tree (2026-10-11): 119 rows, 118 valid, 1 open failure (the clspv OpSelect bug above). To re-validate a family:

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

`run_batch.py` reads the `// Origin:` and `// Representative call:` header comments, compiles each `.metal` with `tools/m2v-compile`, validates the produced SPIR-V with `spirv-val --target-env vulkan1.3`, and writes a per-row result line. It scans `--msl-dir` non-recursively, so the top-level call (`--msl-dir sweep/mlx-batch`) compiles exactly the 14 top rows and a family call compiles exactly that family's own sources (the nested `vlmA/`, `vlmb/`, `others/`, `vlmb/repro/` directories keep their own tables and runners). The `vlmA`, `vlmb` and `others` families document their own assembler commands in their READMEs; `vlmb` regenerates byte-identically from its committed specs.

## CI coverage floor

`sweep/check_coverage.py` is keyed to one sweep `RESULT.json` floor and does not support a second corpus directory without changing the `610` floor that the existing sweep enforces. The batch is therefore not added to the CI coverage floor. A second-corpus support change is out of scope for this task and the task forbids touching the floor.

## Source review

- Reviewer grep: `git grep -n -i -E '/home/|jw16|jwm1|macstudio|gpu-turn' -- sweep/mlx-batch` is clean of live hits; the only matches are this summary's own quotes of the rule.
- The four `vlmb/*.metal` wide-dash comment fixes are now produced by the assembler itself, so regeneration cannot reintroduce them.
- No GPU run was performed or claimed.
