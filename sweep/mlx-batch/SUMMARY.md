# MLX custom-kernel compile batch - combined summary

This directory holds statically assembled MSL for the pinned MLX custom-kernel inventory: the
compile/validation runner, four family directories (this top level, `vlmA/`, `vlmb/`, `others/`),
per-family results tables, a combined `results-all.tsv`, and this summary. Each `.metal` source
names its upstream callsite and representative call contract in a header comment. The sources are
compile-tested only; no GPU dispatch is performed.

## Top-level counts (historical batch snapshot)

Historical assemble/compile batch as integrated (not a live re-measure):

| family | sources | valid (spirv-val vulkan1.3) | failed | results file |
|---|---:|---:|---:|---|
| top (mlx-lm, mlx-vlm helpers, mlx-audio) | 14 | 14 | 0 | `results.tsv` |
| vlmA (mlx-vlm first half) | 29 | 29 | 0 | `vlmA/results-vlmA.tsv` |
| vlmb (mlx-vlm second half) | 60 | 60 | 0 | `vlmb/results-vlmb.tsv` |
| others (mlx-lm / mlx-examples / mlx-audio) | 16 | 16 | 0 | `others/results-others.tsv` |
| **batch total (assembled)** | **119** | **119** | **0** | `results-all.tsv` |

Historical sibling combine (same snapshot era): existing 32-kernel corpus (TensorFold layout, 31 compile-valid, 1 needs M5 tensor ops): 119 + 31 = **150 valid**, 119 + 32 = **151 total**. Compiler-only `spirv-val` success does not prove Workgroup bounds or numerics.

**Unresolved:** `quantized_verify_nvfp4_argmax_t2_k64_n64_fp16` still fails with an invalid pointer `OpSelect` on the unpatched pipeline. Historical "119 valid / 0 failures" assumed the removed WG-load pre-pass (see Failures).

## Denominators

- 107 AST `metal_kernel()` callsites across the four pinned upstream trees (mlx-lm@2184db2: 11; mlx-examples@796f5b5: 3; mlx-vlm@8f5dc3d: 85; mlx-audio@17001a6: 8). This corrects the earlier draft's 106: `mlx-examples musicgen/encodec.py:14` is a real third mlx-examples callsite, not a duplicate of the census.
- The 119 rows cover 105 distinct callsites. The other 14 rows are extra variants out of the multi-variant generators: `mlx_lm/models/gated_delta.py:130` (4 bodies), `mlx_vlm/models/gated_delta.py:148` (8), `mlx_vlm/models/qwen3_5/gated_delta.py:109` (4), `qwen3_5/gated_delta.py:540` (2).
- Equation: 105 covered + 2 not assembled = 107 callsites (check). The 2 not assembled carry no new kernel body: `mlx-vlm bitnet/bitlinear.py:44` (the same `bitlinear_matmul` body as mlx-lm's, owned by the parent `results.tsv`) and `mlx-examples musicgen/encodec.py:14` (the same body as `encodec/encodec.py:14`, owned by `others`). The earlier draft's "103 assembled + 3 not-assembled = 106" guessed the gaps were the two mlx-examples lstm callsites; in fact both mlx-examples encodec sites are assembled and the real gaps are the duplicate-body sites.
- Result-row source sha256: 119 distinct (no two assembled rows share bytes). The `vlmb/repro/` isolation files are not result rows; `repro/full.metal` repeats the nvfp4 source verbatim by design.
- The wwdc25 notebook callsite is counted once (file-level; one `metal_kernel()` call in one code cell).

## Failures and classification

| kernel | status | cause |
|---|---|---|
| `nemotron_bm32_steel_linear_nt` | fixed on this branch | missing prelude. The source expands the MLX `steel/gemm` headers, which reference `complex64_t` from MLX's device-side prelude (`kernels/complex.h`), and a standalone kernel file never sees it. The shim now injects that prelude (`include/mlx_prelude.h`, ported from the pinned mlx header, plus `metal::is_convertible_v` in `include/metal_stdlib`). Row is `yes`. |
| `quantized_verify_nvfp4_argmax_t2_k64_n64_fp16` | **unresolved** | clspv lowers a guarded Workgroup-array load (LLVM branch + phi join) into a select over differently typed Workgroup pointers (element pointer vs whole `[2 x [2 x uint]]` variable) and emits that pointer select with one inferred pointee type, so spirv-val rejects at vulkan1.3 ("Expected both objects to be of Result Type: Select"). Pinned evidence: incoming LLVM IR is branch/phi (no pointer selects); `-O0` validates clean; `--spv-version=1.6` does not help. Reduced repro + 21-case matrix: `vlmb/repro/`. Former pre-pass `sweep/guarded_wg_load_select.py` (and `M2V_KEEP_WG_GUARD`) is removed: it hoisted the load past the guard and OOB-read Workgroup memory — Trellis `partials[lane]` outside `lane < 16` on a 16-element WG array; Weaver the same with `lane < 8`. Without it, nvfp4 is still the invalid pointer `OpSelect`. Historical compile-`yes` was not a bounds/numerics proof. |

## Integration provenance

- The `others` family was ported from the coreglass worktree branch `feat/mlx-batch-others` (ca76804), where it had been committed to the wrong repository; this repository owns `sweep/mlx-batch` and `tools/m2v-compile`.
- One row was dropped in the port: `mlx_lm/tuner/losses.py:168` (`kl_forward`) was already assembled in the parent `results.tsv`, and the no-duplicate-rows rule keeps the parent row (the vlmA-family convention). `others` carries 16 bodies over 13 callsites.
- KernelBatch's reduced repro for the nvfp4 failure (`feat/mlx-batch-vlmb` c8182cc) is merged under `vlmb/repro/`.
- `vlmb/specs/ssm_kernel.json`: the `module` field is now the runnable repo-relative pin (`mlx-vlm/mlx_vlm/models/ssm.py`). The earlier rewrite had put the provenance string (`repo@sha:path`) in the path field, which `assemble.py` cannot open; the provenance stays in the `origin` field.
- `vlmb/assemble.py` normalizes the wide dash at emit time, so regenerating the family reproduces every committed `.metal` byte for byte (60 of 60 verified on this branch).
- `vlmb/compile_all.sh` now captures the JSON line explicitly and its status pattern accepts `spirv-val failed`; the old `tail -1` capture recorded `no-json` whenever stderr trailed the JSON.
- `vlmb/README.md`: the results path points at `results-vlmb.tsv` in this directory.

## Re-run

**Historical** (2026-10-12, WG-load pre-pass still wired): 119 rows, 119 valid, 0 failures. Without that pass, nvfp4 is invalid pointer `OpSelect` again. To re-validate a family:

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

`run_batch.py` reads the `// Origin:` and `// Representative call:` header comments, compiles each `.metal` with `tools/m2v-compile`, validates the produced SPIR-V with `spirv-val --target-env vulkan1.3`, and writes a per-row result line. It scans `--msl-dir` non-recursively, so the top-level call (`--msl-dir sweep/mlx-batch`) compiles exactly the 14 top rows and a family call compiles exactly that family's own sources (the nested `vlmA/`, `vlmb/`, `others/`, `vlmb/repro/` directories keep their own tables and runners). The `vlmA`, `vlmb` and `others` families document their own assembler commands in their READMEs; `vlmb` regenerates byte-identically from its committed specs. Compile-only validity still does not prove Workgroup bounds or GPU numerics.

## CI coverage floor

`sweep/check_coverage.py` is keyed to one sweep `RESULT.json` floor and does not support a second corpus directory without changing the `610` floor that the existing sweep enforces. The batch is therefore not added to the CI coverage floor. A second-corpus support change is out of scope for this task and the task forbids touching the floor.

### Removed guarded-WG-load pre-pass

`sweep/guarded_wg_load_select.py` / `M2V_KEEP_WG_GUARD` removed (bounds reason under Failures: Trellis/Weaver OOB). **Historical** (2026-10-12, uzu c92723bd + TensorFold f8fe17d2, `--set all`): 591/608 saved IRs byte-identical through the pass; **17** entries over 11 sources rewritten (`uzu`: trellis, gdn chunked output-and-state, gdn norm-gate, gdn tree-verify out, gdn update, normalization, radix-top-k-small, unified-sampling, weaver top-children; `tf`: moe-route route-select, argmax bf16-part) still reported compile `val=ok`. Volatile WG loads were non-matches (`sweep/tests/wg_volatile_skip.ll`). Flake note: `tf/kimi/dense.metal` `k3_rows_bf16_r4/r8` hit a transient parallel clang crash on one skipped run and compiled clean on re-run. Those historical rewrite/`val=ok` counts are not a bounds or numerics proof — the 17 included the Trellis/Weaver OOB shapes.

## Source review

- Reviewer grep: `git grep -n -i -E '/home/|jw16|jwm1|macstudio|gpu-turn' -- sweep/mlx-batch` is clean of live hits; the only matches are this summary's own quotes of the rule.
- The four `vlmb/*.metal` wide-dash comment fixes are now produced by the assembler itself, so regeneration cannot reintroduce them.
- No GPU run was performed or claimed.
