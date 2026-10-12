# ggml corpus sweep

Runs every kernel entry of llama.cpp's ggml Metal sources through the metal2vk pipeline
(shim, clang, opt passes, patched clspv, spirv-val) and writes one row per entry to
`results.tsv`. This widens the corpus for the any-hardware vision: ggml's kernel
inventory is the second large Metal corpus (after uzu and TensorFold) the translator
is asked to cover. Compile only: no GPU dispatch, no numerical check.

## Pinned source

- Commit: `65840ed53c8653bfbf3e9014d9cbf71ac8c08725`, the commit the omarchy-mlx
  docs (BENCHMARKS.md, llamacpp-demo.md) cite for the llama.cpp numbers.
- Corpus: `ggml/src/ggml-metal/kernels/*.metal` plus the headers they include
  (`common.h`, `dequantize.h`, `quantize.h`, `fa_common.metal`, `fa_vec_common.metal`,
  `ggml-metal-impl.h`, `../ggml-common.h`). At this commit the old monolithic
  `ggml-metal.metal` no longer exists: the kernels live in the per-kind files.

Fetch (the script does this itself; shallow, read only, never committed here):

```sh
git init <src-dir> && git -C <src-dir> remote add origin https://github.com/ggml-org/llama.cpp
git -C <src-dir> sparse-checkout init --cone
git -C <src-dir> sparse-checkout set ggml/src
git -C <src-dir> fetch --depth 1 origin 65840ed53c8653bfbf3e9014d9cbf71ac8c08725
git -C <src-dir> checkout FETCH_HEAD
```

llama.cpp is MIT licensed: `Copyright (c) 2023-2026 The ggml authors`. The corpus is
not committed to this repository; `results.tsv` holds only kernel names, source lines
and compiler diagnostics from it.

## Entries

One row per `host_name` kernel entry: the `template [[host_name("N")]] ...`
instantiations (three spellings: `decltype(f<...>)`, `void f<...>`, and ggml's
`alias_t f<...>`, including the names the `MUL_MV_MMA_GEN` macro pastes) plus the
plain `[[host_name]]`-attributed kernels. The TUs are assembled exactly as
`ggml-metal/CMakeLists.txt` assembles them (shared headers concatenated, internal
include lines removed, `ggml-common.h` and `ggml-metal-impl.h` inlined), so every
entry compiles against the same text upstream feeds Apple's compiler.

## Run

Set `CLANG`, `OPT`, `CLSPV` (the patched clspv, built as CI builds it:
`CLSPV_SHA` from `.github/workflows/sweep.yml` plus every `clspv-patches` script in
the workflow's order) and `SPIRV_VAL`, then:

```sh
python3 sweep/ggml/run_ggml.py \
  --src ~/scratch/sweep/corpus-src/llama.cpp \
  --out ~/scratch/sweep/ggml-corpus/run \
  --results sweep/ggml/results.tsv \
  --jobs "$(nproc)"
```

Progress lands in `--out/progress.jsonl` (one JSON row per entry), so an interrupted
run resumes where it stopped; `--fresh` restarts. The script is read-only on the
corpus and writes only under `--out` and `--results`.

## Results

`results.tsv`: one row per entry: `name`, `source` (upstream file:line of the
host_name; macro-generated names point at the invocation line), `compile` (ok, error,
refused), `spirv_val` (`spirv-val --target-env vulkan1.3`: pass, fail, n/a),
`first_error`, `m2v_giveup` (the construct a refusal names, `-` otherwise), `cause`.

Run 2026-10-12 (baseline, no Metal matrix types in the shim), clang 23.1.1, clspv
`f2b01dd6` + the 20 `clspv-patches` scripts of that tree in CI order: **1019
entries, 103 valid, 916 failed** -- 908 `unsupported-type`, 7 `translator-bug`, 1
`shim-missing-feature`.

Run 2026-10-12 (after `metal::matrix<T, C, R>` and the float / half / bfloat
`NxM` typedefs land in the shim, plus a one-line `tparam_names` fallback so the
rewriter substitutes ggml's function-pointer template parameters): **1019
entries, 131 valid, 888 failed** -- 0 `unsupported-type` (the matrix wall is
gone), 827 `translator-bug`, 61 `shim-missing-feature`; every failing entry
still carries one of those classes (no `giveup` refusals fired: the corpus
stays off the Metal 4 tensor path, `GGML_METAL_HAS_TENSOR` undefined).

The 28 entries the matrix shim unblocks are the kernels whose first error used
to be the matrix type wall. Two of them exercise the new matrix API directly
(`lightning_indexer_f{16,32}` / `_q{4_0,5_0,8_0}` -- 5 entries, and the 14
`mul_mm_id_*_f{16,32}` quant variants that dequantise A-tiles into
`float4x4` / `bfloat4x4` / `half4x4` and reinterpret B-tiles as
`half2x4`); the remaining 9 are `mul_mm_id_map0_ne20_*` kernels, which never
name a matrix type in their own bodies but live in the same `mul_mm.metal` TU
whose earlier parse failure had been blocking every host_name instantiation
in the file. The 888 that still fail split into the classes below, and each
class is now small enough to fix in its own PR:

1. `translator-bug` (827), sub-breakdown by first error (counts from
   `results.tsv`):
   - 407 (`330 'C' + 77 'Q'`) `use of undeclared identifier` on the
     function-constant template arguments the entries read via
     `[[function_constant(int C)]]` / `[[function_constant(int Q)]]`. The
     rewriter substitutes the values for template parameters it knows
     (`tparam_names`), but `[[function_constant]]` is a kernel attribute,
     not a template parameter, and the rewriter's attribute model does not
     know the value. Two attributes cover the bulk of the corpus.
   - 231 `pointers to functions are not allowed`. The rewriter substitutes
     the function-pointer template parameter NAME in body calls (the
     `tparam_names` fallback that landed with this run), but the function
     pointer TYPE still appears in the emitted `.cl` -- inside a
     `decltype` or default-argument expression the rewriter does not
     rewrite -- and OpenCL C++ rejects the type at parse time.
   - 89 `no matching function for call to 'best_index_int8'`. The function
     lives in ggml's C++ runtime, not the Metal TU; the kernel relies on a
     device-side host bridge that this pipeline does not model.
   - 43 `invalid LLVM IR input: Instruction does not dominate all uses!`.
     The IR-shape bug the baseline already ranked on `kernel_soft_max_f32`
     and neighbours; the matmul inner loops' B-tile load is non-dominating.
   - 26 `OpPtrAccessChain result type ... does not match` (the
     `line NNNN:` spirv-val-format error, emitted by clspv when the typed
     GEP / pointer-shape pass rejects a struct-typed access). Concentrated
     in the `mul_mm_id` quant variants for `iq{1_s,2_s,2_xs,2_xxs,3_s,3_xxs,
     4_xs}` and `q{1_0,2_0,3_K,6_K}` / `tq2_0`. Compile produces a
     `compile=ok, spirv_val=fail` row.
   - 22 (`9 at 12:1 + 6 at 14:1 + 3 at 13:1 + 2 at 15:1 + 2 at 11:1`)
     `forward references to non-struct type`. A clang ordering edge in
     the rewriter output.
   - 6 `C-style cast from vector vec<float, 4> to vector half4 of
     different size`. Metal accepts the elementwise narrowing; OpenCL C++
     does not.
   - 2 `expected unqualified-id`. A rewriter edge.
   - 1 `cannot initialize a parameter`. A rewriter edge.
2. `shim-missing-feature` (61), sub-breakdown:
   - 60 `scalar operand type has greater rank than the type of the vector
     element. ('ushort2' and 'int')`. Metal promotes the scalar to the
     vector's element type; the affected OpenCL builtins (the
     `metal::max` / `metal::min` / `metal::fma` overloads) reject mixed
     rank. Adding a (vector, scalar) overload that promotes the scalar to
     the vector's element type for the specific builtins the corpus uses
     covers the bulk of these.
   - 1 `scalar operand type has greater rank than the type of the vector
     element. ('float' and 'half4')` (`kernel_soft_max_f16_4`): the same
     promotion, the case the baseline run reported alone.

## Not verified here

- Compile and validate only: a valid module says nothing about runtime behavior,
  performance, or numerical results on a GPU.
- The Apple Metal compiler's acceptance of the same source is not compared; "valid"
  means this pipeline's Vulkan 1.3 SPIR-V validates.
- `GGML_METAL_HAS_TENSOR` is left undefined (the default build): tensor-based code
  paths are absent from the corpus and untested.
- Template values come from the instantiation lines themselves; shapes that depend on
  runtime dispatch are not exercised.
