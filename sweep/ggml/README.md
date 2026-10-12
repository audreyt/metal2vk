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

Run 2026-10-12, clang 23.1.1, clspv `f2b01dd6` + the 20 `clspv-patches` scripts of
that tree in CI order: **1019 entries, 103 valid, 916 failed**: 908
`unsupported-type`, 7 `translator-bug`, 1 `shim-missing-feature`; every failing entry
carries one of those classes (no `giveup` refusals fired: the corpus stays off the
Metal 4 tensor path, `GGML_METAL_HAS_TENSOR` undefined).

The three causes, what fails and why:

1. `unsupported-type` (908): the shim has no Metal matrix types. First error:
   `unknown type name 'float4x4'; did you mean 'float4'?`: every entry of the 19
   kernel files whose live code names a matrix type (`float4x4`, `bfloat4x4`,
   `simdgroup_half8x8` template arguments of the mul_mm and flash-attn families;
   0 failures outside those files). A real fix is matrix semantics in the shim plus
   the clspv coopmat lowering that consumes them, not a typedef.
2. `translator-bug` (7): driver codegen artifacts on individual entries. First error:
   `error: invalid LLVM IR input: Instruction does not dominate all uses!`
   (`kernel_soft_max_f32` and two neighbours; the generated IR fails opt's verifier),
   plus `expected unqualified-id` (`kernel_snake_*`), a vector-to-scalar call the
   wrapper emits (`kernel_unary_f32_f32_4`) and an oversize C-style cast
   (`kernel_unary_f16_f16_4`).
3. `shim-missing-feature` (1): `kernel_soft_max_f16_4`: `scalar operand type has
   greater rank than the type of the vector element. ('float' and 'half4')`: Metal
   promotes the float scalar into the half4 expression, OpenCL C++ refuses.

## Not verified here

- Compile and validate only: a valid module says nothing about runtime behavior,
  performance, or numerical results on a GPU.
- The Apple Metal compiler's acceptance of the same source is not compared; "valid"
  means this pipeline's Vulkan 1.3 SPIR-V validates.
- `GGML_METAL_HAS_TENSOR` is left undefined (the default build): tensor-based code
  paths are absent from the corpus and untested.
- Template values come from the instantiation lines themselves; shapes that depend on
  runtime dispatch are not exercised.
