# Reduced repro: clspv emits an OpSelect spirv-val rejects

Status: fixed on `fix/nvfp4-select-wg` by `sweep/guarded_wg_load_select.py` (an IR pre-pass that
hoists the guarded Workgroup load into the branch block and turns the join phis into value
selects). `repro-results.txt` is regenerated post-fix: all 21 cases validate. The matrix below is
the pre-fix isolation record and its statuses are what the unpatched pipeline produces (skip the
pass with `M2V_KEEP_WG_GUARD=1` to reproduce it).

Upstream finding: `quantized_verify_nvfp4_argmax` (mlx-vlm@8f5dc3d,
mlx_vlm/models/quantized_verifier.py:1974) compiles through the m2v pipeline
(clang, opt, patched clspv) but `spirv-val --target-env vulkan1.3` rejects the
module:

    error: line N: Expected both objects to be of Result Type: Select
      %NN = OpSelect %_ptr_Workgroup__arr_uint_uint_4 %NN %NN %5

(clspv 32-bit target, --spv-version=1.5, Vulkan 1.3 validation.)

## Reproduce

```sh
source ~/scratch/m2v-ct/env.sh
tools/m2v-compile --msl sweep/mlx-batch/vlmb/repro/v20_single_token.metal \
  --out /tmp/m2v-repro --name v20_single_token --json   # exit 5, spirv-val failed
```

`v20_single_token.metal` is the smallest failing form (21 lines, no Metal-only
constructs beyond threadgroup/threadgroup_barrier). `full.metal` is the
original assembled kernel and fails identically.

## Trigger shape

A `threadgroup int` array whose element is read inside a data-dependent branch
that updates a loop-carried scalar, where the branch condition comes from a
value in a private (thread) array:

```c
threadgroup int group_indices[4];
float private_values[4];                 // condition source, thread storage
...
int best_index = group_indices[0];
for (int group = 1; group < 2; ++group) {
  float value = private_values[group];
  if (value > best_value) {
    best_value = value;
    best_index = group_indices[group];   // guarded threadgroup-array load
  }
}
```

Reading the threadgroup element outside the branch (`int index =
group_indices[group];` before the `if`) makes the module validate (v7, v10).
Replacing the threadgroup array with a private int array also validates (v12).
The IR pipeline stages are clang -O0, then
`opt -passes='cgscc(inline),function(sroa,early-cse,simplifycfg)'
-inline-threshold=100000`, then clspv; the pointer select that becomes the
rejected OpSelect reaches clspv's input already as a select over
workgroup-array pointers, so the malformed type pairing appears during clspv's
lowering of that select.

## Variant matrix (repro-results.txt, one line per case)

| file | kernel | status |
|---|---|---|
| full.metal | (original) | spirv-val failed |
| v1_no_scoring.metal | v1_no_scoring | spirv-val failed |
| v2_v3_v4.metal | v2_indices_only | ok |
| v2_v3_v4.metal | v3_values_only | ok |
| v2_v3_v4.metal | v4_flat_arrays | spirv-val failed |
| v5_v6_v7.metal | v5_no_cross_read | ok |
| v5_v6_v7.metal | v6_one_array_cross_read | spirv-val failed |
| v5_v6_v7.metal | v7_two_arrays_ifelse | ok |
| v8_v9_v10.metal | v8_min_guarded_tg_read | ok |
| v8_v9_v10.metal | v9_min_guarded_const | ok |
| v8_v9_v10.metal | v10_min_read_no_tg_update | ok |
| v11_v12.metal | v11_int_update_only | spirv-val failed |
| v11_v12.metal | v12_both_private_int | ok |
| v13_v14_v15.metal | v13_uncond_store | spirv-val failed |
| v13_v14_v15.metal | v14_guarded_store_plus_read | ok |
| v13_v14_v15.metal | v15_tg_condition | ok |
| v16.metal | v16_single_update_private_cond | spirv-val failed |
| v17_v18.metal | v17_min_private_cond_tg_update | ok |
| v17_v18.metal | v18_min_tg_cond_tg_update | ok |
| v19_v20.metal | v19_no_first_pass | spirv-val failed |
| v19_v20.metal | v20_single_token | spirv-val failed |

Reading of the axes:

- One threadgroup array is enough (v6, v16 fail with a single one); a second
  array is not required.
- The guarded branch must read the threadgroup array. Constant updates (v9),
  private-array updates (v12), or hoisting the threadgroup read out of the
  branch (v7, v10, v15) all validate.
- Tiny one-loop forms pass (v17, v18) even though their branch has the same
  guarded threadgroup read; the failure additionally needs the two-phase
  structure around a barrier (populate under one divergent if, reduce inside
  another) - v20 is the smallest failing combination of those.

Toolchain: LLVM 23.1.1 clang/opt, clspv from the m2v build, host x86-64 Linux.
No GPU execution involved; this is compile-plus-validation only.
