# Family vlmA: mlx-vlm first half

Every `mx.fast.metal_kernel` call site in the pinned mlx-vlm tree
(8f5dc3d) whose file sorts in the first half of the alphabetical list of
files containing such a call (13 of 26 files: bitnet/bitlinear.py through
nemotron_h/speculative_verifier.py). Sources are extracted from the pinned
tree with ast; the MSL assembly encodes MLX's own rules from
ml-explore/mlx@1f8e74e mlx/backend/common/metal_kernel.cpp
(write_signature, write_template, make_template_hash, the ordered
metal_attributes list, the kernel-name construction with the constant/scalar
dtype suffixes and max_constant_array_size 8).

bitnet/bitlinear.py defines the same `bitlinear_matmul` kernel already
assembled from mlx-lm in the parent results.tsv, so it is not repeated here.
inkling/language.py `inkling_banded_mask(_v2)`, structured.py
`mlx_vlm_llguidance_mask` and kernels.py `nearest_interpolation`,
`grid_sample`, `separable_interpolate` are likewise upstream rows; this
family adds the remaining call sites, including all eight flag variants of
the gated_delta.py generator.

## Exact commands

    # 1. assemble the 29 kernels (writes *.metal + kernels.json here)
    python3 assemble.py gen

    # 2. compile + validate each through the m2v pipeline (clang -> clspv ->
    #    spirv-val vulkan1.3) and write results-vlmA.tsv
    source ~/scratch/m2v-ct/env.sh
    python3 assemble.py check

    # single kernel by hand, same pipeline:
    #   tools/m2v-compile --msl sweep/mlx-batch/vlmA/<kernel>.metal \
    #     --out /tmp/<kernel>-out --json

`tools/m2v-compile` is the repo runner: exit 0 ok (the emitted .spv is
spirv-val clean), 3 refused, 4 compile error, 5 spirv-val failure.

## Result

29/29 assembled, compiled and spirv-val clean (results-vlmA.tsv). Each row
records the origin (repo@sha:file:line), the assembled-source sha256, the
representative shapes/dtypes with the reason for the choice, and the grid
and threadgroup of that representative call. No M2V-GIVEUP rows.

Dtype notes: bfloat16 representatives for the two flux2_norm_rope kernels
and float16 for the dtype-gated short_block/glm/minimax kernels match what
their call sites pass; everything else is float32. gated_delta's bool mask
and scalar int32 step count, and attnres_mix's scalar int32 N, exercise the
constant-buffer and scalar-argument assembly paths.
