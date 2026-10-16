# Muse fused Metal kernels: translate, run and compare

The Muse decoder calls fused Metal kernels through `mx.fast.metal_kernel`.
This directory checks one thing. Does the metal2vk translation of those kernels give the same numbers as native Metal?
It has three parts: a builder, a runner and a comparer.

## The kernels

There are twelve kernels. They come from two sources that are not in this repository.
Fetch them read-only into a directory `SRC`:

- `SRC/mlxlm/*.py`: `mlx_lm/kernels/` from ml-explore/mlx-lm at commit `06e03f7ce7c059ad105483a9dded0bb338cbaf53`.
- `SRC/omlx_norm_kernels.py`: `omlx/patches/muse_spark/norm_kernels.py` from jundot/omlx (Apache-2.0).

## Build the AOT modules (toolchain host, no GPU)

```sh
source $M2V_CT/env.sh
export M2V_COMPILE=$PWD/tools/m2v-compile M2V_CACHE_DIR=$PWD/.m2v-cache
python3 bench/mlx-muse/muse_corpus.py build SRC OUT
```

`OUT/aot` holds one SPIR-V module and one reflection file per instantiation. Each is named by the key omarchy-mlx looks up.
The command exits 1 if any instantiation fails to compile.
All 21 representative instantiations compile. All 21 pass `spirv-val --target-env vulkan1.3`.

## Run the kernels (any host with MLX)

`qmuse.py` runs the pinned wrapper code unmodified. It checks every output against a float32 reference from the CPU stream.
It can also write a dump for the comparer.
Inputs come from numpy and are seeded from the case name. Every platform draws the same bytes.

macOS MLX, native Metal. It needs `mlx`, `numpy` and `SRC`:

```sh
python3 bench/mlx-muse/qmuse.py --src SRC --route native --cases omlx_moe_tail_AXIS8192,qk_norm_rope_D128_HQ32_HK8_os1 \
    --tokens 16,64 --dump macos.npz
```

Linux omarchy-mlx, metal2vk route. It needs a wheel that contains the route, `numpy`, `SRC` and `OUT/aot` from the build step:

```sh
python3 bench/mlx-muse/qmuse.py --src SRC --route m2v --aot OUT/aot --cases omlx_moe_tail_AXIS8192,qk_norm_rope_D128_HQ32_HK8_os1 \
    --tokens 16,64 --dump linux.npz 2> route.err
grep -q 'route summary: m2v=[0-9]* translator=0 fallbacks=0 refusals=0' route.err
```

The last line must succeed. If it fails, some kernel did not take the metal2vk route. The dump is then not a translated run.
The default `--cases` runs all twelve kernels (21 instantiations).
The two named above move the most bytes at T = 64 (10.5 MB each). They use different features.
`omlx_moe_tail` uses `precise::rsqrt` and a threadgroup reduction.
`qk_norm_rope` uses `simd_shuffle_xor`, `fast::cos` and `fast::sin`.

## Compare the two dumps (CPU, no GPU)

```sh
python3 bench/mlx-muse/qmuse_compare.py macos.npz linux.npz
```

For each case and output, the table shows three numbers. They are the largest row error between the platforms, the fraction of bit-identical elements and the largest absolute difference.
It also shows each side against its own CPU reference. It shows the two references against each other too.
A case agrees when three errors are within `--tol` (default 2e-2): platform A against B, A against its reference, and B against its reference.
An output that the runner checks tighter, such as an index output, is compared as tight.
The exit code is 0 when every case agrees. It is 1 when one does not. It is 2 when the dumps are not comparable.
Dumps are not comparable when the case sets, the input hashes or the shapes differ.
They are also not comparable when the two CPU references differ by more than the tolerance.
`qmuse.py` itself exits 1 when any case is a mismatch or an error, or when no case ran.

## Limits

- The numbers show agreement on random inputs at the dimensions in `muse_corpus.py`. They say nothing about model quality.
- Muse's real configuration is gated on Hugging Face. The dimensions are representative values.
- A one step difference in bf16 is expected where the platforms round a reduction differently. The tolerance allows it.
- `fast::cos` and `fast::sin` can differ between Metal and a Vulkan driver by more than one unit in the last place.

## Test

```sh
python3 -m unittest bench/mlx-muse/test_qmuse.py bench/mlx-muse/test_qmuse_compare.py
```

`test_qmuse.py` needs an MLX install. It is skipped without one.
