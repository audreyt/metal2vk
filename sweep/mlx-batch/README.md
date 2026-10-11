# MLX custom-kernel compile batch

This directory holds assembled MSL from the pinned MLX custom-kernel inventory. Each source names its upstream callsite and representative call contract in comments. The selected shapes and template values make the source reproducible. This run tests compilation only. It does not dispatch kernels on a GPU.

## Run

Set `CLSPV` to the patched clspv binary. Set `SPIRV_VAL` to the SPIR-V Tools validator. Then run:

```sh
source /path/to/toolchain/env.sh
python3 sweep/mlx-batch/run_batch.py \
  --msl-dir sweep/mlx-batch \
  --tool tools/m2v-compile \
  --out "$M2V_OUT" \
  --cache "$M2V_CACHE_DIR" \
  --clspv "$CLSPV" \
  --spirv-val "$SPIRV_VAL" \
  --results sweep/mlx-batch/results.tsv
```

The runner compiles each `.metal` file, validates each output with `spirv-val --target-env vulkan1.3`, and writes the source SHA256 and first failure line to `results.tsv`. Exit status 0 means every source in the selected directory compiled and validated. A Metal 4 tensor-op refusal appears in the `m2v_giveup` column.

The source comments record the input and output shapes, dtypes, template values, grid, and threadgroup used for each representative call. These values describe compile inputs. They do not establish numerical correctness or GPU behavior.
