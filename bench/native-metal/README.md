# Native Metal timing harness

This harness compiles the flattened uzu kernels at run time with Metal. It uses the same shapes, grids, buffer order, constants, iteration sizing, and CPU checks as `bench/m2v-bench.py`. It needs a Mac with a Metal device and `swiftc` from Xcode Command Line Tools or Xcode. It does not need the `metal` command-line compiler or a uzu checkout.

## Run

From this directory, run:

```sh
bash run.sh
```

The script prints the chip, macOS version, and load average, then builds and runs the harness. Results go to `results/results-native-metal-YYYY-MM-DD.json`. Pass an output directory as the first argument to choose another location:

```sh
bash run.sh /path/to/output
```

Each row reports effective microseconds per dispatch, GPU time per dispatch from the batch command buffer's `GPUStartTime` and `GPUEndTime`, iteration count, maximum CPU-reference error, tolerance, and throughput. The batch encodes all dispatches into one command buffer with a buffer memory barrier after each dispatch, then times the submit, matching the translated runner. The batch uses the same pilot-and-cap rule as `m2v-bench.py`: `min(500, 2,000,000 / pilot wall microseconds)`, with at least one iteration. For fewer than 50 iterations, reported time is the larger of GPU time and wall time per dispatch. For 50 or more iterations, it is wall time per dispatch. The check field says `OK` only when error is below the translated-side tolerance. Any failed check makes the process exit nonzero.

The native row names match the translated JSON keys: `activation_f32`, `activation_f16`, `softmax_f32`, `softmax_f16`, `tilematmul_f32_256`, `tilematmul_f32_1024`, and `rt2x2_256` through `rt4x4_1024`. The JSON also records GPU time, wall time, iteration count, error, tolerance, and metric value for each row.

## Ticket for the macOS window

1. Copy the `bench/native-metal/` directory to the Mac.
2. Open Terminal in the copied directory and run `bash run.sh`.
3. Keep the Mac quiet during the run. Do not run another GPU workload. The display does not need to sleep.
4. Expect about 10 seconds on the Mac Studio. Allow 15 to 60 seconds on the 16-inch M1 Max window for compilation, all 12 rows, and the double-precision matmul references.
5. Read the printed table. Every check field must say `OK`; the process exits nonzero if any check fails.
6. Find the JSON file under `results/results-native-metal-YYYY-MM-DD.json`. Keep the table and JSON together with the device, macOS version, and load average when sharing the measurement.
