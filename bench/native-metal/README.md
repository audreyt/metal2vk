# Native Metal timing harness

This harness compiles the flattened uzu kernels at run time with Metal. It uses the same shapes, grids, buffer order, constants, iteration sizing, and CPU checks as `bench/m2v-bench.py`. It needs a Mac with a Metal device and `swiftc` from Xcode Command Line Tools or Xcode. It does not need the `metal` command-line compiler or a uzu checkout.

The kernel sources come from uzu and keep its MIT license. The full text is in [LICENSE-uzu](LICENSE-uzu).

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
4. A run takes about 10 seconds (that figure is the build smoke test on the Mac Studio; the Studio serves live models, and no timed run is approved there). Allow 15 to 60 seconds on the 16-inch M1 Max window for compilation, all 12 rows, and the double-precision matmul references.
5. Read the printed table. Every check field must say `OK`; the process exits nonzero if any check fails.
6. Find the JSON file under `results/results-native-metal-YYYY-MM-DD.json`. Keep the table and JSON together with the device, macOS version, and load average when sharing the measurement.

## Pair with a Linux run

`compare.py` prints the first translated-vs-native table for one chip. It reads one or more native JSONs and one or more translated JSONs. Per case it reports the median and the spread (max minus min) over the runs. Both sides are wall per dispatch: the columns are native wall us, translated us, the translated/native ratio on that same wall basis, native GPU us (context only), and the MLX hand-kernel time. On the activation kernel the native wall and GPU times sit far apart, so the basis matters. It refuses, with a message and a nonzero exit, on any of these: chips that differ, a case on one side only, a check that is not `OK`, a native JSON without a `load` or `device` field. `--native-load` supplies the macOS load1 the window logged, and `--native-meta chip=... macos=... thermal=...` supplies or overrides the native header fields; a missing `load` is fatal only when neither the JSON nor `--native-load` has it. It prints a loud `WARNING` line when a recorded load average is above 0.5 on either side.

One chip means one host on each side: for example the 16-inch M1 Max (native, macOS) against the same 16-inch M1 Max running Linux, or the 14-inch M2 Max (native, macOS) against the g14c translated runs. The native run of 2026-10-11 and the g14c runs of the same date are different chips and must never be compared. No Mac Studio number exists and none is planned; the Studio served only the build smoke test.

macOS side (16-inch M1 Max window), after the machine has sat idle for 5 minutes past boot:

```sh
uptime | tee system.txt
bash run.sh && cp results/results-native-metal-*.json results/native-run1.json
bash run.sh && cp results/results-native-metal-*.json results/native-run2.json
bash run.sh && cp results/results-native-metal-*.json results/native-run3.json
```

The native JSON has no `load` field today. Read the first load number from the `uptime` line and pass it as `--native-load` in the pairing step. The dated filename is overwritten by every run, so rename the JSON between runs or you keep only the last run. Three back-to-back runs are the minimum for a median and a spread. The 5-minute settle exists because the 2026-10-11 run started 36 seconds after boot. Its load average was 16.00. Those numbers are load-affected and are not quiet-machine numbers. The Mac must stay quiet for the whole run. No other GPU workload, no builds, no model serving.

Linux side (14-inch M2 Max), same quiet discipline, one governor for the whole session:

```sh
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
cat "$(dirname "$M2V_ICD")/mesa-git-sha"
PATH=$HOME/bin:$PATH <gpu-queue-wrapper> -m 25 -- bash bench/run-all.sh /tmp/m2v-out --device g14c
```

The ICD directory holds `mesa-git-sha` next to the driver library and the ICD JSON. The queue wrapper caps the run; the repo's `docs/bench-status.md` documents the form.

Then, from `bench/native-metal/`:

```sh
python3 compare.py \
  --native results/native-run1.json results/native-run2.json results/native-run3.json \
  --linux ../results-<device>-<date>.json \
  --native-load 0.12 --native-meta thermal="Nominal" \
  --meta kernel=$(uname -r) \
  --meta driver=$(cat "$(dirname "$M2V_ICD")/mesa-git-sha") \
  --meta governor=performance --meta ane=idle --meta load=0.12
```

`--meta KEY=VALUE` (repeatable) fills the Linux-side header: `kernel`, `driver` (the Mesa git SHA), `governor`, `ane`, and `load`. A key given to `--meta` overrides the same key in the Linux JSON.

The governor rule: translated timings move by tens of percent with the Linux CPU governor. Every table names the governor as found (or `performance` if you set it). Every table names the Mesa git driver. One table never mixes governors. A run that starts above 0.5 load gets the `WARNING` line. It is not a quiet-machine table. `TICKET.md` has the copy-paste tickets for the macOS windows and the Linux side.

