# metal2vk

Run Metal GPU kernels on Linux: metal2vk translates Apple's Metal compute code to Vulkan with open source tools.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink)](https://github.com/sponsors/joshuaswarren)

The fast local AI software for Apple Silicon (oMLX, mlx-lm, mlx-vlm, the uzu and TensorFold engines) gets its speed from GPU kernels
written in Metal, Apple's GPU language, and Metal only exists on Apple's operating systems. metal2vk reads those kernels as they are and translates them
to Vulkan with clang, clspv and SPIRV-Tools, with no Apple software, so they can run on Apple Silicon under Linux through an open
source Vulkan driver.

## What works today

Measured, with the account of how in [docs/what-this-enables.md](docs/what-this-enables.md):

| Measured | Result |
|---|---|
| Kernels from uzu and TensorFold that compile to valid Vulkan programs | 606 of 610 |
| uzu kernels that compile | 121 of 121 |
| oMLX, mlx-lm, mlx-vlm kernels sampled: compile | 31 of 32 |
| Same kernels, run on an M1 and checked against a reference | 31 of 31 agree (5 values of about 9,200 differ by 2 to 3 units in the last place in one kernel; about 0.4% of the values in one attention kernel differ by at most 0.02% of their row's largest value) |
| Refused | 1 kernel needs the M5 chip's tensor operations |
| Simple tiled matrix multiply, M1 Max | 4.27 TFLOP/s (hand-tuned Vulkan: 4.07) |
| uzu tiled matrix multiply, M1 Max | about 0.3 times the hand-tuned kernel |
| Compile time per kernel | 1.8 s first time, 0.8 s cached |
| Whole-model speed through this route | not measured yet |

A kernel that fails to translate or fails its check is never run: it falls back to the older route or stops with an error that names
the kernel. The reference for a check is the original translator's output where it had one, otherwise a step-by-step Python model of
the Metal source.

## Why it matters

- **Linux gets the Mac's kernels.** The speed of these engines is in their custom kernels. Translating the Metal source avoids
  rewriting each kernel in another language.
- **The long tail works.** The models that depend on custom kernels (Qwen3.5-class mixture-of-experts and gated-delta-net models
  served through oMLX, mlx-lm and mlx-vlm, and the uzu and TensorFold engines) are the ones a plain port misses.
- **It is open end to end.** clang, clspv, SPIRV-Tools and Mesa's Vulkan driver for Apple silicon. Nothing from Apple is needed.

## Quick start

You need Linux, `git`, `python3` with `numpy`, `cmake` and `ninja`, clang and LLVM 23 (with C++ for OpenCL), SPIRV-Tools, and the
Vulkan loader and headers (1.3.255 or newer: the first headers that define `VK_KHR_cooperative_matrix`, which `m2v-run.c` uses). Running (the last step) needs a Vulkan 1.3 GPU; compiling does not.

**0. Get the code.** Everything below runs from one new directory:

```bash
mkdir m2v && cd m2v
git clone https://github.com/joshuaswarren/metal2vk
git clone https://github.com/google/clspv
git clone https://github.com/trymirai/uzu && git -C uzu checkout c92723bdc04e40f55286de458a457172ce62c5b6
```

**1. Build the patched clspv.** metal2vk uses clspv at a pinned commit plus the patches in `metal2vk/clspv-patches/`. From `m2v/`:

```bash
cd clspv
git checkout f2b01dd6dce7058585eb4bcdb5326997cfeebfd4
python3 utils/fetch_sources.py --shallow
../metal2vk/tools/apply-clspv-patches.sh .
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCLSPV_BUILD_TESTS=OFF -B build && ninja -C build clspv
cd ../metal2vk
```

The build compiles LLVM, so it takes a while. You are back in `m2v/metal2vk/` afterwards.

**2. Translate a kernel.** Take uzu's activation and softmax kernels, unmodified. From `m2v/metal2vk/`:

```bash
export UZU_ROOT=$PWD/../uzu CLSPV=$PWD/../clspv/build/bin/clspv CLANG=clang-23 OPT=opt-23   # your clang 23 and opt 23
./compile.sh out activation softmax
```

`out/` now holds `activation.spv` and `softmax.spv` (Vulkan 1.3 SPIR-V, checked by `spirv-val`) and a `.json` next to each that
says which buffer is which argument, plus the LLVM IR and the clspv log.

**3. Run it** on a machine with a Vulkan 1.3 GPU, for example Apple Silicon under Linux with Mesa's Honeykrisp driver, and compare
with a CPU reference. Still in `m2v/metal2vk/`:

```bash
cc -O2 m2v-run.c -lvulkan -o m2v-run
python3 m2v-test.py results ./m2v-run out activation softmax     # writes results/results.json
```

To translate one Metal file of your own, `tools/m2v-compile --msl kernel.metal --out DIR` does the same with a cache
([docs/m2v-compile.md](docs/m2v-compile.md)).

## How it works

1. clang parses the Metal source as C++ for OpenCL against a small `metal_stdlib` shim (`include/`) that gives Metal's types, address
   spaces, `simdgroup_matrix` and the other builtins.
2. LLVM's inliner keeps the IR typed, and a few small scripts rewrite shapes that clspv cannot lower.
3. A patched clspv turns the IR into Vulkan 1.3 compute SPIR-V; cooperative matrix operations use `VK_KHR_cooperative_matrix`.
4. `spirv-val` checks the module; a pass that had to give up makes the build fail instead of keeping wrong code.
5. The module runs on any Vulkan 1.3 device with 32-wide subgroups, storage buffers and push constants. The target is Honeykrisp,
   Mesa's driver for Apple silicon.

[docs/how-it-works.md](docs/how-it-works.md) has the details and the reasons for each choice.

## Limits

- **Not every kernel translates.** 606 of 610 do today; [docs/coverage.md](docs/coverage.md) lists the rest by cause, and
  `sweep/results/FAILURES.md` has the first error of each. One group still fails: kernels that keep a GPU address inside a buffer
  (a pointer used as runtime data). They need a change to the driver's argument model or to clspv's physical-pointer support.
- **Translated is not the same as correct.** A kernel was checked only if it is in the 31 above. Where Metal leaves the order of
  floating-point sums or the rounding of `exp` open, results can differ in the last bits.
- **Matrix multiplies are slower.** uzu's tiled matrix multiply runs at about 0.3 times the hand-tuned Vulkan kernel, so matrix
  multiplies keep using hand-tuned kernels and metal2vk covers the many smaller ones.
- **Newest-chip features.** One sampled kernel uses a Metal feature that only exists on the M5 chip and is refused by name. Metal 4's
  tensor operations are emulated in the shim for the matrix layout that MLX, uzu and TensorFold pack for; a kernel written for Apple's
  other (strict) layout translates but returns permuted results, and the shim header says which.
- **Cooperative matrix needs a driver that offers it.** Stock Mesa Honeykrisp does not yet; the
  [omarchy-mlx](https://github.com/joshuaswarren/omarchy-mlx) fork does.
- **No whole-model speed number yet.** Kernel speed is measured; tokens per second through this route is not.

## Status and roadmap

metal2vk is a working translator with a growing, gated test: CI runs the whole uzu and TensorFold sweep on every pull request and
fails if a kernel that used to translate stops translating.

Next is whole engines on Linux, starting with oMLX. That route is the open omarchy-mlx pull request 70 (not merged). It is opt-in
through an environment variable and the default stays the old translator. Checking that the GPU results match is its next gate, and
there is no tokens-per-second number for it yet.

`host/` holds the host layer (buffers, queues, pipelines) that gives Metal-style host code a Vulkan backend.

## Documentation

- [What metal2vk enables](docs/what-this-enables.md): the measured results and what is not measured yet.
- [How it works](docs/how-it-works.md): the pipeline, the shim, the clspv patches.
- [Kernel measurements](docs/measurements.md) and [coverage by failure class](docs/coverage.md).
- [m2v-compile](docs/m2v-compile.md): the command line front door and its reflection format.
- [Host layer](host/DESIGN.md) and [bench harness](docs/bench-status.md).

## Contributing

Issues with a failing kernel and its first error message are the most useful thing you can send. See
[CONTRIBUTING.md](CONTRIBUTING.md) for how to run the sweep and the rules for shim and clspv changes.

## Support

Every bit of support helps keep metal2vk alive and free. If you are able, [sponsor on GitHub](https://github.com/sponsors/joshuaswarren) or send a Lightning donation to `joshuaswarren@strike.me` to directly fund continued development and new integrations. If financial support is not an option, [star the repo](https://github.com/joshuaswarren/metal2vk), share it, or recommend it to a colleague.

## License

MIT. See [LICENSE](LICENSE).
