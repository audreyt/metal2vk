# How metal2vk works

metal2vk turns a Metal compute kernel (`.metal` source) into a Vulkan 1.3 compute module (SPIR-V) with open source tools. It does not edit the kernel. This page is the long version of the five steps in the README.

## The pipeline

```
kernel.metal  +  cases/<name>.cl (a small entry wrapper)
        |
        |  clang --target=spir -x cl -cl-std=clc++2021
        |        -cl-ext=-__opencl_c_generic_address_space -O0 -Xclang -disable-O0-optnone -I include/
        v
   LLVM IR      then LLVM's inliner and SROA (no InstCombine), then a few small rewrites of the IR text
        |
        |  clspv -x ir --cl-std=CLC++2021 --fp16 --inline-entry-points --spv-version=1.5 --long-vector
        v
   SPIR-V       ->  spirv-val --target-env vulkan1.3  ->  vkCreateComputePipelines
```

The script `compile.sh OUTDIR [case ...]` runs the whole chain for the kernels in `cases/`. The tool `tools/m2v-compile` runs it for one Metal file and caches the result. See [m2v-compile.md](m2v-compile.md). The script `syntax-check.sh` runs only the clang front end. That is the fast loop when you extend the shim. The driver `sweep/m2v_sweep.py` runs every kernel of the uzu and TensorFold sources through the chain and keeps a table. See [coverage.md](coverage.md).

## Why these choices

clang is the front end, not clspv's own. C++ for OpenCL with the generic address space switched off makes the implicit `this`, unqualified pointers and references private. That is what Metal's `thread` means. It is the only way uzu's `thread`-qualified methods and constructors compile. The shim defines `thread` as nothing.

The IR stays typed on purpose. clang's InstCombine rewrites `getelementptr float` into byte-offset `getelementptr i8` forms. clspv's pointer passes then lose the element type. A `float2` fragment load became eight byte loads per lane, and an 8x8 tile matmul ran 4x slower. uzu's `Gemm` ended in an `OpPhi` of a float pointer and a `[4 x i8]` pointer that fails validation. At -O0 clang marks every function `noinline`. Dropping that and running LLVM's inliner plus SROA (without InstCombine) keeps the types. The route needs `opt` from the same LLVM as clang, and `OPT=` selects it. The variable `M2V_OPT` overrides the flags. The old `clang -O2` route is `M2V_OPT="-O2 -fno-slp-vectorize -fno-vectorize -mllvm -inline-threshold=100000"`.

Entry points come from a wrapper. A kernel's grid, threadgroup size, optional buffers and specialization values are annotations in the Metal source. uzu uses `AXIS`, `GROUPS`, `THREADS`, `SPECIALIZE` and `OPTIONAL`. TensorFold uses attributes and `host_name` instantiations. The sweep driver reads them. The files `cases/*.cl` are hand-written wrappers for the kernels used by the tests.

## IR rewrites between clang and clspv

Some shapes in the IR that clang produces cannot be lowered by clspv. Small scripts in `sweep/` rewrite them. They do nothing unless a module has the shape. The scripts are `bfloat_to_i16.py` (clspv rejects LLVM's `bfloat` type), `flatten_single_member_structs.py`, `memcpy_mixed_as.py`, `promote_ptr_arrays.py` and `pointer_phi_to_index.py`. The last one turns loops that advance a pointer through a phi into an integer index plus indexed addressing. The driver `sweep/m2v_sweep.py` and `compile.sh` run them.

## clspv patches

clspv is patched in `clspv-patches/`. The scripts are idempotent and tested against clspv `f2b01dd6dce7058585eb4bcdb5326997cfeebfd4`. The script `tools/apply-clspv-patches.sh /path/to/clspv` applies them in the order the CI workflow uses. Each script's docstring says what it fixes and why. They fall into four groups.

The first group is cooperative matrix. A builtin `m2v_mma_f32` (and a half-input variant) lowers to `OpCooperativeMatrixMulAddKHR`. The group also adds the capability and the Vulkan memory model it requires. It drops the `Coherent` decoration that the memory model bans.

The second group is spec constants and a version marker. The `__m2v_spec_*` globals become specialization constants. A no-op `--m2v-spec-constants` option lets the host toolchain check that its clspv knows this protocol. An unpatched clspv rejects the flag.

The third group is the pointer passes. These are fixes in clspv's `SimplifyPointerBitcastPass`, the implicit-cast folds and the producer. Pointer arithmetic that came from C++ templates ends as valid typed access chains. Every give-up path prints `M2V-GIVEUP`. The sweep and `compile.sh` refuse any module whose clspv log contains it. A pass that gives up leaves a half rewritten module.

The fourth group is structured control flow. It covers a nested loop whose exit is the enclosing loop's latch, and merge and continue blocks that must not coincide. The module then passes the structured-CFG rules of `spirv-val`.

## The shim

```
include/
  metal_stdlib                 the shim: address spaces and fixed-width ints, type traits, vec<T,N> (a native ext_vector for
                               arithmetic types), bfloat, as_type, barriers, math (metal::, fast::, precise::),
                               simd_* (sub_group_* builtins), simdgroup_matrix<T,8,8> and simdgroup_multiply_accumulate
  m2v_extra.h, m2v_vec_ctors.h   additions found by the sweep; the vector constructors are generated by gen-shim-extras.py
  MetalPerformancePrimitives/  Metal 4 tensor operations (mpp::tensor_ops, matmul2d) emulated with the cooperative matrix path
                               and plain loops
  metal_simdgroup, metal_atomic, ...   umbrella headers that forward to metal_stdlib (make-header-stubs.sh)
cases/        one wrapper per translated kernel: includes the .metal file, declares the OpenCL entry
probes/       small experiments (cooperative matrix lane layout, noinline behaviour)
sweep/        the bulk driver, the IR rewrites, the coverage gate and the run stage
host/         a host layer that gives Metal-style host code (buffers, queues, pipelines) on Vulkan; see host/DESIGN.md
```

### Address spaces and `thread`

| Metal | shim |
| --- | --- |
| `device T*` | `__global T*` |
| `threadgroup T*` | `__local T*` |
| `constant T&` | `__constant T&` |
| `thread` | nothing (private by default) |

### simdgroup_matrix

The type `simdgroup_matrix<T,8,8>` keeps Apple's per-lane fragment layout. It has two elements per lane. The row is `(lane>>4&1)*4 + (lane>>1&3)` and the first column is `(lane>>3&1)*4 + (lane&1)*2`. uzu reads and writes `thread_elements()` directly, so the layout has to match. The 8x8 `VK_KHR_cooperative_matrix` fragment of Honeykrisp has the same layout for the accumulator, A and B uses. The script `probes/layout-run.py` checks all 32 lanes. So `simdgroup_multiply_accumulate` lowers to one `OpCooperativeMatrixMulAddKHR` with no data movement beyond element access. Without cooperative matrix, define `M2V_NO_COOPMAT`. The shim then emulates the multiply with subgroup shuffles, which is about 30x slower.

### Metal 4 tensor operations

The header `include/MetalPerformancePrimitives/MetalPerformancePrimitives.h` emulates `mpp::tensor_ops` (`tensor`, `dextents`, `matmul2d`, cooperative tensors). It uses the layout that MLX's NAX kernels, uzu and TensorFold pack for. Kernels written for the other (strict) hardware layout translate to valid SPIR-V, but their results land permuted. The header says which. See [coopmat-shapes.md](coopmat-shapes.md) for the cooperative matrix shapes the driver offers.

## Running a kernel

The runner must follow the clspv conventions. The file `m2v-run.c` does.

Kernel arguments become storage buffers on set 0, with the binding equal to the argument order. Plain-old-data arguments go to push constants after a 16 byte block of group and region offsets (zeros). The workgroup size is spec constants 0 to 2 with default (1,1,1). Set them from the reflection (`PropertyRequiredWorkgroupSize`) or from the kernel's `reqd_work_group_size`. Enable `VK_KHR_cooperative_matrix` and its feature when the module uses it.

clspv drops unused arguments, so bindings and push offsets differ per module. The script `compile.sh` writes a reflection JSON next to every module. The test drivers bind from it.
