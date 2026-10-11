#!/usr/bin/env python3
"""Inline uzu's MSL headers and emit standalone native Metal kernels."""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = {
    "activation": "activation/activation.metal",
    "softmax": "softmax/softmax.metal",
    "tilematmul": "matmul/common/simdgroup_multiply_accumulate.h",
    "tilematmul_rt": "matmul/common/simdgroup_multiply_accumulate.h",
}


def inline(path, kernel_dir, seen):
    real = os.path.realpath(path)
    if real in seen:
        return ""
    seen.add(real)
    with open(real) as f:
        src = f.read()
    out = []
    for line in src.splitlines():
        m = re.match(r'\s*#include\s+[<"]([^">]+)[">]', line)
        if not m:
            out.append(line)
            continue
        inc = m.group(1)
        if inc == "metal_stdlib":
            out.extend(["#include <metal_stdlib>", "using namespace metal;"])
            continue
        candidates = (os.path.join(os.path.dirname(path), inc),
                      os.path.join(kernel_dir, inc),
                      os.path.join(kernel_dir, "generated", os.path.basename(inc)))
        target = next((p for p in candidates if os.path.isfile(p)), None)
        if target is None:
            raise SystemExit(f"include not found: {inc} (from {path})")
        out.append(f"// ---- begin {inc} ----")
        out.append(inline(target, kernel_dir, seen))
        out.append(f"// ---- end {inc} ----")
    return "\n".join(out)


WRAPPERS = {
"activation": r'''#define M2V_ACT_WG 256
#define M2V_ACT_EPT 1
#define ACT_KERNEL(NAME, T) \\
kernel void NAME(device const T* input [[buffer(0)]], device T* output [[buffer(1)]], constant uint* n [[buffer(2)]], constant uint* act [[buffer(3)]], constant uint& in_place [[buffer(4)]], uint gid [[thread_position_in_grid]]) { \\
  for (uint e = 0; e < (uint)M2V_ACT_EPT; ++e) { uint tid = gid * (uint)M2V_ACT_EPT + e; if (tid < n[0]) Activation<T>(input, output, n[0], *(constant ActivationType*)act, in_place != 0, tid); } \\
}
ACT_KERNEL(activation_f32, float)
ACT_KERNEL(activation_f16, half)
''',
"softmax": r'''#define SOFTMAX_KERNEL(NAME, T) \\
kernel void NAME(device T* values [[buffer(0)]], device const T* sinks [[buffer(1)]], constant uint* meta [[buffer(2)]], constant uint& has_sinks [[buffer(3)]], uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) { \
  threadgroup float shared_max[METAL_SIMD_SIZE]; threadgroup float shared_norm[METAL_SIMD_SIZE]; \
  Softmax<T>(values, sinks, meta[0], meta[1], meta[2], shared_max, shared_norm, has_sinks != 0, gid.x, gid.y, lid.x); \
}
SOFTMAX_KERNEL(softmax_f32, float)
SOFTMAX_KERNEL(softmax_f16, half)
''',
"tilematmul": r'''kernel void tile_matmul_f32(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]], constant uint* dims [[buffer(3)]], uint3 gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint M = dims[0], N = dims[1], K = dims[2]; (void)M;
  const short2 rc = MMA::get_lane_coordinates(lane); const uint tile_x = gid.x, tile_y = gid.y;
  MMA::ThreadDataType acc; acc[0] = 0.0f; acc[1] = 0.0f;
  for (uint k = 0; k < K; k += 8) { MMA::ThreadDataType a, b; MMA::load(a, A + (tile_y * 8 + rc.y) * K + k + rc.x, K, 1); MMA::load(b, B + (k + rc.y) * N + tile_x * 8 + rc.x, N, 1); MMA::mma(acc, a, b, acc); }
  MMA::store(acc, C + (tile_y * 8 + rc.y) * N + tile_x * 8 + rc.x, N, 1);
}
''',
"tilematmul_rt": r'''#define RT_KERNEL(NAME, MR, NR) \\
kernel void NAME(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]], constant uint* dims [[buffer(3)]], uint3 gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \\
  const uint N = dims[1], K = dims[2]; const short2 rc = MMA::get_lane_coordinates(lane); const uint row0 = gid.y * (8 * MR), col0 = gid.x * (8 * NR); \\
  MMA::ThreadDataType acc[MR][NR]; for (int i=0;i<MR;i++) for (int j=0;j<NR;j++) { acc[i][j][0]=0.0f; acc[i][j][1]=0.0f; } \\
  for (uint k=0;k<K;k+=8) { MMA::ThreadDataType a[MR],b[NR]; for(int i=0;i<MR;i++) MMA::load(a[i],A+(row0+i*8+rc.y)*K+k+rc.x,K,1); for(int j=0;j<NR;j++) MMA::load(b[j],B+(k+rc.y)*N+col0+j*8+rc.x,N,1); for(int i=0;i<MR;i++) for(int j=0;j<NR;j++) MMA::mma(acc[i][j],a[i],b[j],acc[i][j]); } \\
  for(int i=0;i<MR;i++) for(int j=0;j<NR;j++) MMA::store(acc[i][j],C+(row0+i*8+rc.y)*N+col0+j*8+rc.x,N,1); \\
}
RT_KERNEL(tile_matmul_rt2x2_f32, 2, 2)
RT_KERNEL(tile_matmul_rt4x2_f32, 4, 2)
RT_KERNEL(tile_matmul_rt4x4_f32, 4, 4)
''',
}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.environ.get("UZU_ROOT", os.path.expanduser("~/src/uzu")),
        "crates/uzu-engine/src/backends/metal/kernel")
    kernel_dir = os.path.abspath(root)
    for case, source in CASES.items():
        src = inline(os.path.join(kernel_dir, source), kernel_dir, set())
        # DSL annotations are compile-time metadata only. Keep the original source
        # text except for stripping those wrappers; the emitted entries below bind
        # the same arguments, dimensions, and thread positions as cases/*.cl.
        src = re.sub(r"^#define METAL_PRAGMA_(?:UNROLL|NO_UNROLL).*\n?", "", src, flags=re.M)
        src = src.replace("METAL_PRAGMA_UNROLL", "").replace("METAL_PRAGMA_NO_UNROLL", "")
        if case.startswith("tilematmul"):
            src += "\nusing MMA = uzu::matmul::SimdgroupMMA<float, 8, 8>;\n"
        header = ("// Generated by flatten.py. Source headers from uzu, MIT License, Copyright (c) 2025 Mirai Tech Inc.\n"
                  "#include <metal_stdlib>\nusing namespace metal;\n")
        out = os.path.join(HERE, case + ".metal")
        wrapper = WRAPPERS[case].replace("\\\\", "\\")
        with open(out, "w") as f:
            f.write(header + src + "\n" + wrapper)


if __name__ == "__main__":
    main()
