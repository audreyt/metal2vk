#!/usr/bin/env python3
"""MLX's own Metal quantized matmul kernels for small M, as mx.fast.metal_kernel call sites.

The kernel bodies call the pinned MLX impl functions (quantized.h at the MLX pin in omarchy-mlx's mlx.lock):
  qmvw   affine_qmv_wide  one weight decode reused across NV vectors (NV 2..5; T 6..8 take two tiles)
  qmvf   affine_qmv_fast  one threadgroup per vector (the plain qmv MLX Metal uses below its batch limit)
  qmmt   affine_qmm_t     32x32x32 steel tile (the Metal matrix route)
The harness (qbench.py, on the Mac) and this builder (on the CT) take source, header, template and grid from here, so the
MSL text mlx assembles at dispatch is the text this file compiles, and the AOT key sha256(msl NUL entry) matches.

usage: qmv_corpus.py build MLX_ROOT OUTDIR       header, one .metal per variant, spv + json in OUTDIR/aot, manifest.json
"""
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
GS, BITS = 64, 4
INC = re.compile(r'^\s*#\s*include\s+"(mlx/[^"]+)"')
HEADERS = ("mlx/backend/metal/kernels/utils.h", "mlx/backend/metal/kernels/steel/gemm/gemm.h",
           "mlx/backend/metal/kernels/quantized_utils.h", "mlx/backend/metal/kernels/quantized.h")
INPUTS = ["w", "scales", "biases", "x", "K", "N", "M"]

SRC_QMVW = """
  qmv_wide_impl<T, GS, BITS, NV, KL>(w, scales, biases, x, y, K, N, M,
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""
SRC_QMVF = """
  qmv_fast_impl<T, GS, BITS>(w, scales, biases, x, y, K, N,
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""
SRC_QMMT = """
  constexpr int BK_padded = (32 + 16 / sizeof(T));
  threadgroup T Xs[32 * BK_padded];
  threadgroup T Ws[32 * BK_padded];
  qmm_t_impl<T, GS, BITS, true, 32, 32, 32>(w, scales, biases, x, y, Xs, Ws, K, N, M, K,
      threadgroup_position_in_grid, thread_index_in_threadgroup, simdgroup_index_in_threadgroup,
      thread_index_in_simdgroup);
"""


def variants():
    """[{id, name, source, tmpl, kind, nv}] in a fixed order."""
    out = [{"id": "qmvf", "name": "qmv_fast", "source": SRC_QMVF, "kind": "qmvf", "nv": 1,
            "tmpl": [("T", "bfloat16"), ("GS", GS), ("BITS", BITS)]}]
    for nv in (2, 3, 4, 5):
        out.append({"id": f"qmvw{nv}", "name": "qmv_wide", "source": SRC_QMVW, "kind": "qmvw", "nv": nv,
                    "tmpl": [("T", "bfloat16"), ("GS", GS), ("BITS", BITS), ("NV", nv), ("KL", 8)]})
    out.append({"id": "qmmt", "name": "qmm_t", "source": SRC_QMMT, "kind": "qmmt", "nv": 32,
                "tmpl": [("T", "bfloat16"), ("GS", GS), ("BITS", BITS)]})
    return out


def variant_for(kind, t):
    """The variant a T-row verify step uses. qmvw: ceil(T/5) tiles of ceil(T/tiles) vectors (MLX's rule)."""
    if kind == "qmvw":
        tiles = (t + 4) // 5
        return f"qmvw{(t + tiles - 1) // tiles}"
    return kind


def grid(kind, nv, t, n):
    """(grid in threads, threadgroup) for T rows and N output columns, as MLX dispatches them."""
    if kind == "qmvw":
        tiles = (t + nv - 1) // nv
        return (tiles * 32, ((n + 7) // 8) * 2, 1), (32, 2, 1)
    if kind == "qmvf":
        return (t * 32, ((n + 7) // 8) * 2, 1), (32, 2, 1)
    return (((n + 31) // 32) * 128, ((t + 31) // 32), 1), (128, 1, 1)  # qmmt: 128 threads, grid y in tiles


def flatten(root):
    seen, out = set(), []

    def inline(rel):
        if rel in seen or rel.endswith("/complex.h"):  # the shim's mlx_prelude.h carries complex.h
            return
        seen.add(rel)
        for line in (pathlib.Path(root) / rel).read_text().splitlines():
            m = INC.match(line)
            if m:
                inline(m.group(1))
            elif line.strip() != "#pragma once" and not line.startswith("instantiate_metal_simd_reduction_funcs(bfloat16_t"):
                out.append(line)  # the shim already defines the bfloat simd reductions bf16_math.h adds

    for rel in HEADERS:
        inline(rel)
    return "\n".join(out) + "\n"


def assemble_module():
    path = os.environ.get("M2V_ASSEMBLE") or str(HERE.parents[1] / "sweep/mlx-batch/vlmb/assemble.py")
    spec = importlib.util.spec_from_file_location("m2v_assemble", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def inputs_meta():
    big = {"ndim": 2, "size": 1 << 20}
    return [{"name": n, "dtype": d, **m} for n, d, m in (
        ("w", "uint32", big), ("scales", "bfloat16", big), ("biases", "bfloat16", big), ("x", "bfloat16", big),
        ("K", "int32", {"ndim": 0, "size": 1}), ("N", "int32", {"ndim": 0, "size": 1}),
        ("M", "int32", {"ndim": 0, "size": 1}))]


def msl_for(asm, variant, header):
    reg = {"name": variant["name"], "source": variant["source"], "header": header, "input_names": INPUTS,
           "output_names": ["y"]}
    return asm.build(reg, inputs_meta(), ["bfloat16"], variant["tmpl"])


def build(root, outdir):
    out = pathlib.Path(outdir)
    (out / "aot").mkdir(parents=True, exist_ok=True)
    header = flatten(root)
    (out / "header.metal").write_text(header)
    asm = assemble_module()
    manifest = []
    for v in variants():
        msl, entry = msl_for(asm, v, header)
        (out / f"{v['id']}.metal").write_text(msl)
        key = hashlib.sha256(msl.encode() + b"\0" + entry.encode()).hexdigest()
        cmd = [sys.executable, os.environ["M2V_COMPILE"], "--msl", str(out / f"{v['id']}.metal"), "--out",
               str(out / "spv"), "--name", entry, "--json"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        row = {"id": v["id"], "entry": entry, "key": key, "compile": r.returncode}
        if r.returncode == 0:
            res = json.loads(r.stdout.strip().splitlines()[-1])
            for ext in ("spv", "json"):
                data = pathlib.Path(res[ext]).read_bytes()
                (out / "aot" / f"{key}.{ext}").write_bytes(data)
            row["spv_sha256"] = hashlib.sha256((out / "aot" / f"{key}.spv").read_bytes()).hexdigest()
        else:
            row["error"] = (r.stderr or r.stdout).strip().splitlines()[-1][:300]
        manifest.append(row)
        print(row["id"], row["compile"], row.get("error", "ok"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if any(row["compile"] for row in manifest):
        sys.exit("build failed: at least one variant did not compile")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
