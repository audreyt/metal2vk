#!/usr/bin/env python3
"""Deployed mx.quantized_matmul against MLX's own Metal small-M quantized kernels, translated by metal2vk.

One process, one device, interleaved: every round runs one timed batch of each (T, arm) cell in a shuffled order
(reversed on odd rounds); round 0 is warm-up. Arms: dep (mx.quantized_matmul, whatever route the build picks),
qmvf (affine_qmv_fast), qmvw (affine_qmv_wide, MLX's tile rule for NV), qmmt (affine_qmm_t). Each cell's output
is checked against a float64 CPU reference on sampled output columns (row error = max abs error of the row over its
largest reference magnitude).

usage: qbench.py --build DIR [--k 5120] [--ns 10240,12288,17408,248320] [--ts 1,2,3,4,5,6,7,8] [--rounds 11]
                 [--out FILE.json] [--selftest]
DIR is the qmv_corpus.py build output (header.metal, aot/). Run under the GPU queue wrapper. Set nothing else.
Route check (not done here, mlx prints it at exit): stderr must end with 'translator=0 fallbacks=0 refusals=0'.
"""
import argparse
import json
import os
import pathlib
import random
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qmv_corpus as corpus  # noqa: E402

CACHE_BYTES = 256 << 20  # rotate weight copies until their total passes this, so the SLC never holds the weights
SAMPLE_COLS = 96


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--ns", default="10240,12288,17408,248320")
    ap.add_argument("--ts", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--rounds", type=int, default=11)
    ap.add_argument("--batch", type=int, default=16, help="dispatches per timed eval")
    ap.add_argument("--arms", default="dep,qmvf,qmvw,qmmt")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true", help="CPU, no translated arms: checks the harness plumbing")
    return ap.parse_args()


def main():
    a = parse()
    build = pathlib.Path(a.build)
    if not a.selftest:
        os.environ.setdefault("MLX_OMARCHY_M2V_AOT_DIR", str(build / "aot"))
        os.environ.setdefault("MLX_OMARCHY_M2V_SUMMARY", "1")
        for base in ("QMV_WIDE", "QMV_FAST", "QMM_T"):
            os.environ.setdefault(f"MLX_OMARCHY_M2V_KERNEL_{base}", "m2v")
    import mlx.core as mx
    import numpy as np

    if a.selftest:
        mx.set_default_device(mx.cpu)
    gs, bits = corpus.GS, corpus.BITS
    k = a.k
    ns = [int(v) for v in a.ns.split(",")]
    ts = [int(v) for v in a.ts.split(",")]
    arms = [v for v in a.arms.split(",") if v]
    if a.selftest:
        arms = ["dep", "ref"]
    header = (build / "header.metal").read_text()
    by_id = {v["id"]: v for v in corpus.variants()}
    kern = {}
    if not a.selftest:
        for v in by_id.values():
            kern.setdefault(v["name"], mx.fast.metal_kernel(
                name=v["name"], input_names=corpus.INPUTS, output_names=["y"], source=v["source"], header=header))
    scal = {n: mx.array(n, dtype=mx.int32) for n in set(ns) | {k} | set(ts)}

    def call(arm, t, n, wq, s, b, x):
        if arm == "dep":
            return mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=gs, bits=bits)
        if arm == "ref":
            return x @ mx.dequantize(wq, s, b, group_size=gs, bits=bits).T
        vid = corpus.variant_for(arm, t)
        v = by_id[vid]
        g, tg = corpus.grid(v["kind"], v["nv"], t, n)
        tmpl = [(nm, getattr(mx, val) if isinstance(val, str) else val) for nm, val in v["tmpl"]]
        return kern[v["name"]](inputs=[wq, s, b, x, scal[k], scal[n], scal[t]], template=tmpl, grid=g,
                               threadgroup=tg, output_shapes=[(t, n)], output_dtypes=[mx.bfloat16])[0]

    def cells():
        out = []
        for arm in arms:
            for t in ts:
                if arm == "qmvw" and t < 2:
                    continue
                out.append((arm, t))
        return out

    result = {"k": k, "gs": gs, "bits": bits, "device": str(mx.default_device()), "rounds": a.rounds,
              "batch": a.batch, "mlx": getattr(mx, "__version__", ""),
              "env": {e: os.environ[e] for e in sorted(os.environ) if e.startswith("MLX_OMARCHY")}, "shapes": []}
    rng = random.Random(20261015)
    dead = {}
    for n in ns:
        bytes_one = n * k // 2 + 2 * n * (k // gs) * 2
        copies = min(16, max(2, -(-CACHE_BYTES // bytes_one)))
        mx.random.seed(1234 + n)
        w = [(mx.random.randint(0, 65536, (n, k // 8), dtype=mx.uint32) << 16)
             | mx.random.randint(0, 65536, (n, k // 8), dtype=mx.uint32) for _ in range(copies)]
        sc = [(mx.random.uniform(shape=(n, k // gs)) * 0.02 + 0.002).astype(mx.bfloat16) for _ in range(copies)]
        bi = [(mx.random.uniform(shape=(n, k // gs)) * 0.2 - 0.1).astype(mx.bfloat16) for _ in range(copies)]
        xs = {t: mx.random.normal((t, k)).astype(mx.bfloat16) for t in ts}
        mx.eval(w, sc, bi, list(xs.values()))
        cols = np.random.default_rng(7).choice(n, size=min(SAMPLE_COLS, n), replace=False)
        idx = mx.array(cols)
        wd = np.array(mx.take(w[0], idx, axis=0))
        q = np.stack([(wd >> (4 * j)) & 0xF for j in range(8)], axis=-1).reshape(len(cols), k).astype(np.float64)
        s64 = np.repeat(np.array(mx.take(sc[0], idx, axis=0).astype(mx.float32)), gs, axis=1).astype(np.float64)
        b64 = np.repeat(np.array(mx.take(bi[0], idx, axis=0).astype(mx.float32)), gs, axis=1).astype(np.float64)
        wref = q * s64 + b64

        errs = {}
        for arm, t in cells():
            if arm in dead:
                continue
            try:
                y = call(arm, t, n, w[0], sc[0], bi[0], xs[t])
                mx.eval(y)
            except Exception as exc:  # a kernel this device cannot run is a result, not a crash
                dead[arm] = f"T={t}: {str(exc).strip().splitlines()[0][:200]}"
                arms.remove(arm)
                continue
            yv = np.array(y.astype(mx.float32)).astype(np.float64)[:, cols]
            ref = np.array(xs[t].astype(mx.float32)).astype(np.float64) @ wref.T
            row = np.abs(yv - ref).max(axis=1) / np.abs(ref).max(axis=1)
            errs[(arm, t)] = float(row.max())
        samples = {c: [] for c in cells() if c in errs}

        for r in range(a.rounds):
            order = cells()
            rng.shuffle(order)
            if r % 2:
                order.reverse()
            for arm, t in order:
                outs = [call(arm, t, n, w[i % copies], sc[i % copies], bi[i % copies], xs[t]) for i in range(a.batch)]
                t0 = time.perf_counter()
                mx.eval(outs)
                dt = time.perf_counter() - t0
                if r > 0:
                    samples[(arm, t)].append(dt / a.batch * 1e3)
        rows = []
        for (arm, t), v in samples.items():
            q1, q2, q3 = statistics.quantiles(v, n=4) if len(v) >= 4 else (v[0], statistics.median(v), v[-1])
            rows.append({"arm": arm, "t": t, "ms_median": q2, "ms_q1": q1, "ms_q3": q3, "ms_min": min(v),
                         "row_err_vs_f64": errs[(arm, t)], "samples": len(v)})
        result["shapes"].append({"n": n, "weight_copies": copies, "rows": rows})
        print(f"\nN={n} K={k} gs={gs} bits={bits} copies={copies}  ms per dispatch (median of {a.rounds - 1} interleaved rounds); err = row error vs float64 on {SAMPLE_COLS} sampled columns")
        print("| T | " + " | ".join(f"{arm} ms | {arm} err" for arm in arms) + " |")
        print("|---|" + "---|---|" * len(arms))
        for t in ts:
            cols_txt = []
            for arm in arms:
                hit = [r for r in rows if r["arm"] == arm and r["t"] == t]
                cols_txt.append(f"{hit[0]['ms_median']:.3f} | {hit[0]['row_err_vs_f64']:.1e}" if hit else "- | -")
            print(f"| {t} | " + " | ".join(cols_txt) + " |")
        for arm in arms:
            m = {r["t"]: r["ms_median"] for r in rows if r["arm"] == arm}
            lo, hi = min(m), max(m)
            print(f"{arm}: T={hi}/T={lo} cost {m[hi] / m[lo]:.2f}x" + (f", vs dep T=1: {m[hi] / next(r['ms_median'] for r in rows if r['arm'] == 'dep' and r['t'] == 1):.2f}x" if 'dep' in arms and 1 in ts else ""))
    result["dead"] = dead
    for arm, why in dead.items():
        print(f"arm {arm} did not run: {why}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(result, indent=1) + "\n")


main()
