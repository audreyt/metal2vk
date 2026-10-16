#!/usr/bin/env python3
"""Run the Muse fused Metal kernels (mlx-lm 06e03f7ce, oMLX muse_spark) on the real Vulkan device and check them.

The pinned wrappers run unmodified: SRC/mlxlm (the mlx_lm/kernels package) and SRC/omlx_norm_kernels.py. Each case calls a
wrapper at T = 1 (decode) and T = 16 (verify) tokens with the dimensions muse_corpus.py compiled, then
  - records whether it ran and its outputs are finite,
  - compares every output with a float32 reference computed by mlx on the CPU stream, with the kernels' own roundings,
    as row error = largest absolute error in a row over the row's largest reference magnitude (index outputs: exact),
  - times batches of 16 dispatches (host encode plus GPU; the kernels are small, so this is a floor, not a GPU-only time).
--route m2v sends the 12 kernels through omarchy-mlx's metal2vk route from the AOT directory built by muse_corpus.py;
--route translator leaves the default route. Run both in separate processes to compare the two.

usage: qmuse.py --src SRC [--route m2v|translator|native] [--aot DIR] [--cases a,b] [--tokens 1,16] [--dump FILE.npz]
                [--reps 9] [--batch 16] [--out FILE.json] [--selftest]
Exit 0 when every case is ok, 1 when any case is MISMATCH or ERROR or no case ran.
--route native is macOS MLX (nothing is routed). With --route m2v mlx prints a route summary at exit: stderr must hold
'route summary: m2v=N translator=0 fallbacks=0 refusals=0' (the total line).
--selftest runs on the CPU with each reference standing in for its kernel (checks the plumbing only).
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import statistics
import sys
import time
import zlib
from types import SimpleNamespace

import numpy as np

RNG: list = [None]  # set per case from the case id, so every platform draws the same inputs

BASES = ("EXPERT_ROUTING", "NORM_ADD_NORM", "NORM_ADD", "NORM_ADD_NORM_PRE_NORM", "POST_EXPERT_NORM_COMBINE", "QK_NORM",
         "QK_NORM_ROPE", "RMS_NORM_SIGMOID_GATE", "OMLX_MUSE_SPARK_SCALED_RMS", "OMLX_MUSE_SPARK_POST_ATTENTION",
         "OMLX_MUSE_SPARK_MOE_TAIL", "OMLX_MUSE_SPARK_POST_MOE")
TOKENS = (1, 16)
BF_TOL = 2e-2  # about 2.5 bf16 ulps of the row maximum: the kernels round at fixed points, the reference mimics them
F32_TOL = 1e-4
HQ, HK, HD, E, K = 32, 8, 128, 256, 8


def setup_env(args):
    if args.selftest or args.route != "m2v":
        return
    os.environ.setdefault("MLX_OMARCHY_M2V_AOT_DIR", str(pathlib.Path(args.aot).resolve()))
    os.environ.setdefault("MLX_OMARCHY_M2V_SUMMARY", "1")
    for base in BASES:
        os.environ.setdefault(f"MLX_OMARCHY_M2V_KERNEL_{base}", "m2v")


def make_cases(mx, k, om):
    """{id: builder(T) -> (run() -> [arrays], ref() -> [(array, tol)], moved_bytes)}; k is the mlx-lm kernels module."""

    def r(shape, dtype=mx.bfloat16, scale=1.0, shift=0.0):
        a = RNG[0].standard_normal(shape).astype("float32") * scale + shift  # numpy: the same bytes on every platform
        return mx.array(a).astype(dtype)

    def softmax_rows(shape):
        a = RNG[0].standard_normal(shape).astype("float64")
        e = np.exp(a - a.max(axis=-1, keepdims=True))
        return mx.array((e / e.sum(axis=-1, keepdims=True)).astype("float32"))

    def f(x):
        return x.astype(mx.float32)

    def rb(x):
        return x.astype(mx.bfloat16).astype(mx.float32)

    def inv_rms(v, eps):
        return mx.rsqrt(mx.mean(v * v, axis=-1, keepdims=True) + eps)

    def outs(value):
        value = value if isinstance(value, (list, tuple)) else [value]
        return [v for v in value if v is not None]

    cases = {}

    def expert_routing(t):
        logits, bias = r((t, E), mx.float32, 2.0), r((E,), mx.float32, 0.1)
        mult, eps, scale = 1.0, 1e-9, 2.5

        def run():
            return outs(k.expert_routing(logits, bias, top_k=K, multiplier=mult, score_eps=eps, route_scale=scale))

        def ref():
            score = mx.sigmoid(logits * mult)
            inds = mx.argsort(-(score + bias), axis=-1)[:, :K]
            w = mx.take_along_axis(score, inds, axis=-1)
            return [(inds.astype(mx.float32), 0.0), (w / (w.sum(-1, keepdims=True) + eps) * scale, F32_TOL)]
        return run, ref, [logits, bias]

    def norm_add_norm_case(t, d, post_scale, next_norm, pre_only=False):
        branch, stream = r((1, t, d)), r((1, t, d), mx.float32)
        alpha, beta = r((d,), mx.float32, 0.1, 1.0), r((d,), mx.float32, 0.1, 0.5)
        pw, nw = r((d,), mx.bfloat16, 0.1, 1.0), r((d,), mx.bfloat16, 0.1, 1.0)
        norm = SimpleNamespace(weight=nw, eps=1e-6)
        eps = 1e-6

        def run():
            if pre_only:
                return outs(k.pre_norm(stream, norm, dtype=mx.bfloat16))
            return outs(k.norm_add_norm(branch, stream, alpha=alpha, beta=beta, post_weight=pw if post_scale else None,
                                        post_eps=eps, next_norm=norm if next_norm else None))

        def ref():
            if pre_only:
                return [(rb(stream * inv_rms(stream, eps) * f(nw)), BF_TOL)]
            v = f(branch)
            if post_scale:
                v = rb(f(pw) * v)
            n = rb(v * inv_rms(v, eps))
            s = alpha * stream + beta * n
            res = [(s, F32_TOL)]
            if next_norm:
                res.append((rb(s * inv_rms(s, eps) * f(nw)), BF_TOL))
            return res
        return run, ref, [branch, stream, alpha, beta, pw, nw]

    def post_expert(t, d):
        experts, weights = r((t, K, d)), softmax_rows((t, K))
        w, eps = r((d,), mx.bfloat16, 0.1, 1.0), 1e-6

        def run():
            return outs(k.post_expert_norm_combine(experts, weights, norm_weight=w, eps=eps))

        def ref():
            acc = mx.zeros((t, d))
            for j in range(K):
                v = f(experts[:, j]) * f(w)
                acc = rb(acc + v * inv_rms(v, eps) * rb(weights[:, j:j + 1]))
            return [(acc, BF_TOL)]
        return run, ref, [experts, weights, w]

    def qk_heads(qkvg, scale, eps):
        x = f(qkvg)[..., :(HQ + HK) * HD].reshape(*qkvg.shape[:-1], HQ + HK, HD)
        y = rb(x * inv_rms(x, eps))
        return mx.concatenate([rb(y[..., :HQ, :] * scale), y[..., HQ:, :]], axis=-2)

    def qk_norm_case(t):
        qkvg, scale, eps = r((1, t, 2 * (HQ + HK) * HD)), 43.784 / HD ** 0.5, 1e-6

        def run():
            return outs(k.qk_norm(qkvg, n_heads=HQ, n_kv_heads=HK, head_dim=HD, eps=eps, query_scale=scale))

        def ref():
            y = qk_heads(qkvg, scale, eps)
            return [(y[..., :HQ, :], BF_TOL), (y[..., HQ:, :], BF_TOL)]
        return run, ref, [qkvg]

    def qk_norm_rope_case(t, batch_offsets):
        b = 4 if batch_offsets else 1
        qkvg, scale, eps, base = r((b, t, 2 * (HQ + HK) * HD)), 43.784 / HD ** 0.5, 1e-6, 1e6
        offset = mx.array([0, 1, 3, 7], dtype=mx.int32) if batch_offsets else 0

        def run():
            return outs(k.qk_norm_rope(qkvg, n_heads=HQ, n_kv_heads=HK, head_dim=HD, eps=eps, query_scale=scale,
                                       rope_base=base, offset=offset))

        def ref():
            y = qk_heads(qkvg, scale, eps)
            off = [0, 1, 3, 7] if batch_offsets else [0]
            pos = mx.array([[off[i] + j for j in range(t)] for i in range(b)], dtype=mx.float32)[..., None, None]
            i = mx.arange(HD // 2, dtype=mx.float32)
            theta = pos * mx.exp(-(i / (HD // 2)) * math.log(base))
            c, s = mx.cos(theta), mx.sin(theta)
            even, odd = y[..., 0::2], y[..., 1::2]
            rot = mx.stack([even * c - odd * s, odd * c + even * s], axis=-1).reshape(y.shape)
            rot = rb(rot)
            return [(rot[..., :HQ, :], BF_TOL), (rot[..., HQ:, :], BF_TOL)]
        return run, ref, [qkvg]

    def rms_gate(t):
        h = HQ
        x = r((1, t, h, HD)).transpose(0, 2, 1, 3)  # (B, H, L, D), not row contiguous, as sdpa returns it
        gate = r((1, t, 2 * h * HD))[..., h * HD:]  # a slice of qkvg: strides matter
        eps = 1e-6

        def run():
            return outs(k.rms_norm_sigmoid_gate(x, gate, eps=eps))

        def ref():
            xn = f(x).transpose(0, 2, 1, 3)
            xn = xn * inv_rms(xn, eps)
            sig = mx.sigmoid(f(gate).reshape(1, t, h, HD))
            return [(rb(xn * sig).reshape(1, t, h * HD), BF_TOL)]
        return run, ref, [x, gate]

    def omlx_scaled_rms(t, a):
        x, w, eps = r((t, a)), r((a,), mx.bfloat16, 0.1, 0.0), 1e-6

        def run():
            return outs(om.scaled_rms(x, w, eps, True))

        def ref():
            return [(rb(f(x) * inv_rms(f(x), eps) * rb(1.0 + f(w))), BF_TOL)]
        return run, ref, [x, w]

    def omlx_post_attention(t, a):
        x, attn = r((t, a)), r((t, a))
        alpha, beta = r((a,), mx.float32, 0.1, 1.0), r((a,), mx.float32, 0.1, 0.5)
        w, eps = r((a,), mx.bfloat16, 0.1, 0.0), 1e-6

        def run():
            return outs(om.post_attention(x, attn, alpha, beta, w, eps, eps))

        def ref():
            v = f(attn)
            h = rb(v * inv_rms(v, eps))
            xv = rb(alpha * f(x) + beta * h)
            return [(xv, BF_TOL), (rb(xv * inv_rms(xv, eps) * rb(1.0 + f(w))), BF_TOL)]
        return run, ref, [x, attn, alpha, beta, w]

    def omlx_moe_tail(t, a):
        y, scores = r((t, K, a)), softmax_rows((t, K))
        w, eps = r((a,), mx.bfloat16, 0.1, 0.0), 1e-6

        def run():
            return outs(om.moe_tail(y, scores, w, eps))

        def ref():
            total = mx.zeros((t, a))
            for j in range(K):
                v = rb(f(y[:, j]) * rb(1.0 + f(w)))
                total = rb(v * inv_rms(v, eps)) * scores[:, j:j + 1] + total
            return [(rb(total), BF_TOL)]
        return run, ref, [y, scores, w]

    def omlx_post_moe(t, a):
        x, m = r((t, a)), r((t, a))
        alpha, beta = r((a,), mx.float32, 0.1, 1.0), r((a,), mx.float32, 0.1, 0.5)
        w, eps = r((a,), mx.bfloat16, 0.1, 0.0), 1e-6

        def run():
            return outs(om.post_moe(x, m, alpha, beta, w, eps))

        def ref():
            v = rb(f(m) * rb(1.0 + f(w)))
            h = rb(v * inv_rms(v, eps))
            return [(rb(alpha * f(x) + beta * h), BF_TOL)]
        return run, ref, [x, m, alpha, beta, w]

    cases["expert_routing_E256_K8"] = expert_routing
    for d in (4096, 8192):
        for ps in (False, True):
            cases[f"norm_add_norm_D{d}_ps{int(ps)}"] = lambda t, d=d, ps=ps: norm_add_norm_case(t, d, ps, True)
    cases["norm_add_D4096_ps1"] = lambda t: norm_add_norm_case(t, 4096, True, False)
    cases["pre_norm_D4096"] = lambda t: norm_add_norm_case(t, 4096, False, True, pre_only=True)
    for d in (1024, 4096):
        cases[f"post_expert_norm_combine_D{d}_K8"] = lambda t, d=d: post_expert(t, d)
    cases["qk_norm_D128_HQ32_HK8"] = qk_norm_case
    cases["qk_norm_rope_D128_HQ32_HK8_os0"] = lambda t: qk_norm_rope_case(t, False)
    cases["qk_norm_rope_D128_HQ32_HK8_os1"] = lambda t: qk_norm_rope_case(t, True)
    cases["rms_norm_sigmoid_gate_D128"] = rms_gate
    for a in (4096, 8192):
        cases[f"omlx_scaled_rms_AXIS{a}"] = lambda t, a=a: omlx_scaled_rms(t, a)
        cases[f"omlx_post_attention_AXIS{a}"] = lambda t, a=a: omlx_post_attention(t, a)
        cases[f"omlx_moe_tail_AXIS{a}"] = lambda t, a=a: omlx_moe_tail(t, a)
        cases[f"omlx_post_moe_AXIS{a}"] = lambda t, a=a: omlx_post_moe(t, a)
    return cases


def row_error(got, ref):
    """Both float32 mlx arrays: the largest row error, a row's largest absolute difference over its largest |ref|."""
    g, e = np.array(got).astype("float64"), np.array(ref).astype("float64")
    g, e = g.reshape(-1, g.shape[-1]), e.reshape(-1, e.shape[-1])
    return float((np.abs(g - e).max(axis=1) / np.maximum(np.abs(e).max(axis=1), 1e-30)).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--aot", default="aot")
    ap.add_argument("--route", choices=("m2v", "translator", "native"), default="m2v",
                    help="native: macOS MLX, nothing is routed")
    ap.add_argument("--cases", default="", help="comma list; a case runs when its id contains one of them")
    ap.add_argument("--tokens", default=",".join(str(t) for t in TOKENS))
    ap.add_argument("--dump", default="", help="write outputs, references and input hashes to this .npz for qmuse_compare.py")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    setup_env(args)
    import mlx.core as mx

    src = pathlib.Path(args.src)
    if args.selftest:
        mx.set_default_device(mx.cpu)
        k = om = None
    else:
        sys.path.insert(0, str(src))
        import mlxlm as k
        spec = importlib.util.spec_from_file_location("omlx_norm_kernels", src / "omlx_norm_kernels.py")
        om = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(om)
    cases = make_cases(mx, k, om)
    rows = []
    dump = {}
    wanted = [c for c in args.cases.split(",") if c]
    tokens = [int(v) for v in args.tokens.split(",")]
    for cid, builder in cases.items():
        if wanted and not any(w in cid for w in wanted):
            continue
        for t in tokens:
            row = {"id": cid, "t": t}
            try:
                RNG[0] = np.random.default_rng(zlib.crc32(f"{cid}:{t}".encode()))
                run, ref, ins = builder(t)
                with mx.stream(mx.cpu):
                    want = ref()
                    mx.eval([w for w, _ in want])
                fn = (lambda: [w for w, _ in want]) if args.selftest else run
                got = fn()
                mx.eval(got)
                assert len(got) == len(want), f"{len(got)} outputs, reference has {len(want)}"
                row["finite"] = all(bool(np.isfinite(np.array(g.astype(mx.float32))).all()) for g in got)
                if args.dump:
                    key = f"{cid}@T{t}"
                    row["inputs_sha16"] = [hashlib.sha256(np.array(a.astype(mx.float32)).tobytes()).hexdigest()[:16]
                                           for a in ins]
                    for i, (g, (w, tol)) in enumerate(zip(got, want)):
                        dump[f"{key}:out{i}"] = np.array(g.astype(mx.float32))
                        dump[f"{key}:ref{i}"] = np.array(w)
                        dump[f"{key}:tol{i}"] = np.array(tol)
                    dump[f"{key}:inputs"] = np.array(json.dumps(row["inputs_sha16"]))
                errs = [row_error(g.astype(mx.float32), w) for g, (w, _) in zip(got, want)]
                row["err"] = errs
                row["match"] = row["finite"] and all(e <= tol for e, (_, tol) in zip(errs, want))
                samples = []
                for rep in range(args.reps + 1):
                    batch = [fn() for _ in range(args.batch)]
                    t0 = time.perf_counter()
                    mx.eval(batch)
                    if rep:
                        samples.append((time.perf_counter() - t0) / args.batch * 1e3)
                row["ms"] = statistics.median(samples)
                row["bytes"] = sum(a.nbytes for a in ins) + sum(g.nbytes for g in got)
                row["status"] = "ok" if row["match"] else "MISMATCH"
            except Exception as exc:  # a kernel this device cannot run is a result, not a crash
                row["status"] = "ERROR"
                row["error"] = str(exc).strip().splitlines()[0][:240] if str(exc).strip() else type(exc).__name__
            rows.append(row)
            print(f"{cid:42s} T={t:2d} {row['status']:8s} "
                  + (f"{row['ms']:.3f} ms  err {max(row['err']):.1e}" if "ms" in row else row.get("error", "")),
                  flush=True)
    result = {"route": args.route, "device": str(mx.default_device()), "reps": args.reps, "batch": args.batch,
              "mlx": getattr(mx, "__version__", ""), "platform": sys.platform,
              "env": {e: os.environ[e] for e in sorted(os.environ) if e.startswith("MLX_OMARCHY")}, "rows": rows}
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    if args.dump:
        dump["meta"] = np.array(json.dumps({k: v for k, v in result.items() if k != "rows"}))
        np.savez(args.dump, **dump)
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"{len(rows) - len(bad)} of {len(rows)} cases ok" + ("" if not bad else "; not ok: " + ", ".join(
        f"{r['id']}@T{r['t']}={r['status']}" for r in bad)))
    return 1 if bad or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
