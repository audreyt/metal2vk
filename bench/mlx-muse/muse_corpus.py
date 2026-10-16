#!/usr/bin/env python3
"""The Muse fused Metal kernels as mlx generates them, compiled by metal2vk into an AOT directory.

Sources (not in this repo; fetch them read-only and pass the directory as SRC):
  SRC/mlxlm/*.py            ml-explore/mlx-lm mlx_lm/kernels/ at 06e03f7ce7c059ad105483a9dded0bb338cbaf53
  SRC/omlx_norm_kernels.py  jundot/omlx omlx/patches/muse_spark/norm_kernels.py (Apache-2.0)
Each instantiation below is one representative call of the real wrapper (dtype, template values, input rank and
size class), assembled with sweep/mlx-batch/vlmb/assemble.py, whose signature writer ports mlx's metal_kernel.
The key omarchy-mlx looks up is sha256(MSL, NUL, entry name); qmuse.py makes the same calls, so the keys match.

usage: muse_corpus.py build SRC OUTDIR     OUTDIR gets msl/, spv/, aot/<key>.spv|json, manifest.json
needs: CLANG OPT CLSPV SPIRV_VAL in the environment, M2V_COMPILE (tools/m2v-compile). Exit 1 if any instantiation fails.
"""
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
F32, BF16, I32, U32 = "float32", "bfloat16", "int32", "uint32"
BIG = 1 << 20
IMPORT_COMMON = re.compile(r"^from \._common import .*\n", re.MULTILINE)


def S(dt):
    return (dt, 0, 1)  # a 0-d array: a constant


def V(dt, n=BIG):
    return (dt, 1, n)


def M(dt, nd=2):
    return (dt, nd, BIG)


def row_threads(dim):
    return min(1024, max(32, dim // 8 // 32 * 32))


def omlx_layout(axis):
    threads = min(1024, 32 * -(-axis // 128))
    return -(-axis // (4 * threads)), threads


def load_assemble():
    path = os.environ.get("M2V_ASSEMBLE") or str(HERE.parents[1] / "sweep/mlx-batch/vlmb/assemble.py")
    spec = importlib.util.spec_from_file_location("m2v_assemble", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jobs(src):
    """[{id, origin, kernel, entry, msl}] for every instantiation qmuse.py launches."""
    asm = load_assemble()
    src = pathlib.Path(src)
    out = []

    def capture(name):
        common = (src / "mlxlm/_common.py").read_text()
        body = IMPORT_COMMON.sub("", (src / f"mlxlm/{name}.py").read_text())
        with tempfile.NamedTemporaryFile("w", suffix=f"_{name}.py") as f:
            f.write(common + "\n" + body)
            f.flush()
            return asm.capture_module(f.name)[1]

    def reg_named(recorder, name):
        hits = [c for c in recorder.calls if c["name"] == name]
        assert len(hits) == 1, (name, [c["name"] for c in recorder.calls])
        return hits[0]

    def add(jid, origin, reg, table, outputs, template):
        inputs = [{"name": n, "dtype": table[n][0], "ndim": table[n][1], "size": table[n][2]}
                  for n in reg["input_names"]]
        msl, entry = asm.build(reg, inputs, outputs, template)
        out.append({"id": jid, "origin": origin, "kernel": reg["name"], "entry": entry, "msl": msl})

    rec = capture("expert_routing")
    add("expert_routing_E256_K8", "mlx_lm/kernels/expert_routing.py", reg_named(rec, "expert_routing"),
        {"logits": M(F32), "bias": V(F32, 256), "multiplier": S(F32), "eps": S(F32), "scale": S(F32)},
        [U32, F32], [("E", 256), ("K", 8)])
    rec = capture("norm_add_norm")
    na = {"branch": M(BF16, 3), "stream": M(F32, 3), "alpha": V(F32, 4096), "beta": V(F32, 4096),
          "post_w": V(BF16, 4096), "post_eps": S(F32), "pre_w": V(BF16, 4096), "pre_eps": S(F32)}
    for d in (4096, 8192):
        for ps in (False, True):
            # POST_SCALE off: the wrapper passes alpha (float32) as post_w, which changes the kernel name and key
            table = dict(na, post_w=V(BF16, 4096) if ps else V(F32, 4096))
            add(f"norm_add_norm_D{d}_ps{int(ps)}", "mlx_lm/kernels/norm_add_norm.py",
                reg_named(rec, "norm_add_norm"), table, [F32, BF16],
                [("T", BF16), ("D", d), ("TG", row_threads(d)), ("POST_SCALE", ps)])
    add("norm_add_D4096_ps1", "mlx_lm/kernels/norm_add_norm.py", reg_named(rec, "norm_add"), na, [F32],
        [("T", BF16), ("D", 4096), ("TG", row_threads(4096)), ("POST_SCALE", True)])
    add("pre_norm_D4096", "mlx_lm/kernels/norm_add_norm.py", reg_named(rec, "norm_add_norm_pre_norm"),
        {"stream": M(F32, 3), "pre_w": V(BF16, 4096), "pre_eps": S(F32)}, [BF16],
        [("T", BF16), ("D", 4096), ("TG", row_threads(4096))])
    rec = capture("post_expert_norm_combine")
    for d in (1024, 4096):
        add(f"post_expert_norm_combine_D{d}_K8", "mlx_lm/kernels/post_expert_norm_combine.py",
            reg_named(rec, "post_expert_norm_combine"),
            {"experts": M(BF16, 4), "weights": M(F32, 3), "norm_w": V(BF16, d), "eps": S(F32)}, [BF16],
            [("T", BF16), ("D", d), ("K", 8), ("TG", min(1024, 2 * row_threads(d)))])
    rec = capture("qk_norm")
    add("qk_norm_D128_HQ32_HK8", "mlx_lm/kernels/qk_norm.py", reg_named(rec, "qk_norm"),
        {"qkvg": M(BF16, 3), "scale": S(F32), "eps": S(F32)}, [BF16, BF16],
        [("T", BF16), ("D", 128), ("HQ", 32), ("HK", 8)])
    rec = capture("qk_norm_rope")
    for stride in (0, 1):
        add(f"qk_norm_rope_D128_HQ32_HK8_os{stride}", "mlx_lm/kernels/qk_norm_rope.py",
            reg_named(rec, "qk_norm_rope"),
            {"qkvg": M(BF16, 3), "scale": S(F32), "eps": S(F32), "offset": V(I32, 1 if stride == 0 else 4),
             "seq_len": S(I32), "log2_base": S(F32)}, [BF16, BF16],
            [("T", BF16), ("D", 128), ("HQ", 32), ("HK", 8), ("OFFSET_STRIDE", stride)])
    rec = capture("rms_norm_sigmoid_gate")
    add("rms_norm_sigmoid_gate_D128", "mlx_lm/kernels/rms_norm_sigmoid_gate.py", reg_named(rec, "rms_norm_sigmoid_gate"),
        {"x": M(BF16, 4), "g": M(BF16, 3), "eps": S(F32)}, [BF16], [("T", BF16), ("D", 128)])

    mod, rec = asm.capture_module(str(src / "omlx_norm_kernels.py"))
    ns = mod.__dict__
    specs = {
        "scaled_rms": (("x", "w", "eps_in"), ("out",), ns["_SCALED_RMS_SOURCE"],
                       {"x": M(BF16), "w": V(BF16, 4096), "eps_in": V(F32, 1)}, [BF16],
                       [("W", BF16), ("CENTERED", True)]),
        "post_attention": (("x", "a", "alpha", "beta", "w", "eps_in"), ("x_out", "n_out"),
                           ns["_POST_ATTENTION_SOURCE"],
                           {"x": M(BF16), "a": M(BF16), "alpha": V(F32, 4096), "beta": V(F32, 4096),
                            "w": V(BF16, 4096), "eps_in": V(F32, 2)}, [BF16, BF16], [("W", BF16)]),
        "moe_tail": (("y", "scores", "w", "eps_in"), ("out",), ns["_MOE_TAIL_SOURCE"],
                     {"y": M(BF16, 3), "scores": M(F32), "w": V(BF16, 4096), "eps_in": V(F32, 1)}, [BF16],
                     [("W", BF16), ("TOPK", 8)]),
        "post_moe": (("x", "m", "alpha", "beta", "w", "eps_in"), ("x_out",), ns["_POST_MOE_SOURCE"],
                     {"x": M(BF16), "m": M(BF16), "alpha": V(F32, 4096), "beta": V(F32, 4096), "w": V(BF16, 4096),
                      "eps_in": V(F32, 1)}, [BF16], [("W", BF16)]),
    }
    for name, (ins, outs, source, table, odt, extra) in specs.items():
        mod._kernel(name, ins, outs, source)
        reg = rec.calls[-1]
        for axis in (4096, 8192):
            loops, threads = omlx_layout(axis)
            tmpl = [("T", BF16), ("AXIS", axis), ("LOOPS", loops), ("STRIDE", 4 * threads), *extra]
            add(f"omlx_{name}_AXIS{axis}", "omlx/patches/muse_spark/norm_kernels.py", reg, table, odt, tmpl)
    return out


def build(src, outdir):
    out = pathlib.Path(outdir)
    for sub in ("msl", "spv", "aot"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    manifest = []
    for j in jobs(src):
        path = out / "msl" / f"{j['id']}.metal"
        path.write_text(j["msl"])
        key = hashlib.sha256(j["msl"].encode() + b"\0" + j["entry"].encode()).hexdigest()
        r = subprocess.run([sys.executable, os.environ["M2V_COMPILE"], "--msl", str(path), "--out", str(out / "spv"),
                            "--name", j["entry"], "--json"], capture_output=True, text=True, check=False)
        row = {"id": j["id"], "kernel": j["kernel"], "origin": j["origin"], "entry": j["entry"], "key": key,
               "compile": r.returncode}
        lines = r.stdout.strip().splitlines()
        if r.returncode == 0 and not lines:
            row["compile"] = 1
            row["error"] = "m2v-compile exited 0 with no JSON on stdout"
        elif r.returncode == 0:
            res = json.loads(lines[-1])
            for ext in ("spv", "json"):
                (out / "aot" / f"{key}.{ext}").write_bytes(pathlib.Path(res[ext]).read_bytes())
            refl = json.loads((out / "aot" / f"{key}.json").read_text())
            row["caps"] = refl["capabilities"]
        else:
            row["error"] = ((r.stderr or r.stdout).strip().splitlines() or [""])[-1][:300]
        manifest.append(row)
        print(row["id"], row["compile"], row.get("error", "ok"), flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if any(row["compile"] for row in manifest):
        sys.exit("build failed: at least one instantiation did not compile")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
