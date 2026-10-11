#!/usr/bin/env python3
"""Merge assembly + compile outcomes into results-vlmb.tsv."""
import importlib.util
import os
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
BATCH = HERE.parent

_spec = importlib.util.spec_from_file_location("assemble", HERE / "assemble.py")
assemble = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(assemble)


_modcache = {}


def _modns(spec):
    key = spec["module"]
    if not os.path.isabs(key):
        key = str(assemble.PINNED_ROOT / key)
    if key not in _modcache:
        _modcache[key] = assemble.capture_module(key, spec.get("mx_file"))
    return _modcache[key]


def metal_for(spec, spec_stem):
    """The assembled file is named by the captured registration name."""
    mod_ns, rec = _modns(spec)
    if spec.get("capture") is not None:
        name = rec.calls[spec["capture"]]["name"]
    else:
        assemble.call_factory(mod_ns, rec, spec["factory"], spec.get("args", []))
        name = rec.calls[-1]["name"]
    got = HERE / f"{name}.metal"
    if not got.exists():  # fall back to the spec-stem file (capture specs of old runs)
        got = HERE / f"{spec_stem}.metal"
    return got

CAUSE_HINTS = [
    ("cannot assemble", "cannot assemble"),
    ("refused", "m2v refused: Metal 4 / untranslatable construct"),
    ("spirv-val failed", "spirv-val failure"),
    ("timed out", "toolchain timeout"),
]

CAUSE_OVERRIDES = {
    "nemotron_bm32_steel_linear_nt":
        "upstream source does not compile standalone: expanded steel gemm headers "
        "reference complex64_t, defined only by MLX's device prelude (kernels/utils.h)",
    "quantized_verify_nvfp4_argmax_t2_k64_n64_fp16":
        "clspv/translator bug: OpSelect with workgroup array-pointer result type fails spirv-val",
}


def cause_for(first_error, stem=""):
    if not first_error:
        return ""
    if stem in CAUSE_OVERRIDES:
        return CAUSE_OVERRIDES[stem]
    low = first_error.lower()
    if "undefined" in low and ("identifier" in low or "function" in low):
        return "missing builtin/header"
    if "bitcode" in low or "unsupported" in low and "type" in low:
        return "unsupported type"
    if "clspv" in low:
        return "clspv limit"
    return "compile error (see detail)"

rows = []
for spec_path in sorted((HERE / "specs").glob("*.json")):
    spec = json.loads(spec_path.read_text())
    metal = metal_for(spec, spec_path.stem)
    if not metal.exists():
        print(f"no assembled file for {spec_path.stem}")
        continue
    sha = hashlib.sha256(metal.read_bytes()).hexdigest()
    stem = metal.stem  # compile-results.txt keys on the assembled file's basename
    first_error = ""
    status = ""
    results = BATCH / "vlmb" / "compile-results.txt"
    if results.exists():
        for line in results.read_text().splitlines():
            parts = line.split("\t")
            if parts[0] == stem:
                status = parts[1]
                if status != "ok":
                    first_error = parts[2] if len(parts) > 2 else ""
                break
    valid = "yes" if status == "ok" else ("no" if status else "missing")
    rows.append([
        stem, spec["origin"], sha,
        status or "missing",                               # compile stage
        "" if status != "ok" else "ok" if status == "ok" else "fail",  # spirv_val
        valid, first_error.replace("\t", " "), "no",       # first_error, m2v_giveup
        cause_for(first_error, stem),
        spec["call"].replace("\t", " "),
    ])

out = HERE / "results-vlmb.tsv"
with out.open("w") as f:
    f.write("kernel\torigin\tsource_sha256\tcompile\tspirv_val\tvalid\tfirst_error\t"
            "m2v_giveup\tfailure_cause\trepresentative_call\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
print(f"{out}: {len(rows)} rows")
