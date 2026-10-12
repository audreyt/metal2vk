#!/usr/bin/env python3
"""sweep/ggml: run every llama.cpp ggml Metal kernel entry through the metal2vk pipeline.

Corpus: ggml/src/ggml-metal/kernels/*.metal of github.com/ggml-org/llama.cpp at the pinned
commit below (the commit the omarchy-mlx docs cite for llama.cpp numbers). The corpus is
fetched shallow into --src on first run and never committed to this repository; only this
script, the sha and results.tsv are.

One row per kernel entry, where entries are the host_name instantiations
(`template [[host_name("N")]] kernel ...`, including ggml's alias-typedef spelling and the
macro-pasted MUL_MV_MMA_GEN forms) plus plain [[host_name]]-attributed kernels.

llama.cpp builds each kernels/<kind>.metal as a concatenation (ggml-metal/CMakeLists.txt):
common.h, then dequantize.h / quantize.h / fa_common.metal / fa_vec_common.metal when the
source includes them, then the source; the internal include lines and #pragma once are
deleted and ggml-common.h / ggml-metal-impl.h are inlined. This script assembles the same
translation units and compiles each entry through tools/m2v-compile (the sweep pipeline:
metal_stdlib shim, clang, opt passes, patched clspv), then validates the module with
spirv-val --target-env vulkan1.3. Compile only: no GPU dispatch, no numerical check.
"""
import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sweep"))
import m2v_sweep  # noqa: E402

LLAMA_SHA = "65840ed53c8653bfbf3e9014d9cbf71ac8c08725"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp"
CORPUS_SUB = "ggml/src/ggml-metal"

# (label, path relative to ggml-metal) of the files CMake concatenates before a source
PARTS = [("kernels/common.h", pathlib.Path("kernels/common.h")),
         ("kernels/dequantize.h", pathlib.Path("kernels/dequantize.h")),
         ("kernels/quantize.h", pathlib.Path("kernels/quantize.h")),
         ("kernels/fa_common.metal", pathlib.Path("kernels/fa_common.metal")),
         ("kernels/fa_vec_common.metal", pathlib.Path("kernels/fa_vec_common.metal"))]
DROP = re.compile(r'^\s*#include\s*"(?:common\.h|dequantize\.h|quantize\.h|fa_common\.metal|fa_vec_common\.metal)"\s*$'
                  r'|^\s*#pragma\s+once\s*$')
SENTINEL = "__embed_ggml-common.h__"
IMPL_INCLUDE = re.compile(r'^\s*#include\s*"ggml-metal-impl\.h"\s*$')
HOST_ATTR = re.compile(r'\[\[\s*host_name\s*\(\s*((?:"[^"]*"\s*)+)\)\s*\]\]')
MARKER = re.compile(r'^#\s*(\d+)\s+"([^"]*)"')

# failure classes; order matters (first match wins). Classes: the five from the assignment
# (missing-builtin, unsupported-type, clspv-limit, translator-bug, shim-missing-feature)
# plus giveup (M2V-GIVEUP refusals). Refined from the observed first errors of this corpus.
CAUSE_RULES = [
    (r"unknown type name|no type named|no member named|unknown template", "unsupported-type"),
    (r"greater rank than the type of the vector", "shim-missing-feature"),
    (r"pointers to functions are not allowed|forward references to non-struct type|undeclared identifier|"
     r"no matching function|no template named|redeclaration|static declaration|redefinition|"
     r"expected unqualified-id|invalid LLVM IR|cannot initialize a parameter|C-style cast", "translator-bug"),
    (r"WorkgroupStorageSize|exceeds|too (?:large|many)|push constant|descriptor|storage buffer|address space|unsupported", "clspv-limit"),
    (r"spirv-val|structure|blocks? |Op[A-Z]", "translator-bug"),
    (r"clspv|opt:", "clspv-limit"),
]


def first_line(text):
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


ERR_NOISE = re.compile(
    r"^\s*error:\s*(?:\S+[/\\])?\S+?\.(?:cl|metal):\d+:\d+:\s*"  # clang diagnostic on generated text
    r"|^\s*error:\s*(?:\S+[/\\])?\S+?\.(?:cl|metal|ll):\s*"      # stage-level error naming a file
    r"|\s*\(log: .*$")


def diag(text):
    """First diagnostic line without the run-specific file path and cache-log suffix."""
    return re.sub(r"^(?:error:\s*)+", "", ERR_NOISE.sub("", first_line(text))).strip() or first_line(text)


def fetch_corpus(src):
    """Shallow, sparse, read-only fetch of the pinned commit; no-op when already present."""
    if (src / CORPUS_SUB / "kernels" / "mul_mv.metal").is_file():
        return
    src.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    subprocess.run(["git", "-C", str(src), "init", "-q"], check=True, env=env)
    subprocess.run(["git", "-C", str(src), "remote", "add", "origin", LLAMA_REPO], check=True, env=env)
    subprocess.run(["git", "-C", str(src), "sparse-checkout", "init", "--cone"], check=True, env=env)
    subprocess.run(["git", "-C", str(src), "sparse-checkout", "set", "ggml/src"], check=True, env=env)
    subprocess.run(["git", "-C", str(src), "fetch", "--depth", "1", "origin", LLAMA_SHA], check=True, env=env)
    subprocess.run(["git", "-C", str(src), "checkout", "-q", "FETCH_HEAD"], check=True, env=env)


def assemble(kind, corpus, tus):
    """One TU exactly as ggml-metal/CMakeLists.txt builds it -> (path, [(upstream label, line)]).

    The origin list maps every assembled line back to the upstream file and line, so each
    entry's `source` column names the upstream host_name line (macro-generated names land
    on the macro invocation line, which is where upstream writes them)."""
    src = corpus / CORPUS_SUB / "kernels" / (kind + ".metal")
    stext = src.read_text()
    uses = [p for p in PARTS[1:] if re.search(r'#\s*include\s*"%s"' % re.escape(p[0].split("/")[-1]), stext)]
    lines, origins = [], []
    for label, rel in [PARTS[0]] + uses + [("kernels/%s.metal" % kind, src.relative_to(corpus / CORPUS_SUB))]:
        for i, line in enumerate((corpus / CORPUS_SUB / rel).read_text().splitlines(), 1):
            lines.append(line)
            origins.append((label, i))
    kept, keep_orig = [], []
    common_h = None  # inline ggml-common.h at the sentinel and at any plain include of it
    for line, org in zip(lines, origins):
        if DROP.match(line):
            continue
        if SENTINEL in line or re.match(r'^\s*#include\s*"ggml-common\.h"\s*$', line):
            if common_h is None:
                common_h = (corpus / "ggml/src/ggml-common.h").read_text().splitlines()
            for i, l in enumerate(common_h, 1):
                kept.append(l)
                keep_orig.append(("ggml-common.h", i))
            continue
        if IMPL_INCLUDE.match(line):
            for i, l in enumerate((corpus / CORPUS_SUB / "ggml-metal-impl.h").read_text().splitlines(), 1):
                kept.append(l)
                keep_orig.append(("ggml-metal-impl.h", i))
            continue
        kept.append(line)
        keep_orig.append(org)
    out = tus / (kind + ".metal")
    out.write_text("\n".join(kept) + "\n")
    return out, keep_orig


def inventory(tu, origins, corpus, clang):
    """Entries of one assembled TU: [(host name, upstream label, upstream line)] from clang -E
    with line markers kept (macro expansions map to the invocation's upstream line)."""
    cmd = [clang, "--target=spir", "-x", "cl", "-cl-std=clc++2021", "-E", "-w", "-cl-no-stdinc",
           "-I" + str(m2v_sweep.STUB), "-I" + str(tu.parent), str(tu)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        print(f"inventory: clang -E failed on {tu.name}: {first_line(r.stderr)}", file=sys.stderr)
    entries, seen = [], set()
    label, line = tu.name, 0
    for text in r.stdout.splitlines():
        m = MARKER.match(text)
        if m:
            line = int(m.group(1))
            continue
        for h in HOST_ATTR.finditer(text):
            name = "".join(re.findall(r'"([^"]*)"', h.group(1)))
            if name not in seen:
                seen.add(name)
                org = origins[line - 1] if 0 < line <= len(origins) else (tu.name, 0)
                entries.append((name, "%s:%d" % org))
        line += 1
    return entries


def classify(error, giveup):
    if giveup:
        return "giveup"
    for rx, cause in CAUSE_RULES:
        if re.search(rx, error, re.I):
            return cause
    return "unclassified"


def run_entry(tool, cache, out, tu, name, validator, log_lock):
    r = subprocess.run([str(tool), "--msl", str(tu), "--out", str(out), "--name", name, "--json"],
                       text=True, capture_output=True, env=cache_env(cache), check=False)
    row = {"name": name, "compile": "error", "spirv_val": "n/a", "error": "", "giveup": "", "cause": ""}
    try:
        result = json.loads(first_line(r.stdout))
    except json.JSONDecodeError:
        result = {}
    if r.returncode == 3 or result.get("status") == "refused":
        row["compile"] = "refused"
        row["giveup"] = result.get("refused", "M2V-GIVEUP")
        row["error"] = diag(r.stderr) or result.get("refused", "")
    elif r.returncode == 0 and result.get("status") == "ok":
        row["compile"] = "ok"
        check = subprocess.run([str(validator), "--target-env", "vulkan1.3", result["spv"]],
                               text=True, capture_output=True, check=False)
        if check.returncode == 0:
            row["spirv_val"] = "pass"
        else:
            row["spirv_val"] = "fail"
            row["error"] = diag(check.stderr or check.stdout)
    else:
        row["error"] = diag(r.stderr) or result.get("detail", "compile failed")
    if row["spirv_val"] == "pass":
        row["cause"] = "-"
    else:
        row["cause"] = classify(row["error"], row["giveup"])
    with log_lock:
        print(f"{row['compile']:>7} {row['spirv_val']:>4} {row['cause']:<22} {name}", flush=True)
    return row


def cache_env(cache):
    env = os.environ.copy()
    env["M2V_CACHE_DIR"] = str(cache)
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=pathlib.Path, required=True, help="corpus checkout (fetched at the pin if missing)")
    ap.add_argument("--tool", type=pathlib.Path, default=REPO / "tools" / "m2v-compile")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="scratch dir for TUs, spv, progress")
    ap.add_argument("--cache", type=pathlib.Path, default=None)
    ap.add_argument("--results", type=pathlib.Path, required=True)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--fresh", action="store_true", help="ignore progress from an earlier partial run")
    args = ap.parse_args()

    clang = os.environ.get("CLANG", m2v_sweep.CLANG)
    validator = os.environ.get("SPIRV_VAL", shutil.which("spirv-val") or "spirv-val")
    cache = args.cache or args.out / "cache"
    tus = args.out / "tus"
    tus.mkdir(parents=True, exist_ok=True)
    fetch_corpus(args.src)

    kernels = args.src / CORPUS_SUB / "kernels"
    kinds = sorted(p.stem for p in kernels.glob("*.metal"))
    entries, seen = [], set()
    for kind in kinds:
        tu, origins = assemble(kind, args.src, tus)
        for name, origin in inventory(tu, origins, args.src, clang):
            if name not in seen:
                seen.add(name)
                entries.append({"name": name, "source": origin, "tu": str(tu)})
    print(f"kinds={len(kinds)} entries={len(entries)}")

    progress = args.out / "progress.jsonl"
    done = set()
    if progress.is_file() and not args.fresh:
        for line in progress.read_text().splitlines():
            try:
                done.add(json.loads(line)["name"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [e for e in entries if e["name"] not in done]
    print(f"todo={len(todo)} (done={len(done)})", flush=True)

    log_lock = threading.Lock()
    with progress.open("a") as pf:
        def work(e):
            row = run_entry(args.tool, cache, args.out, pathlib.Path(e["tu"]), e["name"], validator, log_lock)
            row["source"] = e["source"]
            with log_lock:
                pf.write(json.dumps(row) + "\n")
                pf.flush()
            return row

        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            list(ex.map(work, todo))

    rows = {}
    if progress.is_file():
        for line in progress.read_text().splitlines():
            try:
                r = json.loads(line)
                rows[r["name"]] = r
            except json.JSONDecodeError:
                pass
    missing = [e["name"] for e in entries if e["name"] not in rows]
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w") as f:
        f.write("name\tsource\tcompile\tspirv_val\tfirst_error\tm2v_giveup\tcause\n")
        for e in sorted(entries, key=lambda e: (e["source"], e["name"])):
            r = rows.get(e["name"])
            if not r:
                f.write(f"{e['name']}\t{e['source']}\nerror\tn/a\tmissing row (run interrupted)\t\tgiveup\n")
                continue
            f.write("\t".join([r["name"], r["source"], r["compile"], r["spirv_val"],
                               r["error"].replace("\t", " "), r["giveup"] or "-", r["cause"] or "-"]) + "\n")
    valid = sum(1 for r in rows.values() if r["spirv_val"] == "pass")
    causes = {}
    for e in entries:
        r = rows.get(e["name"])
        if not r or r["spirv_val"] != "pass":
            causes[r["cause"] if r else "missing"] = causes.get(r["cause"] if r else "missing", 0) + 1
    print(f"entries={len(entries)} valid={valid} failed={len(entries) - valid} missing={len(missing)}")
    for cause, n in sorted(causes.items(), key=lambda kv: -kv[1]):
        print(f"  {cause}: {n}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
