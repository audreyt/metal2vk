#!/usr/bin/env python3
"""sweep/mlx-core: MLX's own library kernels (backend/metal/kernels at a pinned sha) through the m2v pipeline.

Unlike sweep/mlx-batch (assembled call-site kernels), this corpus is the .metal sources MLX ships:
the macro instantiations (instantiate_kernel, instantiate_unary_all, ...) expand into
`template [[host_name("...")]] [[kernel]] decltype(f<A...>) f<A...>;` entries, and those generated
names are the rows. Plain `[[kernel]]` definitions in the same files (fence, random, the base
library) are entries too, exactly as m2v_sweep's inventory finds them.

Phases (all three run by default, in order):
  expand   clang -E (line markers kept) over every pinned .metal with the shim's empty Metal-header
           stand-ins and -I <MLX_SRC>, so the instantiation macros expand. The expanded text goes to
           WORK/<flat>.metal (self-contained: no includes left), and find_entries/variant_sets from
           sweep/m2v_sweep.py - the same discovery the sweep and CI use - turn it into the entry
           manifest: one row per (work file, kernel name) with the origin file:line of the
           instantiation, taken from the line markers.
  compile  tools/m2v-compile per entry (--name selects the instantiation; the other entries of the
           file are dropped from its translation unit), CLSPV pointing at the patched clspv.
           spirv-val --target-env vulkan1.3 re-validates each produced module.
  results  sweep/mlx-core/results.tsv: name, origin file:line, family, compile status, spirv-val
           status, first error line, M2V-GIVEUP, cause class. Compile only: nothing here dispatches
           on a GPU.

Usage:
  python3 sweep/mlx-core/mlx_core_run.py --mlx ~/scratch/sweep/mlx-batch-src/mlx \
      --work ~/scratch/sweep/mlxcore-run/work --out ~/scratch/sweep/mlxcore-run/out \
      [--jobs N] [--skip-expand] [--limit N] [--compile-only NAME]

env: CLANG (default clang-19), OPT (default opt-19), CLSPV (patched clspv, required for the compile
     phase), SPIRV_VAL (default spirv-val), M2V_CACHE_DIR, M2V_INCLUDE (the shim; default the
     repo's include/), M2V_CLSPV_TIMEOUT (m2v-compile --timeout, default 240 s).
"""
import argparse
import bisect
import concurrent.futures as cf
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "sweep"))
import m2v_sweep  # noqa: E402  (blank_comments, find_entries, variant_sets: the sweep's own discovery)

MLX_PIN = "1f8e74e3f12f31365464a6867c6579f0e9b29d85"
MARKER = re.compile(r'^# (\d+) "([^"]*)"')
HOST_NAME = re.compile(r"\[\[\s*host_name\s*\(")


def flat_name(rel):
    stem = rel[: -len(".metal")] if rel.endswith(".metal") else rel
    return stem.replace("/", "__") + ".metal"


def family_of(rel):
    """Kernel family for the group table: steel subdirs share one family, everything else is the file stem."""
    parts = rel.split("/")
    if parts[0] == "steel" and len(parts) > 1:
        return "steel_" + parts[1]
    return parts[-1][: -len(".metal")]


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


def expand(mlx_root, work):
    """Every pinned .metal -> expanded text in work/ plus manifest rows [(file, name, origin, line)]."""
    kernels = pathlib.Path(mlx_root) / "mlx/backend/metal/kernels"
    clang = os.environ.get("CLANG", "clang-19")
    with tempfile.TemporaryDirectory(prefix="m2v-mlxcore-stubs-") as stub:
        stub = pathlib.Path(stub)
        # the same empty stand-ins the sweep's preprocess generates for the Metal system headers
        for n in [f.name for f in m2v_sweep.INC.iterdir() if f.name.startswith(("metal_", "simd"))] + ["metal_stdlib"]:
            (stub / n).write_text("")
        (stub / "MetalPerformancePrimitives").mkdir()
        (stub / "MetalPerformancePrimitives" / "MetalPerformancePrimitives.h").write_text("")
        rows, bad_files = [], []
        files = sorted(kernels.rglob("*.metal"))
        for f in files:
            rel = str(f.relative_to(kernels))
            r = run([clang, "--target=spir", "-x", "cl", "-cl-std=clc++2021", "-E", "-w", "-cl-no-stdinc",
                     "-I" + str(stub), "-I" + str(mlx_root), f])
            if r.returncode != 0:
                bad_files.append({"file": rel, "error": (r.stderr.strip().splitlines() or ["?"])[0][:200]})
                continue
            text = r.stdout
            lines = text.splitlines(keepends=True)
            # char offset -> output line -> (file, line) of the pin, via the line markers clang keeps
            starts, off, cur = [], 0, ("?", 0)
            origin_of_line = []
            for ln in lines:
                m = MARKER.match(ln)
                if m:
                    cur = (m.group(2), int(m.group(1)))
                starts.append(off)
                origin_of_line.append(cur)
                off += len(ln)

            def origin_at(char_off):
                i = bisect.bisect_right(starts, char_off) - 1
                return origin_of_line[max(i, 0)]

            # manifest from the sweep's own entry discovery, so one manifest row == one sweep job
            blanked = m2v_sweep.blank_comments(text)
            ents = m2v_sweep.find_entries(blanked)
            work_rows, seen = [], set()
            for e in ents:
                for kname, _targs, err in m2v_sweep.variant_sets(e, blanked):
                    if err:
                        continue  # template kernel without an instantiation: not an entry
                    if kname in seen:
                        continue  # duplicate generated name: m2v-compile --name would refuse it anyway
                    seen.add(kname)
                    span = e["inst"][0]["span"][0] if e["inst"] else e["tmpl"][0] if e["tmpl"] else e["head"][0]
                    ofile, oline = origin_at(span)
                    if not pathlib.Path(ofile).is_absolute():
                        cand = kernels / ofile
                        ofile, oline = (str(cand), oline) if cand.exists() else (str(f), 0)
                    if not pathlib.Path(ofile).resolve().is_relative_to(pathlib.Path(mlx_root).resolve()):
                        ofile, oline = str(f), 0  # instantiation spelled outside the pin: attribute to the file
                    work_rows.append({"name": kname, "origin": str(pathlib.Path(ofile).resolve().relative_to(mlx_root)),
                                      "line": oline})
            if not work_rows:
                continue  # header-only after expansion: nothing to compile
            wf = work / (flat_name(rel) + ".metal")
            wf.write_text(text)
            for r2 in work_rows:
                r2["work"] = wf.name
                r2["family"] = family_of(rel)
            rows += work_rows
            print(f"  {rel}: {len(work_rows)} entries", file=sys.stderr)
    (work / "manifest.json").write_text(json.dumps({"pin": MLX_PIN, "rows": rows, "bad_files": bad_files}, indent=1))
    print(f"expanded {len(files)} files -> {len(rows)} entries, {len(bad_files)} files failed to expand", file=sys.stderr)
    return rows, bad_files


def classify(rc, status, detail, spv_error):
    """Cause class from the failure text: the task's five classes plus the bookkeeping ones."""
    err = (spv_error or detail or "").lower()
    if rc == 3 or status == "refused":
        return "refused"
    if "timeout" in err:
        return "clspv_timeout"
    if "m2v-giveup" in err or "gave up on a pointer rewrite" in err:
        return "clspv_limit"
    if "matches" in err and "instantiations" in err:
        return "duplicate_host_name"  # two template instantiations share one host_name (select time)
    if "clang frontend command failed" in err:
        return "clang_frontend_exit"  # clang exited nonzero without a usable diagnostic (139/134 under load)
    if "an attribute list cannot appear here" in err:
        return "attribute_list_position"
    if "conditional expression is ambiguous" in err:
        return "ambiguous_conditional"
    if "excess elements" in err:
        return "excess_elements_init"
    if "too few elements in vector" in err:
        return "vector_element_count"
    if any(k in err for k in ("assertion", "unreachable", "stack dump", "llvm error", "crash", "iselecode", "fatal error: backend")):
        return "translator_bug"
    if any(k in err for k in ("unknown type name", "no member named", "no matching", "no template named",
                              "use of undeclared", "no namespace named", "unknown namespace", "must be a constant",
                              "address space", "cannot be overlapped", "redeclaration")):
        return "shim_gap"
    if "builtin" in err or "spirv" in err and "unimplemented" in err:
        return "missing_builtin"
    if any(k in err for k in ("unsupported", "not supported", "cannot allocate", "exceeds", "too large", "limit",
                              "no suitable", "functionally", "bitcast of a non-numeric")):
        return "unsupported_type" if "type" in err or "unsupported operand" in err else "clspv_limit"
    if "spirv-val" in err or "valid id" in err or "was not declared" in err:
        return "translator_bug"
    return "other"


def compile_rows(rows, args):
    """One m2v-compile per entry, in a thread pool; each ok module is re-validated for the TSV column."""
    env = os.environ.copy()
    tool = REPO / "tools/m2v-compile"
    val = env.get("SPIRV_VAL", "spirv-val")
    jobs = int(args.jobs or (os.cpu_count() or 4))
    results = [None] * len(rows)

    def one(i_row):
        i, row = i_row
        work = args.work / row["work"]
        r = run([sys.executable, tool, "--msl", work, "--out", args.out, "--name", row["name"],
                 "--json", "--timeout", str(args.timeout)], env=env)
        try:
            js = json.loads(next((ln for ln in r.stdout.splitlines() if ln.startswith("{")), "{}"))
        except json.JSONDecodeError:
            js = {}
        first = (r.stderr.strip().splitlines() or [""])[0]
        detail = str(js.get("detail", "")) or first
        val_status, spv_error = "", ""
        if r.returncode == 0 and js.get("status") == "ok":
            v = run([val, "--target-env", "vulkan1.3", js["spv"]])
            val_status = "ok" if v.returncode == 0 else "FAIL"
            spv_error = (v.stderr.strip().splitlines() or [""])[0]
        results[i] = {
            "name": row["name"], "origin": row["origin"], "line": row["line"], "family": row["family"],
            "compile": "ok" if r.returncode == 0 and js.get("status") == "ok" else "FAIL",
            "spirv_val": val_status, "first_error": (spv_error or detail)[:200],
            "m2v_giveup": str(js.get("refused", "no")) if (r.returncode == 3 or js.get("status") == "refused") else "no",
            "cause": "" if (r.returncode == 0 and js.get("status") == "ok" and val_status == "ok")
                    else classify(r.returncode, js.get("status"), detail, spv_error),
        }
        return r.returncode

    with cf.ThreadPoolExecutor(jobs) as ex:
        for n, _rc in enumerate(ex.map(one, enumerate(rows))):
            if (n + 1) % 100 == 0:
                print(f"  {n + 1}/{len(rows)}", file=sys.stderr)
    return results


def write_tsv(results, path, bad_files):
    cols = ["origin", "name", "family", "compile", "spirv_val", "first_error", "m2v_giveup", "cause"]

    def clean(s):
        return str(s).replace("\t", " ").replace("\n", " | ")[:200]

    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in sorted(results, key=lambda r: (r["family"], r["origin"], r["line"], r["name"])):
            origin = f"{r['origin']}:{r['line']}"
            f.write(origin + "\t" + "\t".join(clean(r[c]) for c in cols[1:]) + "\n")
        for b in bad_files:
            f.write(f'{b["file"]}:0\t(file failed to expand)\t\tFAIL\t\t{clean(b["error"])}\tno\texpand_error\n')
    ok = sum(r["compile"] == "ok" and r["spirv_val"] == "ok" for r in results)
    print(f"entries {len(results)} valid {ok} failed {len(results) - ok + len(bad_files)}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mlx", required=True, metavar="MLX_SRC", help="pinned MLX checkout (mlx/backend/metal/kernels)")
    ap.add_argument("--work", required=True, type=pathlib.Path, help="directory for the expanded .metal files")
    ap.add_argument("--out", required=True, type=pathlib.Path, help="m2v-compile output directory")
    ap.add_argument("--results", type=pathlib.Path, default=HERE / "results.tsv")
    ap.add_argument("--jobs", default=0, type=int, help="parallel compiles (default: cpu count)")
    ap.add_argument("--timeout", default=os.environ.get("M2V_CLSPV_TIMEOUT", "240"), help="m2v-compile per-stage seconds")
    ap.add_argument("--skip-expand", action="store_true", help="reuse work/manifest.json from an earlier expand run")
    ap.add_argument("--limit", type=int, default=0, help="compile only the first N entries (smoke run)")
    ap.add_argument("--compile-only", default="", metavar="SUBSTR", help="compile only entries whose work file or name contains SUBSTR")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.skip_expand:
        man = json.loads((args.work / "manifest.json").read_text())
        rows, bad_files = man["rows"], man["bad_files"]
        print(f"manifest: {len(rows)} entries", file=sys.stderr)
    else:
        rows, bad_files = expand(args.mlx, args.work)
    if not os.environ.get("CLSPV"):
        print("CLSPV is not set; expanded manifest is complete, compile phase skipped", file=sys.stderr)
        write_tsv([dict(r, compile="", spirv_val="", cause="not_compiled") for r in rows], args.results, bad_files)
        return
    if not pathlib.Path(os.environ["CLSPV"]).exists():
        sys.exit(f"error: CLSPV={os.environ['CLSPV']} does not exist")
    if args.compile_only:
        rows = [r for r in rows if args.compile_only in r["work"] or args.compile_only in r["name"]]
    if args.limit:
        rows = rows[: args.limit]
    results = compile_rows(rows, args)
    write_tsv(results, args.results, bad_files)


if __name__ == "__main__":
    main()
