#!/usr/bin/env python3
"""Compare native Metal timings (macOS JSONs) with translated bench timings (Linux JSONs).

Both sides must be the same chip, cover the same cases, and pass their checks;
otherwise the tool refuses (exit 1) and prints no table. Prints the chip, macOS
version and the Linux kernel, Mesa git driver, CPU governor and ANE state when
the Linux JSONs or --meta KEY=VALUE flags supply them, plus a loud warning when
any recorded load average exceeds 0.5.

Native side:   bench/native-metal results-native-metal-YYYY-MM-DD.json
               (needs "device" (chip) and "load"; number or [1, 5, 15] minutes;
               --native-load / --native-meta supply or override them)
Translated:    bench/results-<device>-<date>.json from bench/m2v-bench.py

Stdlib only.
"""
import argparse
import json
import statistics
import sys

CASES = [
    "activation_f32", "activation_f16", "softmax_f32", "softmax_f16",
    "tilematmul_f32_256", "tilematmul_f32_1024",
    "rt2x2_256", "rt4x2_256", "rt4x4_256", "rt2x2_1024", "rt4x2_1024", "rt4x4_1024",
]
# vulkaninfo deviceName class tags (bench/devices.json): g13c = M1 Max class,
# g13g = 13-inch M1 class, g14c = M2 Max class.
TAG_CHIP = {"g13c": "apple m1 max", "g13g": "apple m1", "g14c": "apple m2 max"}
LOAD_LIMIT = 0.5


def fail(msg):
    print(f"REFUSE: {msg}", file=sys.stderr)
    sys.exit(1)


def read_doc(path):
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        fail(f"cannot read {path}: {e}")
    if not isinstance(doc.get("rows"), dict):
        fail(f"{path} has no rows object")
    return doc


def chip_of(doc):
    """Chip string, lowercased, tag names expanded, '(G14C C0)' suffixes dropped."""
    raw = str(doc.get("device", ""))
    return TAG_CHIP.get(raw.lower(), raw.split("(")[0].strip().lower())


def load1(v):
    """First (1-minute) load average from a number, [1, 5, 15] list, or sysctl string."""
    if isinstance(v, (list, tuple)):
        v = v[0]
    return float(str(v).strip("{} ").split()[0])


def side(paths, kind, overrides):
    """(chip, docs, per-case us lists) for one side; enforces the shared refusals.
    overrides are header fields (e.g. from --native-meta/--native-load) applied
    over each document before the checks."""
    docs = [read_doc(p) for p in paths]
    for d in docs:
        d.update(overrides)
    chips = {chip_of(d) for d in docs}
    if len(chips) != 1 or not next(iter(chips)):
        fail(f"{kind} files disagree on chip or name none: {sorted(chips)} ({', '.join(paths)})")
    times = {}
    for p, d in zip(paths, docs):
        for case, row in d["rows"].items():
            if row.get("ok") is not True:
                fail(f"{kind} file {p}: case {case} check is not OK")
            us = row.get("gpu_us") if kind == "macOS" else None
            if us is None:
                us = row.get("us")
            if not isinstance(us, (int, float)):
                fail(f"{kind} file {p}: case {case} has no us/gpu_us time")
            times.setdefault(case, []).append(float(us))
    return chips.pop(), docs, times


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog="--meta keys: kernel, driver, governor, ane, load. "
                                        "JSON keys of the same name are overridden by --meta.")
    ap.add_argument("--native", nargs="+", required=True, metavar="JSON",
                    help="native Metal result JSON(s), macOS side")
    ap.add_argument("--linux", nargs="+", required=True, metavar="JSON",
                    help="translated bench result JSON(s), Linux side")
    ap.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE",
                    help="Linux-side header field, e.g. --meta governor=performance")
    ap.add_argument("--native-load", type=float, default=None, metavar="LOAD1",
                    help="macOS 1-minute load average the window logged (uptime); "
                         "used when the native JSON has no load, overrides it when it does")
    ap.add_argument("--native-meta", action="append", default=[], metavar="KEY=VALUE",
                    help="macOS-side header field: chip, macos, thermal (overrides the JSON)")
    args = ap.parse_args()

    meta = {}
    for kv in args.meta:
        k, _, v = kv.partition("=")
        if not v:
            fail(f"--meta wants KEY=VALUE, got {kv!r}")
        meta[k.strip().lower()] = v.strip()

    native_meta = {}
    for kv in args.native_meta:
        k, _, v = kv.partition("=")
        if not v:
            fail(f"--native-meta wants KEY=VALUE, got {kv!r}")
        k = k.strip().lower()
        if k not in ("chip", "macos", "thermal"):
            fail(f"--native-meta key must be chip, macos, or thermal, got {k!r}")
        native_meta["device" if k == "chip" else k] = v.strip()
    if args.native_load is not None:
        native_meta["load"] = args.native_load

    nchip, ndocs, ntimes = side(args.native, "macOS", native_meta)
    lchip, ldocs, ltimes = side(args.linux, "Linux", {})
    if nchip != lchip:
        fail(f"chips differ: native {nchip!r} vs linux {lchip!r}; "
             "compare only runs recorded on the same chip")
    for p, d in zip(args.native, ndocs):
        if d.get("load") in (None, ""):
            fail(f"macOS file {p} lacks load field; pass --native-load with the "
                 "load1 the window logged (number, or [1, 5, 15] minutes in JSON)")

    missing = (ntimes.keys() - ltimes.keys()) | (ltimes.keys() - ntimes.keys())
    if missing:
        fail(f"case(s) missing on one side: {', '.join(sorted(missing))}")

    linux = ldocs[0]
    hdr = {k: meta.get(k, linux.get(k)) for k in ("kernel", "driver", "governor", "ane", "load")}
    for k, v in meta.items():
        if k not in hdr:
            hdr[k] = v

    loads = [("macOS", load1(d["load"])) for d in ndocs]
    if hdr.get("load") not in (None, ""):
        loads.append(("Linux", load1(hdr["load"])))

    def fmt(x):
        return f"{x:.1f}"

    print(f"# chip: native {ndocs[0].get('device')} | linux {linux.get('device')} ({nchip})")
    print(f"# macOS: {ndocs[0].get('macos', '(not recorded)')}")
    print(f"# Thermal state (macOS): {ndocs[0].get('thermal') or '(not recorded)'}")
    for label, key in (("Linux kernel", "kernel"), ("Driver (Mesa git)", "driver"),
                       ("CPU governor", "governor"), ("ANE state", "ane")):
        print(f"# {label}: {hdr.get(key) or '(not recorded)'}")
    print(f"# runs: native {len(ndocs)} | linux {len(ldocs)}")
    hot = [(s, v) for s, v in loads if v > LOAD_LIMIT]
    if hot:
        detail = " | ".join(f"{s} {v:.2f}" for s, v in loads)
        print(f"# WARNING: load average above {LOAD_LIMIT} recorded ({detail}) - "
              "not quiet-machine numbers")
    else:
        print(f"# load: all recorded loads at or below {LOAD_LIMIT}")

    print("| case | native GPU us | native spread | translated us | translated spread "
          "| translated/native | MLX hand us |")
    print("|---|---|---|---|---|---|---|")
    for case in sorted(ntimes, key=lambda c: (CASES.index(c) if c in CASES else len(CASES), c)):
        nmed, lmed = statistics.median(ntimes[case]), statistics.median(ltimes[case])
        hand = ldocs[0]["rows"][case].get("hand") or {}
        fused = (hand.get("fused") or {}).get("us")
        unfused = (hand.get("unfused") or {}).get("us")
        ctx = "-"
        if fused is not None:
            ctx = fmt(fused) + (f"/{fmt(unfused)}" if unfused is not None else "")
        ratio = f"{lmed / nmed:.2f}" if nmed else "n/a"
        print(f"| {case} | {fmt(nmed)} | {fmt(max(ntimes[case]) - min(ntimes[case]))} "
              f"| {fmt(lmed)} | {fmt(max(ltimes[case]) - min(ltimes[case]))} | {ratio} | {ctx} |")


if __name__ == "__main__":
    main()
