#!/usr/bin/env python3
"""Read a qbench.py result against a pre-registered bar and print pass or fail.

usage: qread.py RESULT.json --bar BAR.json [--route ROUTE.err] [--baseline SEALED.json]
Prints, per N, the median ms of the deployed arm ("dep") and of each translated arm at the bar's M values, the ratio,
and one verdict per arm. Exit 0 if the bar passes, 1 if it fails, 2 if the result is not valid evidence.

BAR.json (write it and hash it BEFORE the run; the reader prints its sha256):
  {"m": [2, 4, 8, 16], "n": [10240, 12288, 17408], "arms": ["qmvf", "qmvw"], "min_gain": 0.10,
   "max_row_err": 0.01, "driver_sha16": "40282811c6e14ef9", "baseline_drift": 0.15}
An arm passes a cell (N, M) when its median ms <= (1 - min_gain) x the dep median, its q3 is below dep's q1 (the
interleaved rounds separate), and its row error <= max_row_err. An arm passes the bar when it passes EVERY cell.
The bar passes when at least one arm passes. A cell that was not measured fails the arm; it is never skipped.

Not valid evidence (exit 2): the loaded driver set is not exactly {driver_sha16}; the dep trace is missing; the route
file (when given) has a total line other than 'translator=0 fallbacks=0 refusals=0'; with --baseline, a dep median
differs from the sealed run's by more than baseline_drift (the GPU was in another state, so the ratios are not
comparable with the sealed numbers).
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys


def load(path):
    return json.loads(pathlib.Path(path).read_text())


def rows_by(result):
    out = {}
    for shape in result["shapes"]:
        for r in shape["rows"]:
            out[(shape["n"], r["arm"], r["t"])] = r
    return out


def invalid_reasons(result, bar, route_text, sealed):
    why = []
    libs = set(result.get("driver_libs", {}).values())
    if libs != {bar["driver_sha16"]}:
        why.append(f"driver libraries {sorted(libs)} are not exactly {{{bar['driver_sha16']}}}")
    if not result.get("dep_trace"):
        why.append("no dep dispatch trace in the result")
    if route_text is not None:
        total = re.search(r"route summary: m2v=\d+ translator=(\d+) fallbacks=(\d+) refusals=(\d+)", route_text)
        if not total or total.groups() != ("0", "0", "0"):
            why.append(f"route total line is not translator=0 fallbacks=0 refusals=0: {total.group(0) if total else 'missing'}")
    if sealed is not None:
        mine, theirs = rows_by(result), rows_by(sealed)
        for key, row in theirs.items():
            n, arm, t = key
            if arm != "dep" or key not in mine:
                continue
            drift = abs(mine[key]["ms_median"] / row["ms_median"] - 1)
            if drift > bar["baseline_drift"]:
                why.append(f"dep N={n} T={t}: {mine[key]['ms_median']:.3f} ms against sealed {row['ms_median']:.3f} ms "
                           f"({drift:.0%} apart, limit {bar['baseline_drift']:.0%})")
    return why


def judge(result, bar):
    """({arm: [(cell text, ok)]}, table lines)"""
    rows = rows_by(result)
    verdicts = {arm: [] for arm in bar["arms"]}
    lines = []
    for n in bar["n"]:
        lines.append(f"\nN={n} K={result['k']}: median ms per dispatch")
        lines.append("| M | dep | " + " | ".join(f"{a} | {a}/dep" for a in bar["arms"]) + " |")
        lines.append("|---|---|" + "---|---|" * len(bar["arms"]))
        for m in bar["m"]:
            dep = rows.get((n, "dep", m))
            cells = [f"{dep['ms_median']:.3f}" if dep else "not measured"]
            for arm in bar["arms"]:
                r = rows.get((n, arm, m))
                if not dep or not r:
                    verdicts[arm].append((f"N={n} M={m} not measured", False))
                    cells += ["not measured", "-"]
                    continue
                ratio = r["ms_median"] / dep["ms_median"]
                ok_ms = r["ms_median"] <= (1 - bar["min_gain"]) * dep["ms_median"]
                ok_sep = r["ms_q3"] < dep["ms_q1"]
                ok_err = r["row_err_vs_f64"] <= bar["max_row_err"]
                verdicts[arm].append((f"N={n} M={m}: {ratio:.2f}x dep" + ("" if ok_ms else " (gain short)")
                                      + ("" if ok_sep else " (rounds overlap)") + ("" if ok_err else " (row error)"),
                                      ok_ms and ok_sep and ok_err))
                cells += [f"{r['ms_median']:.3f}", f"{ratio:.2f}x"]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return verdicts, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result")
    ap.add_argument("--bar", required=True)
    ap.add_argument("--route", default="")
    ap.add_argument("--baseline", default="")
    a = ap.parse_args()
    bar_bytes = pathlib.Path(a.bar).read_bytes()
    bar = json.loads(bar_bytes)
    result = load(a.result)
    print(f"bar {a.bar} sha256 {hashlib.sha256(bar_bytes).hexdigest()[:16]}: {json.dumps(bar, sort_keys=True)}")
    print(f"result {a.result} sha256 {hashlib.sha256(pathlib.Path(a.result).read_bytes()).hexdigest()[:16]}"
          f" device {result.get('device')} wheel {result.get('mlx')}")
    route_text = pathlib.Path(a.route).read_text() if a.route else None
    sealed = load(a.baseline) if a.baseline else None
    why = invalid_reasons(result, bar, route_text, sealed)
    if result.get("dead"):
        print("arms that did not run:", "; ".join(f"{k} ({v[:90]})" for k, v in result["dead"].items()))
    verdicts, lines = judge(result, bar)
    print("\n".join(lines))
    if why:
        print("\nNOT VALID EVIDENCE:")
        for w in why:
            print("  -", w)
        return 2
    passed = []
    print()
    for arm, cells in verdicts.items():
        bad = [text for text, ok in cells if not ok]
        print(f"{arm}: {'PASS' if not bad else 'FAIL'}" + ("" if not bad else " - " + "; ".join(bad[:6])
                                                          + (f"; and {len(bad) - 6} more" if len(bad) > 6 else "")))
        if not bad:
            passed.append(arm)
    print("BAR: " + ("PASS (" + ", ".join(passed) + ")" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
