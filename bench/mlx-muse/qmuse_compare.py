#!/usr/bin/env python3
"""Compare two qmuse.py dumps: the same Muse kernels run on macOS MLX (native Metal) and on Linux omarchy-mlx (metal2vk).

usage: qmuse_compare.py A.npz B.npz [--tol 2e-2]
Each dump holds, per case and token count, the kernel outputs (float32 copies of the stored dtype), the CPU float32
references and the sha256 of every input. The reader checks, case by case:
  - the inputs are byte-identical on both sides (else NOT VALID: the two runs did not see the same data),
  - A against B: the largest row error, the fraction of elements that are bit-identical, and the largest absolute difference,
  - each side against its own CPU reference, and the two references against each other (they should be near-identical;
    if not, the CPU stream differs between the platforms and the A-versus-B number cannot be trusted alone).
A case agrees when the A-versus-B row error is at most --tol (default 2e-2, the tolerance qmuse.py uses against the reference).
Exit 0 when every case agrees, 1 when one does not, 2 when the dumps are not comparable.
Row error = a row's largest absolute difference over that row's largest absolute reference value.
"""
import argparse
import json
import sys

import numpy as np


def row_error(a, b):
    a, b = a.astype("float64"), b.astype("float64")
    a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    return float((np.abs(a - b).max(axis=1) / np.maximum(np.abs(b).max(axis=1), 1e-30)).max())


def cases(dump):
    return sorted({k.split(":")[0] for k in dump.files if k != "meta"})


def outputs(dump, case):
    n = 0
    while f"{case}:out{n}" in dump.files:
        n += 1
    return n


def compare(a, b, tol):
    """(report lines, exit code)"""
    lines, code = [], 0
    ma, mb = json.loads(str(a["meta"])), json.loads(str(b["meta"]))
    lines.append(f"A: {ma['platform']} {ma['device']} mlx {ma['mlx']} route {ma['route']}")
    lines.append(f"B: {mb['platform']} {mb['device']} mlx {mb['mlx']} route {mb['route']}")
    ca, cb = cases(a), cases(b)
    if not ca or ca != cb:
        return lines + [f"NOT COMPARABLE: case sets differ or are empty ({ca} vs {cb})"], 2
    lines.append("| case | out | A vs B row err | identical | max abs diff | A vs refA | B vs refB | refA vs refB |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for case in ca:
        if json.loads(str(a[f"{case}:inputs"])) != json.loads(str(b[f"{case}:inputs"])):
            return lines + [f"NOT COMPARABLE: the inputs of {case} differ between A and B"], 2
        n = outputs(a, case)
        if n == 0 or n != outputs(b, case):
            return lines + [f"NOT COMPARABLE: {case} has {n} outputs in A and {outputs(b, case)} in B"], 2
        for i in range(n):
            oa, ob = a[f"{case}:out{i}"], b[f"{case}:out{i}"]
            ra, rb = a[f"{case}:ref{i}"], b[f"{case}:ref{i}"]
            if oa.shape != ob.shape or not (np.isfinite(oa).all() and np.isfinite(ob).all()):
                lines.append(f"| {case} | {i} | shape or non-finite mismatch | | | | | |")
                code = max(code, 1)
                continue
            err = row_error(oa, ob)
            ok = err <= tol
            code = max(code, 0 if ok else 1)
            lines.append(f"| {case} | {i} | {err:.2e}{'' if ok else ' FAIL'} | {float((oa == ob).mean()):.4f} | "
                         f"{float(np.abs(oa.astype('float64') - ob).max()):.3e} | {row_error(oa, ra):.2e} | "
                         f"{row_error(ob, rb):.2e} | {row_error(ra, rb):.2e} |")
    return lines, code


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=2e-2)
    args = ap.parse_args()
    try:
        with np.load(args.a) as a, np.load(args.b) as b:
            lines, code = compare(a, b, args.tol)
    except (OSError, KeyError, ValueError) as exc:
        print(f"NOT COMPARABLE: {type(exc).__name__}: {exc}")
        return 2
    print("\n".join(lines))
    print({0: "AGREE", 1: "DISAGREE", 2: "NOT COMPARABLE"}[code])
    return code


if __name__ == "__main__":
    sys.exit(main())
