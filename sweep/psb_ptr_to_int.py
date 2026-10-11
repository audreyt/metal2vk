#!/usr/bin/env python3
"""Convert pointer-typed loads and stores to 64-bit integer accesses (PSB mode).

clspv's producer has no codegen for a pointer-typed load or store: the LoadInst and
StoreInst handlers reach getSPIRVType on the pointer result type, which is an upstream
unreachable (compiled out in Release, leaving a segfault). What the PSB address model
wants anyway is the bits: the table rows of the kimi/experts kernels hold raw device
addresses, and the producer emits OpConvertUToPtr/OpConvertPtrToU at the SSA edges
(which is why a module whose table GEP is already i64-typed compiles and validates).

This pass rewrites every remaining `load ptr addrspace(1)` / `store ptr addrspace(1)`
in place, keeping the loaded value's name on the inttoptr:

  %r = load ptr addrspace(1), ptr addrspace(N) %p   ->  %r.i = load i64, ... %p
                                                        %r = inttoptr i64 %r.i to ptr addrspace(1)
  store ptr addrspace(1) %v, ptr addrspace(N) %p    ->  %v.i = ptrtoint ptr addrspace(1) %v to i64
                                                        store i64 %v.i, ... %p

Only addrspace(1) values are rewritten; anything else (a pointer of another address
space in memory) is left untouched for clspv, which then fails the module visibly.
Run last in the PSB prep, after psb_promote_ptr_arrays.py demoted the pointer arrays.

Usage: psb_ptr_to_int.py FILE.ll (in place; a module without the shape is a no-op)
"""
import pathlib
import re
import sys

LOAD = re.compile(
    r"^(\s*)(%[\w.]+) = load ptr addrspace\(1\), (ptr(?: addrspace\(\d+\))?) (%[\w.]+), align (\d+)\s*$"
)
STORE = re.compile(
    r"^(\s*)store ptr addrspace\(1\) (%[\w.]+|null), (ptr(?: addrspace\(\d+\))?) (%[\w.]+), align (\d+)\s*$"
)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: psb_ptr_to_int.py FILE.ll")
    path = pathlib.Path(sys.argv[1])
    lines = path.read_text().splitlines(keepends=True)
    out = []
    changed = 0
    seq = 0
    for line in lines:
        m = LOAD.match(line)
        if m:
            indent, dst, pty, ptr, align = m.groups()
            seq += 1
            out.append(f"{indent}%psbl{seq} = load i64, {pty} {ptr}, align {align}\n")
            out.append(f"{indent}{dst} = inttoptr i64 %psbl{seq} to ptr addrspace(1)\n")
            changed += 1
            continue
        m = STORE.match(line)
        if m:
            indent, val, pty, ptr, align = m.groups()
            if val == "null":
                out.append(f"{indent}store i64 0, {pty} {ptr}, align {align}\n")
            else:
                seq += 1
                out.append(f"{indent}%psbs{seq} = ptrtoint ptr addrspace(1) {val} to i64\n")
                out.append(f"{indent}store i64 %psbs{seq}, {pty} {ptr}, align {align}\n")
            changed += 1
            continue
        out.append(line)
    if changed:
        path.write_text("".join(out))
        print(f"psb_ptr_to_int: rewrote {changed} pointer accesses in {path}")
    else:
        print("psb_ptr_to_int: no pointer-typed loads or stores")


if __name__ == "__main__":
    main()
