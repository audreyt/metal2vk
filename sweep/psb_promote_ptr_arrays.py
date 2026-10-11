#!/usr/bin/env python3
"""Promote private arrays of device pointers to slot allocas with select chains (PSB mode).

The kimi/experts kernels keep the per-lane weight pointers of the current table row in a
private array (`alloca [N x ptr addrspace(1)]`): the stores pick static slots, a helper
reads the slots back through a loop index (a dynamic `getelementptr [N x ptr
addrspace(1)]` whose base escaped through an `addrspacecast ptr %arr to ptr
addrspace(4)`). clspv's producer cannot generate a SPIR-V type for pointer-typed storage
(getSPIRVType on a pointer type is an unreachable that Release compiles out, leaving a
segfault), so the array must not reach it.

This pass rewrites the array exactly like sweep/promote_ptr_arrays.py rewrites the
buffer-select tables, plus the two shapes that pass does not match: stores that address
the array directly or through a constant-index element GEP, and loads whose GEP base is
the generic-address-space cast of the array.

  store ptr addrspace(1) %v, ptr %arr                  -> store to slot 0
  %g = getelementptr inbounds ptr addrspace(1), ptr %arr, i64 <c>
  store ptr addrspace(1) %v, ptr %g                    -> store to slot <c>
  %c = addrspacecast ptr %arr to ptr addrspace(4)
  %g = getelementptr inbounds [N x ptr addrspace(1)], ptr addrspace(4) %c, i64 0, i64 %i
  %l = load ptr addrspace(1), ptr addrspace(4) %g      -> per-slot loads plus an
                                                          icmp/select chain over %i; a
                                                          single slot loads straight

Each slot starts as null, so an out-of-range index yields null exactly where the source
had undefined behaviour. The gep and the cast are dropped when nothing reads them anymore.
Anything else (a dynamic store, a second user of the cast, more than 16 slots) is left
untouched for clspv, which then fails the module visibly instead of miscompiling it.
Run after the dead row-copy structs are removed (opt -passes=function(dse)), which is what
leaves every surviving pointer alloca in the shape this pass matches.

Usage: psb_promote_ptr_arrays.py FILE.ll (in place; a module without the shape is a no-op)
"""
import pathlib
import re
import sys

MAX_SLOTS = 16

ALLOCA = re.compile(
    r"^(\s*)(%[\w.]+) = alloca \[(\d+) x ptr addrspace\(1\)\], align (\d+)\s*$"
)
CAST = re.compile(
    r"^(\s*)(%[\w.]+) = addrspacecast ptr (%[\w.]+) to ptr addrspace\(4\)\s*$"
)
GEP_DYN = re.compile(
    r"^(\s*)(%[\w.]+) = getelementptr (?:inbounds )?\[(\d+) x ptr addrspace\(1\)\], "
    r"ptr addrspace\(4\) (%[\w.]+), i64 0, (i32|i64) (%[\w.]+)\s*$"
)
GEP_CONST = re.compile(
    r"^(\s*)(%[\w.]+) = getelementptr (?:inbounds )?ptr addrspace\(1\), ptr (%[\w.]+), i64 (\d+)\s*$"
)
STORE = re.compile(
    r"^(\s*)store ptr addrspace\(1\) (%[\w.]+|null), ptr (%[\w.]+), align (\d+)\s*$"
)
LOAD = re.compile(
    r"^(\s*)(%[\w.]+) = load ptr addrspace\(1\), ptr addrspace\(4\) (%[\w.]+), align (\d+)\s*$"
)


def slot(arr, j):
    return f"%psb{arr[1:]}.s{j}"


def rewrite_function(chunk):
    """chunk: lines of one function (header included). Returns (new lines, changed count)."""
    arrays = {}
    for line in chunk:
        m = ALLOCA.match(line)
        if m and 1 <= int(m.group(3)) <= MAX_SLOTS:
            arrays[m.group(2)] = int(m.group(3))
    if not arrays:
        return chunk, 0

    # const-index element geps on the arrays (static store addressing)
    const_geps = {}  # gep name -> (array, slot)
    for line in chunk:
        m = GEP_CONST.match(line)
        if m and m.group(3) in arrays:
            const_geps[m.group(2)] = (m.group(3), int(m.group(4)))

    # dynamic geps through the generic cast of an array
    casts = {}  # cast name -> array
    for line in chunk:
        m = CAST.match(line)
        if m and m.group(3) in arrays:
            casts[m.group(2)] = m.group(3)
    dyn_geps = {}  # gep name -> (array, idx type text, idx value)
    for line in chunk:
        m = GEP_DYN.match(line)
        if m and m.group(4) in casts:
            dyn_geps[m.group(2)] = (casts[m.group(4)], m.group(5), m.group(6))
    if not dyn_geps:
        return chunk, 0

    # classify every consumer; an array with a store or load this pass cannot rewrite
    # (a dynamic store, a second use of a gep, anything else naming its geps) is skipped
    load_geps, store_geps, bad = {}, {}, set()
    for i, line in enumerate(chunk):
        m = LOAD.match(line)
        if m and m.group(3) in dyn_geps:
            if m.group(3) in load_geps:
                bad.add(m.group(3))
            else:
                load_geps[m.group(3)] = i
            continue
        m = STORE.match(line)
        tgt = m.group(3) if m else None
        if tgt in dyn_geps:
            bad.add(tgt)  # dynamic store: not rewritten
            continue
        if m and tgt in const_geps:
            g = const_geps[tgt]
            if tgt in store_geps:
                bad.add(tgt)
            else:
                store_geps[tgt] = i
            continue
        # any other mention of a managed gep or cast name disqualifies it
        for name in list(dyn_geps) + list(const_geps) + list(casts):
            if name in line and not re.match(rf"^\s*{re.escape(name)} = ", line):
                if name in dyn_geps:
                    bad.add(name)
                if name in const_geps:
                    bad.add(name)
                if name in casts:
                    bad.add(name)
    if bad:
        for name in bad:
            if name in dyn_geps:
                arrays.pop(dyn_geps[name][0], None)
    if not arrays:
        return chunk, 0

    seq = 0
    out = []
    changed = 0
    dead_names = set()
    for line in chunk:
        m = ALLOCA.match(line)
        if m and m.group(2) in arrays:
            out.append(line)
            indent, name, slots = m.group(1), m.group(2), arrays[m.group(2)]
            for j in range(slots):
                out.append(f"{indent}{slot(name, j)} = alloca ptr addrspace(1), align {m.group(4)}\n")
                out.append(f"{indent}store ptr addrspace(1) null, ptr {slot(name, j)}, align 8\n")
            changed += 1
            continue
        m = GEP_CONST.match(line)
        if m and m.group(2) in const_geps and const_geps[m.group(2)][0] in arrays:
            dead_names.add(m.group(2))  # its store is rewritten to the slot
            continue
        m = GEP_DYN.match(line)
        if m and m.group(2) in dyn_geps and dyn_geps[m.group(2)][0] in arrays:
            dead_names.add(m.group(2))  # its load is rewritten below
            continue
        m = CAST.match(line)
        if m and m.group(2) in casts and casts[m.group(2)] in arrays:
            # kept only if some other line still names it (checked below)
            out.append(line)
            continue
        m = STORE.match(line)
        if m and m.group(3) in const_geps and const_geps[m.group(3)][0] in arrays:
            arr, j = const_geps[m.group(3)]
            out.append(f"{m.group(1)}store ptr addrspace(1) {m.group(2)}, ptr {slot(arr, j)}, align {m.group(4)}\n")
            changed += 1
            continue
        if m and m.group(3) in arrays:
            out.append(f"{m.group(1)}store ptr addrspace(1) {m.group(2)}, ptr {slot(m.group(3), 0)}, align {m.group(4)}\n")
            changed += 1
            continue
        m = LOAD.match(line)
        if m and m.group(3) in dyn_geps and dyn_geps[m.group(3)][0] in arrays:
            indent = m.group(1)
            arr, idxty, idx = dyn_geps[m.group(3)]
            slots = arrays[arr]
            vals = []
            for j in range(slots):
                seq += 1
                out.append(f"{indent}%psb{seq} = load ptr addrspace(1), ptr {slot(arr, j)}, align 8\n")
                vals.append(f"%psb{seq}")
            if slots == 1:
                out.append(f"{indent}{m.group(2)} = load ptr addrspace(1), ptr {slot(arr, 0)}, align 8\n")
            else:
                sel = vals[-1]
                for j in range(slots - 2, -1, -1):
                    seq += 1
                    out.append(f"{indent}%psb{seq}c = icmp eq {idxty} {idx}, {j}\n")
                    out.append(f"{indent}%psb{seq}s = select i1 %psb{seq}c, ptr addrspace(1) {vals[j]}, ptr addrspace(1) {sel}\n")
                    sel = f"%psb{seq}s"
                out.append(f"{indent}{m.group(2)} = select i1 %psb{seq}c, ptr addrspace(1) {vals[0]}, ptr addrspace(1) {sel}\n")
            changed += 1
            continue
        out.append(line)

    # drop casts whose only users were the rewritten geps
    body = "".join(out)
    final = []
    for line in out:
        m = CAST.match(line)
        if m and m.group(2) in casts and casts[m.group(2)] in arrays:
            uses = len(re.findall(re.escape(m.group(2)) + r"(?![\w.])", body)) - 1
            if uses == 0:
                changed += 1
                continue
        final.append(line)
    return final, changed


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: psb_promote_ptr_arrays.py FILE.ll")
    path = pathlib.Path(sys.argv[1])
    lines = path.read_text().splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if ln.startswith("define ")]
    if not starts:
        print("psb_promote_ptr_arrays: no functions")
        return
    out, total = [], 0
    bounds = []
    prev = 0
    for cs in starts:
        if cs > prev:
            bounds.append((prev, cs))
        prev = cs
    bounds.append((starts[-1], len(lines)))
    for a, b in bounds:
        new, n = rewrite_function(lines[a:b])
        out.extend(new)
        total += n
    if total:
        path.write_text("".join(out))
        print(f"psb_promote_ptr_arrays: promoted {total} pointer array(s) in {path}")
    else:
        print("psb_promote_ptr_arrays: no promotable pointer arrays")


if __name__ == "__main__":
    main()
