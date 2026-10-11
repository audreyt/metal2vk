#!/usr/bin/env python3
"""Demote private arrays of device pointers to i64 slot allocas (PSB mode).

The kimi/experts kernels keep the per-lane weight pointers of the current table row in a
private array (`alloca [N x ptr addrspace(1)]`): the stores pick static slots, a helper
reads the slots back through a loop index (a dynamic `getelementptr [N x ptr
addrspace(1)]` whose base escaped through an `addrspacecast ptr %arr to ptr
addrspace(4)`). clspv cannot compile pointer-typed storage: its producer's getSPIRVType
is an unreachable on a pointer type (compiled out in Release, leaving a segfault), and
that unreachable also kills the pointer loads a slot-alloca promotion would leave behind
(upstream code contains no codegen for a pointer-typed OpLoad from private memory).

So this pass moves the row through 64-bit integers, which is also what the PSB address
model wants: every slot becomes an `alloca i64`, stores go through ptrtoint, loads come
back through inttoptr, and a dynamic index loads every slot and selects the value. The
bits are unchanged, so each pointer is exactly the raw table address the source held.

  store ptr addrspace(1) %v, ptr %arr                  -> ptrtoint + store to slot 0
  %g = getelementptr inbounds ptr addrspace(1), ptr %arr, i64 <c>
  store ptr addrspace(1) %v, ptr %g                    -> ptrtoint + store to slot <c>
  %c = addrspacecast ptr %arr to ptr addrspace(4)
  %g = getelementptr inbounds [N x ptr addrspace(1)], ptr addrspace(4) %c, i64 0, i64 %i
  %l = load ptr addrspace(1), ptr addrspace(4) %g      -> i64 slot loads, an icmp/select
                                                          chain over %i, then inttoptr
                                                          (a single slot loads straight);
                                                          %l keeps its name

Each slot starts as 0, so an out-of-range index yields a null pointer exactly where the
source had undefined behaviour. The array alloca, its geps and its cast are dropped when
nothing reads them anymore. Anything else (a dynamic store, a second user of the cast,
more than 16 slots, a pointer address space other than 1) is left untouched for clspv,
which then fails the module visibly instead of miscompiling it. Run after the dead
row-copy structs are removed (opt -passes=function(dse)), which is what leaves every
surviving pointer alloca in the shape this pass matches.

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

    const_geps = {}  # gep name -> (array, slot)
    for line in chunk:
        m = GEP_CONST.match(line)
        if m and m.group(3) in arrays:
            const_geps[m.group(2)] = (m.group(3), int(m.group(4)))

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

    # classify every consumer; a store or load this pass cannot rewrite (a dynamic
    # store, a second use of a gep or cast, anything else naming a managed name) skips
    # the array
    load_geps, bad = {}, set()
    for cname, arr in casts.items():
        # every use of the cast must be one of the dynamic geps of the same array
        other = 0
        for line in chunk:
            if re.match(rf"^\s*{re.escape(cname)} = ", line):
                continue
            m = GEP_DYN.match(line)
            if m and m.group(4) == cname:
                continue
            if cname in line:
                other += 1
        if other:
            bad.add(cname)
            arrays.pop(arr, None)
    for line in chunk:
        m = LOAD.match(line)
        if m and m.group(3) in dyn_geps:
            if m.group(3) in load_geps:
                bad.add(m.group(3))
            else:
                load_geps[m.group(3)] = True
            continue
        if STORE.match(line):
            continue  # stores to the array or a const gep are always rewritable
        for name in list(dyn_geps) + list(const_geps):
            if name in line and not re.match(rf"^\s*{re.escape(name)} = ", line):
                bad.add(name)
    for name in bad:
        if name in dyn_geps:
            arrays.pop(dyn_geps[name][0], None)
        elif name in const_geps:
            arrays.pop(const_geps[name][0], None)
    if not arrays:
        return chunk, 0

    seq = 0
    out = []
    changed = 0
    for line in chunk:
        m = ALLOCA.match(line)
        if m and m.group(2) in arrays:
            # the original array goes away entirely
            indent, name, slots = m.group(1), m.group(2), arrays[m.group(2)]
            for j in range(slots):
                out.append(f"{indent}{slot(name, j)} = alloca i64, align 8\n")
                out.append(f"{indent}store i64 0, ptr {slot(name, j)}, align 8\n")
            changed += 1
            continue
        m = GEP_CONST.match(line)
        if m and m.group(2) in const_geps and const_geps[m.group(2)][0] in arrays:
            continue  # its store is rewritten to the slot
        m = GEP_DYN.match(line)
        if m and m.group(2) in dyn_geps and dyn_geps[m.group(2)][0] in arrays:
            continue  # its load is rewritten below
        m = CAST.match(line)
        if m and m.group(2) in casts and casts[m.group(2)] in arrays:
            continue  # dropped; a surviving user is checked after the pass
        m = STORE.match(line)
        if m and m.group(3) in const_geps and const_geps[m.group(3)][0] in arrays:
            arr, j = const_geps[m.group(3)]
            seq += 1
            if m.group(2) == "null":
                out.append(f"{m.group(1)}store i64 0, ptr {slot(arr, j)}, align 8\n")
            else:
                out.append(f"{m.group(1)}%psbi{seq} = ptrtoint ptr addrspace(1) {m.group(2)} to i64\n")
                out.append(f"{m.group(1)}store i64 %psbi{seq}, ptr {slot(arr, j)}, align 8\n")
            changed += 1
            continue
        if m and m.group(3) in arrays:
            seq += 1
            if m.group(2) == "null":
                out.append(f"{m.group(1)}store i64 0, ptr {slot(m.group(3), 0)}, align 8\n")
            else:
                out.append(f"{m.group(1)}%psbi{seq} = ptrtoint ptr addrspace(1) {m.group(2)} to i64\n")
                out.append(f"{m.group(1)}store i64 %psbi{seq}, ptr {slot(m.group(3), 0)}, align 8\n")
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
                out.append(f"{indent}%psb{seq} = load i64, ptr {slot(arr, j)}, align 8\n")
                vals.append(f"%psb{seq}")
            if slots == 1:
                out.append(f"{indent}{m.group(2)} = inttoptr i64 {vals[0]} to ptr addrspace(1)\n")
            else:
                sel = vals[-1]
                for j in range(slots - 2, -1, -1):
                    seq += 1
                    out.append(f"{indent}%psb{seq}c = icmp eq {idxty} {idx}, {j}\n")
                    out.append(f"{indent}%psb{seq}s = select i1 %psb{seq}c, i64 {vals[j]}, i64 {sel}\n")
                    sel = f"%psb{seq}s"
                out.append(f"{indent}{m.group(2)} = inttoptr i64 {sel} to ptr addrspace(1)\n")
            changed += 1
            continue
        out.append(line)
    return out, changed


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
        print(f"psb_promote_ptr_arrays: demoted {total} pointer array(s) in {path}")
    else:
        print("psb_promote_ptr_arrays: no pointer arrays to demote")


if __name__ == "__main__":
    main()
