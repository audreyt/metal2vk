#!/usr/bin/env python3
"""Convert pointer-typed memory accesses to 64-bit integer accesses (PSB mode).

clspv's producer has no codegen for pointer-typed storage or for aggregate types
with pointer members: the LoadInst/StoreInst handlers reach getSPIRVType on the
pointer result type, and a getelementptr whose source element type is a struct
with pointer members makes getSPIRVPointerType infer that struct as the pointee
data type, so getSPIRVType recurses into the pointer member - an upstream
unreachable (compiled out in Release, leaving a segfault). What the PSB address
model wants anyway is the bits: the table rows of the kimi/experts kernels hold
raw device addresses, and the producer emits OpConvertUToPtr/OpConvertPtrToU at
the SSA edges (which is why a module whose table GEP is already i64-typed
compiles and validates).

Two rewrites, in place:

1. GEPs whose source element type transitively contains a pointer become byte
   GEPs (the shape clspv handles natively); the byte offset is computed from the
   module's own type definitions:

     %r = getelementptr inbounds nuw %struct.ExpertPtrs, ptr addrspace(1) %b, i64 %i
       ->  %psbg1 = mul i64 %i, 48
           %r = getelementptr i8, ptr addrspace(1) %b, i64 %psbg1

   Only the address arithmetic changes representation; the address produced is
   the same. Types without a pointer are left alone, and a type the layout
   calculator cannot model leaves its GEPs untouched for clspv to fail loudly.

2. Every remaining `load ptr addrspace(1)` / `store ptr addrspace(1)` becomes an
   i64 access, keeping the loaded value's name on the inttoptr:

     %r = load ptr addrspace(1), ptr addrspace(N) %p   ->  %psbl1 = load i64, ... %p
                                                           %r = inttoptr i64 %psbl1 to ptr addrspace(1)
     store ptr addrspace(1) %v, ptr addrspace(N) %p    ->  %psbs1 = ptrtoint ptr addrspace(1) %v to i64
                                                           store i64 %psbs1, ... %p

Only addrspace(1) values are rewritten; anything else (a pointer of another
address space in memory) is left untouched for clspv, which then fails the
module visibly. Run last in the PSB prep, after psb_promote_ptr_arrays.py
demoted the pointer arrays.

Usage: psb_ptr_to_int.py FILE.ll (in place; a module without the shapes is a no-op)
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
TYPEDEF = re.compile(r"^%([\w.]+) = type (.+?)\s*$")
GEP = re.compile(
    r"^(\s*)(%[\w.]+) = getelementptr ((?:(?:inbounds|nuw|inrange)\s+)*)([^,]+?), "
    r"(ptr(?: addrspace\(\d+\))?) (%[\w.]+)((?:, i\d+ (?:%[\w.]+|\d+))*)\s*$"
)

PRIMS = {
    "void": (0, 1), "i1": (1, 1), "i8": (1, 1), "i16": (2, 2), "half": (2, 2),
    "bfloat": (2, 2), "i32": (4, 4), "float": (4, 4), "i64": (8, 8), "double": (8, 8),
}


def split_top(body):
    """Split a comma-separated type list at the top nesting level."""
    parts, depth, cur = [], 0, ""
    for ch in body:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == "," and depth == 0:
            if cur.strip():
                parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def deref(name, types):
    m = re.fullmatch(r"%([\w.]+)", name.strip())
    if not m or m.group(1) not in types:
        return None
    return types[m.group(1)]


def layout(ty, types):
    """(size in bytes, alignment) for a type string, or None if unmodellable."""
    ty = ty.strip()
    if re.fullmatch(r"ptr(?: addrspace\(\d+\))?", ty):
        return (8, 8)  # spir64 PSB modules only: 64-bit pointers
    if ty in PRIMS:
        return PRIMS[ty]
    if m := re.fullmatch(r"i(\d+)", ty):
        bits = int(m.group(1))
        if bits % 8:
            return None
        return (bits // 8, max(1, bits // 8))
    if body := deref(ty, types):
        return layout(body, types)
    if m := re.fullmatch(r"\[(\d+) x (.+)\]", ty):
        n, elem = int(m.group(1)), m.group(2)
        e = layout(elem, types)
        if not e:
            return None
        stride = -(-e[0] // e[1]) * e[1]  # align_up(size, align)
        return (n * stride, e[1])
    if m := re.fullmatch(r"<(\d+) x (.+)>", ty):
        n, elem = int(m.group(1)), m.group(2)
        e = layout(elem, types)
        if not e:
            return None
        return (n * e[0], e[0])
    if ty.startswith("{") and ty.endswith("}"):
        off, align = 0, 1
        for mem in split_top(ty[1:-1]):
            ml = layout(mem, types)
            if not ml:
                return None
            align = max(align, ml[1])
            off = -(-off // ml[1]) * ml[1] + ml[0]
        return (-(-off // align) * align, align)
    return None


def contains_ptr(ty, types, seen=frozenset()):
    ty = ty.strip()
    if re.fullmatch(r"ptr(?: addrspace\(\d+\))?", ty):
        return True
    if ty in seen:
        return False
    if body := deref(ty, types):
        return contains_ptr(body, types, seen | {ty})
    if m := re.fullmatch(r"[\[<](\d+) x (.+)[\]>]", ty):
        return contains_ptr(m.group(2), types, seen | {ty})
    if ty.startswith("{") and ty.endswith("}"):
        return any(contains_ptr(mem, types, seen | {ty}) for mem in split_top(ty[1:-1]))
    return False


def gep_offset(src_ty, idxs, types):
    """(literal byte offset, [(operand, index type, byte stride)]) or None.

    LLVM GEP walk: the first index steps whole objects of the source element
    type, every later index steps members (struct) or elements (array/vector).
    """
    whole = layout(src_ty, types)
    if not whole:
        return None
    off, ssa, cur = 0, [], src_ty
    for pos, (ity, iop) in enumerate(idxs):
        if pos == 0:
            stride = whole[0]
        else:
            cur = cur.strip()
            if body := deref(cur, types):
                cur = body
            if cur.startswith("{") and cur.endswith("}"):
                if not re.fullmatch(r"\d+", iop):
                    return None  # dynamic struct member index
                members = split_top(cur[1:-1])
                j = int(iop)
                acc = 0
                for mem in members[:j]:
                    ml = layout(mem, types)
                    if not ml:
                        return None
                    acc = -(-acc // ml[1]) * ml[1] + ml[0]
                off += acc
                cur = members[j]
                continue
            m = re.fullmatch(r"\[(\d+) x (.+)\]", cur)
            v = re.fullmatch(r"<(\d+) x (.+)>", cur)
            if not m and not v:
                return None
            cur = (m or v).group(2)
            el = layout(cur, types)
            if not el:
                return None
            stride = -(-el[0] // el[1]) * el[1]
        if re.fullmatch(r"\d+", iop):
            off += int(iop) * stride
        else:
            ssa.append((iop, ity, stride))
    return (off, ssa)


def flatten_geps(lines, types):
    out, changed, seq = [], 0, 0
    for line in lines:
        m = GEP.match(line)
        if m and contains_ptr(m.group(4), types):
            indent, dst, _flags, src_ty, pty, base, idxtail = m.groups()
            pairs = []
            for part in split_top(idxtail) if idxtail else []:
                ity, _, iop = part.partition(" ")
                pairs.append((ity, iop))
            if all(
                re.fullmatch(r"i\d+", a) and re.fullmatch(r"%[\w.]+|\d+", b)
                for a, b in pairs
            ):
                r = gep_offset(src_ty, pairs, types)
                if r is not None:
                    off, ssa = r
                    total = None
                    for name, ity, stride in ssa:
                        seq += 1
                        v = f"%psbg{seq}"
                        if ity != "i64":
                            out.append(f"{indent}{v} = sext {ity} {name} to i64\n")
                            seq += 1
                            v = f"%psbg{seq}"
                            out.append(f"{indent}{v} = mul i64 {v}, {stride}\n")
                        else:
                            out.append(f"{indent}{v} = mul i64 {name}, {stride}\n")
                        if total is None:
                            total = v
                        else:
                            seq += 1
                            out.append(f"{indent}%psbg{seq} = add i64 {total}, {v}\n")
                            total = f"%psbg{seq}"
                    if total is None:
                        total = str(off)
                    elif off:
                        seq += 1
                        out.append(f"{indent}%psbg{seq} = add i64 {total}, {off}\n")
                        total = f"%psbg{seq}"
                    out.append(f"{indent}{dst} = getelementptr i8, {pty} {base}, i64 {total}\n")
                    changed += 1
                    continue
        out.append(line)
    return out, changed


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: psb_ptr_to_int.py FILE.ll")
    path = pathlib.Path(sys.argv[1])
    lines = path.read_text().splitlines(keepends=True)
    types = {}
    for line in lines:
        if m := TYPEDEF.match(line):
            types[m.group(1)] = m.group(2)
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
    out, geps = flatten_geps(out, types)
    if changed or geps:
        path.write_text("".join(out))
        print(f"psb_ptr_to_int: rewrote {changed + geps} pointer accesses "
              f"({geps} pointer GEPs flattened) in {path}")
    else:
        print("psb_ptr_to_int: no pointer-typed loads or stores")


if __name__ == "__main__":
    main()
