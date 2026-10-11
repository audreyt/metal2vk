#!/usr/bin/env python3
"""Expand llvm.memcpy calls whose two pointer operands live in different address spaces.

clspv's ReplaceLLVMIntrinsicsPass lowers a memcpy to element load/stores only when it can
infer one element type for both sides; a copy between distinct address spaces (private
alloca <- global buffer, the shape SROA leaves behind after inlining a device-side struct
assignment) survives that pass and hits clspv's "Unexpected instruction" abort. The copies
this sweep produces are small fixed-size struct/record copies, so a byte-wise expansion is
exact and cheap: every source byte is loaded first, then stored (loads up front keep the
expansion correct even if the operands overlap; the signature says noalias, but the cost is
trivial at these sizes). Same-address-space memcpys are left for clspv, which lowers them
to OpCopyMemory itself.

--ptr-copies (physical-storage-buffer mode): a copy whose size is a whole number of
pointers carries pointer values - raw device addresses, e.g. a Metal `device const
ExpertPtrs*` table row. Those are expanded as pointer loads/stores instead of bytes: the
loaded value is a `ptr addrspace(1)` (the address the bytes encode), which is exactly what
a 64-bit physical-pointer module wants, and it is the only shape clspv's pointer-bitcast
pass can rewrite without reassembling a pointer from bytes. Requires a 64-bit module
(pointer size read from the datalayout); a mixed-size copy still falls back to bytes.

Runs as a text pass between the clang/opt front end and clspv; a module without a
mixed-address-space memcpy is a no-op.

Usage: memcpy_mixed_as.py [--ptr-copies] FILE.ll
"""
import pathlib
import re
import sys

CALL = re.compile(
    r"^(\s*)(?:tail )?call void @llvm\.memcpy\.[\w.]+\("
    r"(?P<args>.*)\)"
    r"(?P<md>, ![\w. !<>{}]+)?$"
)
PTR = re.compile(r"^ptr(?: addrspace\((\d+)\))?(?: align (\d+))?(?: (?P<val>%[\w.]+|null))?$")


def die(msg):
    sys.exit("memcpy_mixed_as: " + msg)


def pointer_size(txt):
    # spir64 modules carry 8-byte addresses; the 32-bit default keeps the byte expansion
    if re.search(r'^target triple = "spir64"', txt, re.M):
        return 8
    return 4


def split_args(text):
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch in "(<":
            depth += 1
        elif ch in ")>":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def parse_ptr(arg):
    m = PTR.match(arg)
    if not m:
        return None
    return (int(m.group(1) or 0), int(m.group(2) or 0), m.group("val"))


def main():
    args = [a for a in sys.argv[1:]]
    ptr_mode = "--ptr-copies" in args
    args = [a for a in args if a != "--ptr-copies"]
    if len(args) != 1:
        die("usage: memcpy_mixed_as.py [--ptr-copies] FILE.ll")
    path = pathlib.Path(args[0])
    txt = path.read_text()
    psize = pointer_size(txt) if ptr_mode else 0
    lines = txt.splitlines(keepends=True)

    out = []
    changed = False
    seq = 0
    for line in lines:
        m = CALL.match(line)
        if not m:
            out.append(line)
            continue
        cargs = split_args(m.group("args"))
        if len(cargs) != 4:
            out.append(line)
            continue
        dst, src = parse_ptr(cargs[0]), parse_ptr(cargs[1])
        size_m = re.fullmatch(r"i32 (\d+)", cargs[2]) or re.fullmatch(r"i64 (\d+)", cargs[2])
        vol = cargs[3].strip() == "true"
        if not dst or not src or not size_m or dst[2] is None or src[2] is None:
            out.append(line)
            continue
        if dst[0] == src[0]:
            out.append(line)  # same address space: clspv lowers this itself
            continue
        size = int(size_m.group(1))
        indent = m.group(1)
        changed = True
        if size == 0:
            continue  # drop the no-op copy
        stail = ", volatile\n" if vol else "\n"
        if psize and size % psize == 0:
            # pointer copies: one typed load/store pair per pointer-sized chunk
            for k in range(size // psize):
                seq += 1
                if k == 0:
                    sptr = src[2]
                    dptr = dst[2]
                else:
                    out.append(f"{indent}%mcp{seq}g = getelementptr i8, ptr addrspace({src[0]}) {src[2]}, i64 {k * psize}\n")
                    sptr = f"%mcp{seq}g"
                    seq += 1
                    out.append(f"{indent}%mcp{seq}g = getelementptr i8, ptr addrspace({dst[0]}) {dst[2]}, i64 {k * psize}\n")
                    dptr = f"%mcp{seq}g"
                seq += 1
                out.append(f"{indent}%mcp{seq} = load ptr addrspace(1), ptr addrspace({src[0]}) {sptr}, align {psize}{stail}")
                out.append(f"{indent}store ptr addrspace(1) %mcp{seq}, ptr addrspace({dst[0]}) {dptr}, align {psize}{', volatile' if vol else ''}\n")
            continue
        # loads first: every source byte, in offset order
        names = []
        for i in range(size):
            if i == 0:
                sptr = src[2]
            else:
                seq += 1
                out.append(f"{indent}%mcp{seq}g = getelementptr i8, ptr addrspace({src[0]}) {src[2]}, i64 {i}\n")
                sptr = f"%mcp{seq}g"
            seq += 1
            out.append(f"{indent}%mcp{seq} = load i8, ptr addrspace({src[0]}) {sptr}, align 1{stail}")
            names.append(f"%mcp{seq}")
        # then the stores, in offset order
        for i in range(size):
            if i == 0:
                dptr = dst[2]
            else:
                seq += 1
                out.append(f"{indent}%mcp{seq}g = getelementptr i8, ptr addrspace({dst[0]}) {dst[2]}, i64 {i}\n")
                dptr = f"%mcp{seq}g"
            out.append(f"{indent}store i8 {names[i]}, ptr addrspace({dst[0]}) {dptr}, align 1{', volatile' if vol else ''}\n")

    if changed:
        path.write_text("".join(out))
        print(f"memcpy_mixed_as: expanded mixed-address-space memcpy calls in {path}")
    else:
        print("memcpy_mixed_as: no mixed-address-space memcpy calls")


if __name__ == "__main__":
    main()
