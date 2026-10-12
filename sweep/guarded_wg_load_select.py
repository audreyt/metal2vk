#!/usr/bin/env python3
"""Rewrite LLVM IR text: a load from Workgroup memory (addrspace(3)) that only runs on one arm of
an if/else diamond and feeds the join phis is hoisted into the branch block, and the join phis
become value selects.

Why: clspv's optimiser speculates exactly this guarded load into a select over the two candidate
pointers, and the SPIR-V producer emits that pointer select with one inferred pointee type for
both operands, so a select between differently-typed workgroup pointers (a row element pointer
and the whole `[N x i32]` variable) becomes an ill-typed `OpSelect` that spirv-val rejects at
vulkan1.3 ("Expected both objects to be of Result Type: Select"; the nvfp4 argmax shape,
sweep/mlx-batch/vlmb/repro/). Selecting the loaded VALUES is the source-level escape the repro
matrix validates (v7/v10/v15/v17/v18) and keeps the pointer selects clspv sees same-typed.

Safety: the pass only fires on diamonds whose guarded block holds nothing but addrspace(3) GEPs
and loads, whose join has exactly the two diamond predecessors, and whose condition is defined in
the branch block - the speculative read is one clspv's own optimiser would have executed anyway on
the unrewritten module (it is what built the bad select). A module without the shape is a no-op.
"""
import pathlib
import re
import sys

LABEL = re.compile(r"^([\w.$-]+):(.*)$")
GEP_WG = re.compile(r"^(\s*)(%[\w.$]+) = getelementptr .*, ptr addrspace\(3\) (%[\w.$]+|@[\w.$]+),")
LOAD_WG = re.compile(r"^(\s*)(%[\w.$]+) = load [\w ]+, ptr addrspace\(3\) (%[\w.$]+|@[\w.$]+)[,.]")
COND_BR = re.compile(r"^(\s*)br i1 (%[\w.$]+), label %([\w.$-]+), label %([\w.$-]+)(\s*)$")
PLAIN_BR = re.compile(r"^(\s*)br label %([\w.$-]+)\s*$")
PHI = re.compile(r"^(\s*)(%[\w.$]+) = phi (.+?) ((?:\[[^\]]*\](?:,\s*)?)+)\s*$")
INCOMING = re.compile(r"\[\s*([^,\]]+?)\s*,\s*%([\w.$-]+)\s*\]")
DEF = re.compile(r"^(\s*)(%[\w.$]+) = ")
TERM = re.compile(r"^\s*(?:br |ret |switch |unreachable )")


def die(msg):
    sys.exit("guarded_wg_load_select: " + msg)


class Block:
    def __init__(self, label, tail, lines):
        self.label = label
        self.tail = tail  # everything after "label:" verbatim (spacing, "; preds = ...")
        self.lines = lines  # without the label line
        self.blank_tail = 0  # blank lines between this block and the next (formatting only)

    def preds_listed(self):
        m = re.search(r";\s*preds\s*=\s*(.+)$", self.tail)
        return [x.strip().lstrip("%") for x in m.group(1).split(",") if x.strip()] if m else []

    def set_preds(self, pred):
        self.tail = " " * max(1, 48 - len(self.label)) + f"; preds = %{pred}"

    def text(self):
        head = f"{self.label}:{self.tail}\n" if self.label else ""
        return head + "".join(self.lines) + "\n" * self.blank_tail


def parse_blocks(lines):
    """-> ordered blocks of one function body; text outside labels lands in the leading block."""
    blocks = []
    cur = None
    for ln in lines:
        m = LABEL.match(ln.rstrip("\n"))
        if m and not ln.startswith((" ", "\t", ";")):
            cur = Block(m.group(1), m.group(2), [])
            blocks.append(cur)
            continue
        if cur is None:
            cur = Block("", None, [])  # function prologue (parameter attributes etc.)
            blocks.append(cur)
        if not ln.strip() and (not cur.lines or TERM.match(cur.lines[-1])):
            cur.blank_tail += 1  # blank run after a terminator: block separator, keep count only
            continue
        cur.lines.append(ln)
    return blocks


def wg_only_body(block):
    """The block's non-terminator lines are only Workgroup GEPs/loads; returns (ok, hoist_lines)."""
    body = block.lines[:-1]
    if not body or not PLAIN_BR.match(block.lines[-1].rstrip("\n")):
        return False, None
    hoist = []
    for ln in body:
        if GEP_WG.match(ln) or LOAD_WG.match(ln.rstrip("\n")):
            hoist.append(ln)
        else:
            return False, None
    return True, hoist


def parse_phis(block):
    """-> (phis, other_first_idx): the leading phi lines split into parsed entries, plus the index
    of the first non-phi instruction line."""
    phis = []
    i = 0
    while i < len(block.lines):
        m = PHI.match(block.lines[i].rstrip("\n"))
        if not m:
            break
        entries = INCOMING.findall(m.group(4))
        phis.append((m, entries, block.lines[i]))
        i += 1
    return phis, i


def rewrite_function(text, counts):
    blocks = parse_blocks(text.splitlines(keepends=True))
    if len(blocks) < 2:
        return text, False
    by_label = {b.label: b for b in blocks if b.label}
    changed = False
    again = True
    while again:
        again = False
        for b in blocks:
            if not b.label:
                continue
            ok, hoist = wg_only_body(b)
            if not ok:
                continue
            j = PLAIN_BR.match(b.lines[-1].rstrip("\n")).group(2)
            join = by_label.get(j)
            if join is None or join.label == b.label:
                continue
            # unique conditional predecessor whose false (or true) arm is B and whose other arm is J
            preds = []
            for p in blocks:
                if not p.label or p.label in (b.label, j):
                    continue
                m = COND_BR.match(p.lines[-1].rstrip("\n"))
                if m and j in (m.group(3), m.group(4)) and b.label in (m.group(3), m.group(4)):
                    preds.append((p, m))
            if len(preds) != 1:
                continue
            p, m = preds[0]
            cond = m.group(2)
            b_is_true = m.group(3) == b.label
            # the branch block must define the condition, and J must sit after both in block order
            if not any(DEF.match(ln) and DEF.match(ln).group(2) == cond for ln in p.lines[:-1]):
                continue
            phis, first = parse_phis(join)
            if not phis or first >= len(join.lines):
                continue
            if not all(sorted(e[1] for e in entries) == sorted((b.label, p.label)) for _m, entries, _l in phis):
                continue
            listed = join.preds_listed()
            if listed and sorted(listed) != sorted((b.label, p.label)):
                continue
            # every value the join phis take from B must be defined by B's hoisted lines, by the
            # branch block, or not be a local (%32-style: computed in P before the branch)
            hoist_defs = {DEF.match(ln).group(2) for ln in hoist if DEF.match(ln)}
            p_defs = {DEF.match(ln).group(2) for ln in p.lines[:-1] if DEF.match(ln)}
            if not all(e[1] == p.label or e[0].strip() in hoist_defs or e[0].strip() in p_defs
                       or not e[0].strip().startswith("%")
                       for _m, entries, _l in phis for e in entries):
                continue
            # rewrite: hoist into P, retarget P's branch to J, drop B, phis become selects
            p.lines[-1] = f"  br label %{j}\n"
            p.lines[-1:-1] = hoist
            select_lines = []
            for _m, entries, _l in phis:
                vb = next(e[0].strip() for e in entries if e[1] == b.label)
                vp = next(e[0].strip() for e in entries if e[1] == p.label)
                tv, fv = (vb, vp) if b_is_true else (vp, vb)
                select_lines.append(f"  {_m.group(2)} = select i1 {cond}, {_m.group(3)} {tv}, {_m.group(3)} {fv}\n")
            join.lines = select_lines + join.lines[first:]
            join.set_preds(p.label)
            blocks.remove(b)
            by_label.pop(b.label)
            again = True
            changed = True
            counts[0] += 1
            break
    if not changed:
        return text, False
    return "".join(b.text() for b in blocks), True


def main():
    if len(sys.argv) != 2:
        die("usage: guarded_wg_load_select.py FILE.ll")
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()
    out = []
    pos = 0
    counts = [0]
    for m in re.finditer(r"^define [^\n]*\{\n", text, re.M):
        start = m.end()
        end = text.index("\n}\n", start) + 3
        body, changed = rewrite_function(text[start:end], counts)
        out.append(text[pos:start])
        out.append(body)
        pos = end
    out.append(text[pos:])
    new = "".join(out)
    if counts[0]:
        path.write_text(new)
    print(f"guarded_wg_load_select: rewrote {counts[0]} guarded workgroup load(s) in {path}")


main()
