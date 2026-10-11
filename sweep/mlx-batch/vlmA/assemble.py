#!/usr/bin/env python3
"""Static assembler + checker for family vlmA (mlx-vlm first half).

Encodes MLX's own metal_kernel MSL assembly, transcribed from
ml-explore/mlx@1f8e74e mlx/backend/common/metal_kernel.cpp (write_signature,
write_template, make_template_hash, the metal_attributes list and the
kernel-name construction), and extracts every call site's source text from the
pinned mlx-vlm tree with ast so no kernel body is hand-transcribed.

Subcommands:
  gen    write <kernel>.metal files + kernels.json into this directory
  check  compile every generated file with tools/m2v-compile (which runs
         clang -> clspv -> spirv-val) and write results-vlmA.tsv
"""
import ast
import hashlib
import json
import pathlib
import re
import subprocess
import sys

TREE = pathlib.Path.home() / "scratch/sweep/mlx-batch-src/mlx-vlm"
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]  # metal2vk worktree root

# get_type_string, from mlx/backend/common/compiled.cpp.
TYPES = {
    "float32": "float", "float16": "float16_t", "bfloat16": "bfloat16_t",
    "bool": "bool", "int8": "int8_t", "int16": "int16_t", "int32": "int32_t",
    "int64": "int64_t", "uint8": "uint8_t", "uint16": "uint16_t",
    "uint32": "uint32_t", "uint64": "uint64_t",
}
# The ordered metal_attributes list from metal_kernel.cpp.
ATTRS = [
    ("dispatch_quadgroups_per_threadgroup", "uint"),
    ("dispatch_simdgroups_per_threadgroup", "uint"),
    ("dispatch_threads_per_threadgroup", "uint3"),
    ("grid_origin", "uint3"),
    ("grid_size", "uint3"),
    ("quadgroup_index_in_threadgroup", "uint"),
    ("quadgroups_per_threadgroup", "uint"),
    ("simdgroup_index_in_threadgroup", "uint"),
    ("simdgroups_per_threadgroup", "uint"),
    ("thread_execution_width", "uint"),
    ("thread_index_in_quadgroup", "uint"),
    ("thread_index_in_simdgroup", "uint"),
    ("thread_index_in_threadgroup", "uint"),
    ("thread_position_in_grid", "uint3"),
    ("thread_position_in_threadgroup", "uint3"),
    ("threadgroup_position_in_grid", "uint3"),
    ("threadgroups_per_grid", "uint3"),
    ("threads_per_grid", "uint3"),
    ("threads_per_simdgroup", "uint"),
    ("threads_per_threadgroup", "uint3"),
]
MAX_CONSTANT_ARRAY_SIZE = 8  # elements, metal_kernel.cpp:18


def prod(xs):
    r = 1
    for x in xs:
        r *= x
    return r


def template_hash(template_def):
    out = []
    i = 0
    while i < len(template_def):
        c = template_def[i]
        if c in "<>":
            out.append("_")
        elif c == "," and i + 1 < len(template_def) and template_def[i + 1] == " ":
            out.append("_")
            i += 1
        else:
            out.append(c)
        i += 1
    return "".join(out)[:-1]


def write_template(args):
    parts = []
    for _name, kind, val in args:
        parts.append(TYPES[val] if kind == "dtype" else str(val).lower())
    return "<" + ", ".join(parts) + ">"


def kernel_and_source(name, header, source, inputs, out_names, out_dtypes,
                      template, atomic=False):
    """inputs: list of (name, dtype, shape); template: list of (name, kind, val)."""
    kn = "custom_kernel_" + name
    td = ""
    if template:
        td = write_template(template)
        kn += "_" + template_hash(td)
    for _n, dt, shape in inputs:
        suffix = "s" if not shape else (
            "c" if prod(shape) < MAX_CONSTANT_ARRAY_SIZE else "")
        kn += "_" + TYPES[dt] + suffix
    for dt in out_dtypes:
        kn += "_" + TYPES[dt]

    s = header
    if template:
        params = ", ".join(
            ("typename" if k == "dtype" else k) + " " + n
            for n, k, _v in template)
        s += "template <" + params + ">\n"
    s += "[[kernel]] void " + kn + "(\n"
    idx = 0
    for n, dt, shape in inputs:
        size = prod(shape) if shape else 1
        loc = "constant" if size < MAX_CONSTANT_ARRAY_SIZE else "device"
        ref = "&" if not shape else "*"
        s += "  const " + loc + " " + TYPES[dt] + ref + " " + n
        s += " [[buffer(" + str(idx) + ")]],\n"
        idx += 1
        if shape:
            if source.find(n + "_shape") != -1:
                s += "  const constant int* " + n + "_shape [[buffer(" + str(idx) + ")]],\n"
                idx += 1
            if source.find(n + "_strides") != -1:
                s += "  const constant int64_t* " + n + "_strides [[buffer(" + str(idx) + ")]],\n"
                idx += 1
            if source.find(n + "_ndim") != -1:
                s += "  const constant int& " + n + "_ndim [[buffer(" + str(idx) + ")]],\n"
                idx += 1
    attrs = [a for a in ATTRS if source.find(a[0]) != -1]
    for j, (n, dt) in enumerate(zip(out_names, out_dtypes)):
        s += "  device " + ("atomic<" + TYPES[dt] + ">" if atomic else TYPES[dt])
        s += "* " + n + " [[buffer(" + str(idx) + ")]]"
        s += ",\n" if (idx < len(inputs) + len(out_names) - 1 or attrs) else ") {\n"
        idx += 1
    for k, (attr, atype) in enumerate(attrs):
        s += "  " + atype + " " + attr + " [[" + attr + "]]"
        s += ",\n" if k < len(attrs) - 1 else ") {\n"
    s += source + "\n}\n"
    if template:
        s += "\ntemplate [[host_name(\"" + kn + "\")]] [[kernel]] decltype("
        s += kn + td + ") " + kn + td + ";\n"
    return kn, s


# ---- extraction from the pinned tree ---------------------------------------


def _str_value(v, strs):
    if isinstance(v, ast.Constant):
        return v.value
    if isinstance(v, ast.Name):
        return strs.get(v.id)
    if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Add):
        a, b = _str_value(v.left, strs), _str_value(v.right, strs)
        return a + b if a is not None and b is not None else None
    return None


def module_strings(tree):
    strs = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            v = _str_value(node.value, strs)
            if v is not None:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        strs.setdefault(t.id, v)
    return strs


def all_strings(tree):
    """Map of every string constant assignment anywhere in the tree."""
    strs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    strs[t.id] = node.value.value
    return strs


def resolve_source(kw, tree, strs, key="source"):
    """A string kwarg: inline constant, module constant, or a local name."""
    v = kw[key]
    if isinstance(v, ast.Constant):
        return v.value
    if isinstance(v, ast.Name):
        return strs.get(v.id) or all_strings(tree)[v.id]
    return strs[v.id]


def find_kernel_calls(tree):
    """Every .metal_kernel(...) call as (lineno, kwargs, scoped strings).

    Scoped strings are the string constants visible at the call site: the
    enclosing top-level function's locals (kernels.py reuses the names
    `source` and `header` in several functions) plus the module constants.
    """
    mod = module_strings(tree)
    scopes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            local = {}
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Constant) \
                        and isinstance(sub.value.value, str):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            local[t.id] = sub.value.value
            scope = dict(mod)
            scope.update(local)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                        and sub.func.attr == "metal_kernel":
                    scopes.append((sub.lineno,
                                   {k.arg: k.value for k in sub.keywords}, scope))
        else:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                        and sub.func.attr == "metal_kernel":
                    scopes.append((sub.lineno,
                                   {k.arg: k.value for k in sub.keywords}, dict(mod)))
    scopes.sort(key=lambda c: c[0])
    return scopes


def const_or_name(v, strs):
    if isinstance(v, ast.Constant):
        return v.value
    if isinstance(v, ast.Name):
        return strs[v.id]
    raise ValueError("unsupported node: " + ast.dump(v)[:80])


def name_prefix(kw):
    n = kw["name"]
    if isinstance(n, ast.Constant):
        return n.value
    if isinstance(n, ast.JoinedStr) and isinstance(n.values[0], ast.Constant):
        return n.values[0].value
    return ""


def joinedstr_template(node):
    parts = []
    for v in node.values:
        if isinstance(v, ast.Constant):
            parts.append(v.value.replace("{", "{{").replace("}", "}}"))
        elif isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Name):
            parts.append("{" + v.value.id + "}")
        else:
            raise ValueError("unsupported f-string part")
    return "".join(parts)


def fn_by_name(tree, fnname):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fnname:
            return node
    raise ValueError("function not found: " + fnname)


def header_from_fn(tree, fnname, replaces):
    for sub in ast.walk(fn_by_name(tree, fnname)):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                and len(sub.value) > 200:
            h = sub.value
            for a, b in replaces:
                h = h.replace(a, b)
            return h
    raise ValueError("header constant not found in " + fnname)


def fn_flag_strings(tree, fnname, flags):
    """Resolve flag-conditional string assignments inside a function."""
    out = {}

    def ev(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.IfExp) and isinstance(n.test, ast.Name):
            return ev(n.body) if flags[n.test.id] else ev(n.orelse)
        raise ValueError("unsupported assignment value")

    def run(body):
        for st in body:
            if isinstance(st, ast.Assign) and isinstance(st.value, (ast.Constant, ast.IfExp)):
                out[st.targets[0].id] = ev(st.value)
            elif isinstance(st, ast.If) and isinstance(st.test, ast.Name) \
                    and st.test.id in flags:
                run(st.body if flags[st.test.id] else st.orelse)

    run(fn_by_name(tree, fnname).body)
    return out


# ---- kernel specs -----------------------------------------------------------

KS = []
ORIGIN_REPO = "mlx-vlm@8f5dc3d"


def spec(kernel, path, line, header, source, inputs, out_dtypes, template,
         grid, threadgroup, shapes_note, out_names):
    rel = str(path).split("mlx-batch-src/", 1)[1]
    KS.append(dict(
        kernel=kernel, msl=kernel + ".metal", origin=ORIGIN_REPO + ":" + rel
        + ":" + str(line), header=header, source=source, inputs=inputs,
        out_dtypes=out_dtypes, template=template, grid=grid,
        threadgroup=threadgroup,
        representative=shapes_note + "; grid=" + str(grid) + ", threadgroup="
        + str(threadgroup), out_names=out_names))


def load_specs():
    F = TREE / "mlx_vlm/models/"
    f32, fp16, bf = "float32", "float16", "bfloat16"

    # bonsai/klein_fast/blocks.py : two formatted sources, bfloat16 end to end
    p = F / "bonsai/klein_fast/blocks.py"
    tree = ast.parse(p.read_text())
    strs = module_strings(tree)
    dbl = strs["_FUSED_DOUBLE_NORM_ROPE_SOURCE"].format(
        S_IMG=2, S_TXT=1, H_DIM=24, D_DIM=128, DIM=3072, EPS="1e-06", ELEMS=4)
    sgl = strs["_FUSED_SINGLE_NORM_ROPE_SOURCE"].format(
        S_DIM=4, H_DIM=24, D_DIM=128, FUSED_DIM=27648, DIM=3072, EPS="1e-06",
        ELEMS=4)
    spec("flux2_fused_double_norm_rope_2_1_3072_24_128_1en06", p, 394, "", dbl,
         [("img_qkv", bf, [1, 2, 9216]), ("txt_qkv", bf, [1, 1, 9216]),
          ("norm_q", bf, [128]), ("norm_k", bf, [128]),
          ("norm_added_q", bf, [128]), ("norm_added_k", bf, [128]),
          ("cos_vals", bf, [3, 64]), ("sin_vals", bf, [3, 64])],
         [bf, bf, bf], [("T", "dtype", "bfloat16")], (2304, 1, 1), (32, 1, 1),
         "default Flux2KleinBlockSpec (dim 3072, 24 heads, head_dim 128, eps 1e-06), img_seq 2, txt_seq 1, batch 1; bfloat16 as both call sites pass template T=bf16 and bf16 arrays",
         ["q_out", "k_out", "v_out"])
    spec("flux2_fused_single_norm_rope_4_3072_24_128_27648_1en06", p, 434, "", sgl,
         [("fused", bf, [1, 4, 27648]), ("norm_q", bf, [128]),
          ("norm_k", bf, [128]), ("cos_vals", bf, [4, 64]),
          ("sin_vals", bf, [4, 64])],
         [bf, bf, bf], [("T", "dtype", "bfloat16")], (3072, 1, 1), (32, 1, 1),
         "default spec, seq_len 4, batch 1; fused width qkv_mlp_out_dim = 3*3072 + 2*9216 = 27648; bfloat16 per call site",
         ["q_out", "k_out", "v_out"])

    # deepseek_v4/hyper_connection.py
    p = F / "deepseek_v4/hyper_connection.py"
    tree = ast.parse(p.read_text())
    line, kw, scope = find_kernel_calls(tree)[0]
    spec("hc_sinkhorn_collapse", p, line, "",
         resolve_source(kw, tree, scope),
         [("x_in", f32, [1, 2, 4, 8]), ("mixes", f32, [2, 24]),
          ("scale", f32, [3]), ("base", f32, [24])],
         [f32, f32, f32],
         [("T", "dtype", "float32"), ("U", "dtype", "float32"),
          ("HC", "int", 4), ("ITERS", "int", 2), ("D", "int", 8),
          ("EPS_INT", "int", 1)],
         (512, 1, 1), (256, 1, 1),
         "float32 x (1,2,4,8), hc_mult 4, sinkhorn_iters 2, width 8, eps 1e-09; float32 keeps x, mixes, scale, base in one dtype",
         ["collapsed", "post", "comb"])

    # exact_speculative_verify.py
    p = F / "exact_speculative_verify.py"
    tree = ast.parse(p.read_text())
    calls = find_kernel_calls(tree)
    hdr = "#include <metal_simdgroup>\nusing namespace metal;\n"
    spec("exact_speculative_verify_gemv", p, calls[0][0], hdr,
         resolve_source(calls[0][1], tree, calls[0][2]),
         [("x", f32, [2, 8]), ("weight", f32, [8, 8])], [f32],
         [("T", "dtype", "float32"), ("K", "int", 8), ("O", "int", 8),
          ("R", "int", 2)],
         (32, 2, 8), (32, 1, 8),
         "float32 x (2,8), weight (8,8); smallest shapes meeting the caller guards (outputs 8 %4==0, K 8 < 16*O); K=8 exercises the leftover path",
         ["out"])
    spec("exact_speculative_verify_switch_gemv", p, calls[1][0], hdr,
         resolve_source(calls[1][1], tree, calls[1][2]),
         [("x", f32, [2, 8]), ("weight", f32, [2, 8, 8]),
          ("expert_indices", "uint32", [2])], [f32],
         [("T", "dtype", "float32"), ("K", "int", 8), ("O", "int", 8),
          ("R", "int", 2), ("S", "int", 1),
          ("BROADCAST_EXPERTS", "bool", False)],
         (32, 2, 8), (32, 1, 8),
         "float32, 2 experts of (8,8); per-token uint32 indices (2,), BROADCAST_EXPERTS=false as when token shapes differ",
         ["out"])

    # fast_ops.py : five kernels
    p = F / "fast_ops.py"
    tree = ast.parse(p.read_text())
    strs = module_strings(tree)
    common = strs["_COMMON_HEADER"]

    def affine_header(group_size, results, simds):
        h = strs["_AFFINE5_SWITCH_HEADER"]
        return (h.replace("__GROUP_SIZE__", str(group_size))
                .replace("__RESULTS_PER_SIMDGROUP__", str(results))
                .replace("__NUM_SIMDGROUPS__", str(simds))
                .replace("__PACKS_PER_THREAD__", "2"))

    for line, kw, _scope in find_kernel_calls(tree):
        nm = name_prefix(kw)
        if nm == "short_block_hc_norm_":
            spec("short_block_hc_norm_fp16_h4_d8_i2_he1_ne1000", p, line, "",
                 strs["_HC_NORM_SOURCE"],
                 [("x_in", fp16, [1, 2, 4, 8]), ("mixes", f32, [2, 24]),
                  ("scale", f32, [3]), ("base", f32, [24]),
                  ("norm_weight", fp16, [8])],
                 [fp16, f32, f32],
                 [("T", "dtype", "float16"), ("HC", "int", 4), ("D", "int", 8),
                  ("ITERS", "int", 2), ("HC_EPS_INT", "int", 1),
                  ("NORM_EPS_INT", "int", 1000)],
                 (64, 1, 1), (32, 1, 1),
                 "fp16 x (1,2,4,8), hc_mult 4, width 8, 2 sinkhorn iters, hc_eps 1e-09, norm_eps 1e-06; fp16 because the caller gates dtype to bfloat16/float16 (float16 chosen over bfloat16)",
                 ["normalized", "post", "comb"])
        elif nm == "short_block_hc_normalized_norm_":
            spec("short_block_hc_normalized_norm_fp16_fp16_h4_d4096_i2_he1_ne1000",
                 p, line, "", strs["_HC_NORMALIZED_NORM_SOURCE"],
                 [("x_in", fp16, [1, 1, 4, 4096]),
                  ("mix_weight", fp16, [24, 16384]), ("scale", f32, [3]),
                  ("base", f32, [24]), ("norm_weight", fp16, [16384])],
                 [fp16, f32, f32],
                 [("T", "dtype", "float16"), ("W", "dtype", "float16"),
                  ("HC", "int", 4), ("D", "int", 4096), ("ITERS", "int", 2),
                  ("HC_EPS_INT", "int", 1), ("NORM_EPS_INT", "int", 1000)],
                 (1024, 1, 1), (1024, 1, 1),
                 "the caller hard-requires x.shape[2:] == (4, 4096) and mix weight (24, 16384); fp16 per the same dtype gate",
                 ["normalized", "post", "comb"])
        elif nm == "short_block_hc_expand_":
            spec("short_block_hc_expand_fp16_r2_h4_d8", p, line, common,
                 strs["_HC_EXPAND_SOURCE"],
                 [("x", fp16, [1, 2, 8]), ("residual", fp16, [1, 2, 4, 8]),
                  ("post", f32, [1, 2, 4]), ("comb", f32, [1, 2, 4, 4])],
                 [fp16],
                 [("T", "dtype", "float16"), ("ROWS", "int", 2), ("HC", "int", 4),
                  ("D", "int", 8), ("SIMDS", "int", 8)],
                 (32, 8, 1), (32, 8, 1),
                 "fp16 x (1,2,8), residual (1,2,4,8), rows 2, simds 8 as in the caller; fp16 per the caller dtype gate",
                 ["out"])
        elif nm == "short_block_affine":
            joined = "".join(
                v.value if isinstance(v, ast.Constant) else "{}"
                for v in kw["name"].values)
            if "switch_gate_up" in joined:
                spec("short_block_affine5_switch_gate_up_fp16_t2_k512_n8_e1_g16",
                     p, line, affine_header(16, 4, 2),
                     strs["_AFFINE5_SWITCH_GATE_UP_SOURCE"],
                     [("x", fp16, [1, 2, 512]), ("indices", "int32", [2]),
                      ("up_w", "uint8", [2, 8, 320]), ("up_scales", fp16, [2, 256]),
                      ("up_biases", fp16, [2, 256]), ("gate_w", "uint8", [2, 8, 320]),
                      ("gate_scales", fp16, [2, 256]),
                      ("gate_biases", fp16, [2, 256])],
                     [fp16, fp16],
                     [("T", "dtype", "float16"), ("VERIFY_T", "int", 2),
                      ("K_SIZE", "int", 512), ("N_SIZE", "int", 8),
                      ("TOP_K", "int", 1), ("GROUP_SIZE", "int", 16)],
                     (32, 2, 2), (32, 2, 1),
                     "affine 5-bit, fp16 x (1,2,512), n_size 8, top_k 1, group_size 16 (smallest sizes passing k%512==0, n%8==0, g%16==0), 2 experts of 8 rows; scales/biases (2, 8*32)",
                     ["up_y", "gate_y"])
            elif "moe_down" in joined:
                spec("short_block_affine5_moe_down_fp16_t2_k512_n8_e1_g16",
                     p, line, affine_header(16, 2, 4),
                     strs["_AFFINE5_MOE_DOWN_SOURCE"],
                     [("x", fp16, [1, 2, 1, 512]), ("indices", "int32", [2]),
                      ("route_weights", fp16, [1, 2, 1]),
                      ("w", "uint8", [2, 8, 320]), ("scales", fp16, [2, 256]),
                      ("biases", fp16, [2, 256]), ("shared_y", fp16, [1, 2, 8])],
                     [fp16],
                     [("T", "dtype", "float16"), ("VERIFY_T", "int", 2),
                      ("K_SIZE", "int", 512), ("N_SIZE", "int", 8),
                      ("TOP_K", "int", 1), ("GROUP_SIZE", "int", 16)],
                     (32, 4, 2), (32, 4, 1),
                     "affine 5-bit, fp16, same guard sizes as switch_gate_up, routed x (1,2,1,512), shared_y (1,2,8)",
                     ["y"])

    # gated_delta.py : one call site, eight flag variants
    p = F / "gated_delta.py"
    tree = ast.parse(p.read_text())
    line, kw, _scope = find_kernel_calls(tree)[0]
    src_node = kw["source"]
    if isinstance(src_node, ast.Name):
        src_node = next(st.value for st in fn_by_name(tree, "_make_gated_delta_kernel").body
                        if isinstance(st, ast.Assign)
                        and st.targets[0].id == src_node.id)
    tmpl = joinedstr_template(src_node)
    for has_mask in (False, True):
        for vectorized in (False, True):
            for states in (False, True):
                vals = fn_flag_strings(
                    tree, "_make_gated_delta_kernel",
                    {"has_mask": has_mask, "vectorized": vectorized,
                     "capture_states": states})
                src = tmpl.format(**vals)
                suffix = ("_vec" if vectorized else "") \
                    + ("_mask" if has_mask else "") + ("_states" if states else "")
                inputs = [("q", f32, [1, 2, 1, 32]), ("k", f32, [1, 2, 1, 32]),
                          ("v", f32, [1, 2, 1, 4]),
                          ("g", f32, [1, 2, 1, 32] if vectorized else [1, 2, 1]),
                          ("beta", f32, [1, 2, 1]), ("state_in", f32, [1, 1, 4, 32]),
                          ("T", "int32", [])]
                if has_mask:
                    inputs.append(("mask", "bool", [2]))
                outs = [f32, f32] + ([f32] if states else [])
                tmpl_args = [("InT", "dtype", "float32"),
                             ("StT", "dtype", "float32"), ("Dk", "int", 32),
                             ("Dv", "int", 4), ("Hk", "int", 1), ("Hv", "int", 1)]
                if states:
                    tmpl_args.append(("StateT", "int", 2))
                note = ("float32 q/k (1,2,1,32), v (1,2,1,4), state (1,1,4,32), "
                        "T=2 steps, Dk 32, Dv 4, single head"
                        + ("; bool mask (2,)" if has_mask else "")
                        + ("; vectorized g (1,2,1,32)" if vectorized else "")
                        + ("; StateT 2" if states else ""))
                spec("gated_delta_step" + suffix, p, line, "", src, inputs, outs,
                     tmpl_args, (32, 4, 1), (32, 4, 1), note,
                     ["y", "state_out"] + (["states"] if states else []))

    # glm_moe_dsa/speculative_verifier.py
    p = F / "glm_moe_dsa/speculative_verifier.py"
    tree = ast.parse(p.read_text())
    strs = module_strings(tree)
    gh = header_from_fn(tree, "_quantized_head_header",
                        [("__BITS__", "4"), ("__GS__", "32")])
    line, kw, _scope = find_kernel_calls(tree)[0]
    spec("glm_moe_dsa_target_verify_qmv_b4_gs32_t2_k512_n8_fp16", p, line, gh,
         resolve_source(kw, tree, strs),
         [("x", fp16, [1, 2, 512]), ("w", "uint8", [8, 256]),
          ("scales", fp16, [8, 16]), ("biases", fp16, [8, 16])],
         [fp16],
         [("T", "dtype", "float16"), ("VERIFY_T", "int", 2),
          ("K_SIZE", "int", 512), ("N_SIZE", "int", 8)],
         (32, 1, 1), (32, 2, 1),
         "affine 4-bit, group 32, fp16 x (1,2,512) (x dtype must equal scales dtype, bf16/fp16 only), 8 output rows, VERIFY_T 2; sizes pass input%512==0 and output%8==0",
         ["y"])

    # inkling/language.py : three kernels beyond the two upstream-done ones
    p = F / "inkling/language.py"
    tree = ast.parse(p.read_text())
    strs = module_strings(tree)
    for line, kw, _scope in find_kernel_calls(tree):
        nm = name_prefix(kw)
        src = resolve_source(kw, tree, strs)
        if nm == "inkling_sconv_decode":
            spec(nm, p, line, "", src,
                 [("x", f32, [1, 4, 8]), ("state", f32, [1, 3, 8]),
                  ("w", f32, [32]), ("res", f32, [1, 4, 8])],
                 [f32, f32],
                 [("T", "dtype", "float32"), ("B", "int", 1), ("L", "int", 4),
                  ("C", "int", 8), ("K", "int", 4),
                  ("HAS_RES", "bool", False)],
                 (32, 1, 1), (32, 1, 1),
                 "float32 x (1,4,8), K=4 causal conv over the (1,3,8) state, HAS_RES=false (residual passed as x); float32 keeps the documented fp32-stability path",
                 ["out", "nstate"])
        elif nm == "inkling_moe_route":
            spec(nm, p, line, "", src,
                 [("logits", f32, [2, 10]), ("corr", f32, [8]),
                  ("wscale", f32, [1])],
                 ["uint32", f32, f32],
                 [("T", "dtype", "float32"), ("N", "int", 2), ("R", "int", 8),
                  ("SH", "int", 2), ("K", "int", 2), ("I", "int", 4)],
                 (32, 2, 1), (32, 1, 1),
                 "float32 logits (2, 8 routed + 2 shared), top_k 2, intermediate 4; one simdgroup per token",
                 ["idx", "wk", "gamma"])
        elif nm == "inkling_moe_down_combine":
            spec(nm, p, line, "", src,
                 [("xin", f32, [1, 2, 2048]), ("wq", "uint32", [2, 8, 256]),
                  ("sc", f32, [2, 8, 32]), ("bi", f32, [2, 8, 32]),
                  ("idx", "uint32", [1, 2]), ("wk", f32, [1, 2])],
                 [f32],
                 [("T", "dtype", "float32"), ("OUT", "int", 8),
                  ("IN", "int", 2048), ("GROUPS", "int", 32), ("K", "int", 2)],
                 (256, 8, 1), (256, 1, 1),
                 "q4/g64 affine down projection; the kernel hard-requires IN=2048 (one 64-wide group per lane over 32 lanes); OUT 8, 2 experts; uint32 packed weights as mlx quantized affine passes them",
                 ["out"])

    # kernels.py : bicubic (the other three call sites are upstream-done)
    p = F / "kernels.py"
    tree = ast.parse(p.read_text())
    for line, kw, _scope in find_kernel_calls(tree):
        if name_prefix(kw) == "bicubic_interpolation_antialias":
            spec("bicubic_interpolation_antialias", p, line,
                 resolve_source(kw, tree, _scope, "header"),
                 resolve_source(kw, tree, _scope),
                 [("input", f32, [4]), ("dims", "int32", [6]),
                  ("params", f32, [7])],
                 [f32], [], (4, 4, 1), (4, 4, 1),
                 "float32, 2x2 input upsampled to 4x4, batch*channels 1; dims/params arrays exactly as the caller builds them; input stays float32 so no dtype template is needed",
                 ["output"])

    # kimi_k3/language.py
    p = F / "kimi_k3/language.py"
    tree = ast.parse(p.read_text())
    strs = module_strings(tree)
    for line, kw, _scope in find_kernel_calls(tree):
        nm = name_prefix(kw)
        src = resolve_source(kw, tree, strs)
        if nm == "k3_short_conv_step":
            spec(nm, p, line, "", src,
                 [("x", f32, [1, 1, 8]), ("state", f32, [1, 3, 8]),
                  ("w", f32, [32])],
                 [f32, f32],
                 [("T", "dtype", "float32"), ("C", "int", 8), ("KS", "int", 4)],
                 (8, 1, 1), (8, 1, 1),
                 "float32 x (1,1,8), kernel_size 4 short conv step over the (1,3,8) state",
                 ["y", "new_state"])
        elif nm == "attnres_mix":
            spec(nm, p, line, "", src,
                 [("raw", f32, [2, 1, 8]), ("inv_rms", f32, [2, 1]),
                  ("partial", f32, [1, 8]), ("w_eff", f32, [8]),
                  ("eps", f32, [1]), ("N", "int32", [])],
                 [f32],
                 [("InT", "dtype", "float32"), ("K", "int", 2), ("D", "int", 8),
                  ("THREADS", "int", 512)],
                 (512, 1, 1), (512, 1, 1),
                 "float32; K=2 prior residual rows plus the partial row, D 8, THREADS 512 as in the caller; eps and N arrive as small buffers (N a scalar)",
                 ["out"])

    # longcat_flash_sparse/indexer_kernel.py
    p = F / "longcat_flash_sparse/indexer_kernel.py"
    tree = ast.parse(p.read_text())
    line, kw, _scope = find_kernel_calls(tree)[0]
    spec("indexer_epilogue_h8", p, line, "",
         resolve_source(kw, tree, scope),
         [("scores_in", f32, [1, 8, 2, 4]), ("weights", f32, [2, 8]),
          ("scale", f32, [1]), ("meta", "int32", [2])],
         [f32],
         [("NHEADS", "int", 8), ("TG", "int", 256)],
         (256, 2, 1), (256, 1, 1),
         "float32 scores (1, 8 heads, L 2, Np 4), TG 256 as fixed by the module constant",
         ["scores"])

    # minimax_m3_vl/language.py
    p = F / "minimax_m3_vl/language.py"
    tree = ast.parse(p.read_text())
    line, kw, _scope = find_kernel_calls(tree)[0]
    spec("minimax_m3_sparse_prefill_1p_fp16_d32_s8_b4_k2_qh2_kh1", p, line,
         resolve_source(kw, tree, {}, "header"), resolve_source(kw, tree, {}),
         [("queries", fp16, [1, 2, 1, 32]), ("keys", fp16, [1, 1, 56, 32]),
          ("values", fp16, [1, 1, 56, 32]), ("block_indices", "int32", [1, 1, 2]),
          ("q_positions", "int32", [1, 1]), ("scale", f32, [1]),
          ("k_size", "int32", [1])],
         [fp16],
         [("T", "dtype", "float16"), ("D_SIZE", "int", 32), ("Q_LEN", "int", 1),
          ("NUM_Q_HEADS", "int", 2), ("NUM_KV_HEADS", "int", 1),
          ("GQA_FACTOR", "int", 2), ("BLOCK_SIZE", "int", 4),
          ("TOPK_BLOCKS", "int", 2), ("SELECTED_LENGTH", "int", 8)],
         (1024, 2, 1), (1024, 1, 1),
         "fp16 (the caller gates dtype to bfloat16/float16), D_SIZE 32, 2 top-k blocks of 4, selected 8, key_length 56 = 7*selected per the caller guard",
         ["out"])

    # nemotron_h/speculative_verifier.py
    p = F / "nemotron_h/speculative_verifier.py"
    tree = ast.parse(p.read_text())
    for line, kw, _scope in find_kernel_calls(tree):
        nm = name_prefix(kw)
        tmpl = [("T", "dtype", "float32"), ("U", "dtype", "float32"),
                ("VERIFY_T", "int", 2), ("H", "int", 2), ("DH", "int", 4),
                ("DS", "int", 32), ("GROUPS", "int", 1),
                ("HEADS_PER_GROUP", "int", 2)]
        if nm.startswith("nemotron_h_target_verify_mamba"):
            spec("nemotron_h_target_verify_mamba_t2_h2_dh4_ds32_g1_fp32_fp32",
                 p, line, "", resolve_source(kw, tree, {}),
                 [("X", f32, [1, 2, 2, 4]), ("A_log", f32, [2]),
                  ("B", f32, [1, 2, 1, 32]), ("C", f32, [1, 2, 1, 32]),
                  ("D", f32, [2]), ("dt", f32, [1, 2, 2]),
                  ("state_in", f32, [1, 2, 4, 32])],
                 [f32, f32], tmpl, (32, 4, 2), (32, 4, 1),
                 "float32 X (1,2,2,4), state (1,2,4,32), DS 32 (multiple of 32 for one state per lane), 2 verify steps, 1 group of 2 heads",
                 ["out", "state_out"])
        elif nm.startswith("nemotron_h_target_replay_mamba"):
            spec("nemotron_h_target_replay_mamba_t2_h2_dh4_ds32_g1_fp32_fp32",
                 p, line, "", resolve_source(kw, tree, {}),
                 [("X", f32, [1, 2, 2, 4]), ("A_log", f32, [2]),
                  ("B", f32, [1, 2, 1, 32]), ("dt", f32, [1, 2, 2]),
                  ("state_in", f32, [1, 2, 4, 32])],
                 [f32], tmpl, (32, 4, 2), (32, 4, 1),
                 "float32, same shapes as the verify kernel without the C/D output path",
                 ["state_out"])


def gen():
    load_specs()
    manifest = []
    for k in KS:
        kn, src = kernel_and_source(
            k["kernel"], k["header"], k["source"], k["inputs"], k["out_names"],
            k["out_dtypes"], k["template"])
        text = ("// Origin: " + ORIGIN_REPO + ", " + k["origin"].split(":", 1)[1]
                + ".\n// Representative call: " + k["representative"] + ".\n") + src
        (HERE / k["msl"]).write_text(text)
        manifest.append(dict(
            kernel=k["kernel"], msl=k["msl"], function=kn,
            sha256=hashlib.sha256(text.encode()).hexdigest(), origin=k["origin"],
            representative=k["representative"]))
        print(k["kernel"], "->", kn)
    (HERE / "kernels.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(len(manifest), "kernels written")


def check():
    manifest = json.loads((HERE / "kernels.json").read_text())
    rows = []
    okc = 0
    for m in manifest:
        out = HERE / "out" / m["kernel"]
        base = ["ionice", "-c3", "nice", "-n", "19",
                str(REPO / "tools/m2v-compile"),
                "--msl", str(HERE / m["msl"]), "--out", str(out)]
        r = subprocess.run(base + ["--json"], capture_output=True, text=True,
                           timeout=900)
        js = {}
        for line in r.stdout.splitlines():
            if line.startswith("{"):
                js = json.loads(line)
        status = js.get("status", "error(rc=%d)" % r.returncode)
        first_error = ""
        if status == "ok":
            okc += 1
        else:
            r2 = subprocess.run(base, capture_output=True, text=True, timeout=900)
            text = (r2.stdout or "") + "\n" + (r2.stderr or "")
            for ln in text.splitlines():
                if re.search(r"error|Error|fail|refus|deprecat", ln):
                    first_error = ln.strip()[:220]
                    break
            else:
                lines = [l for l in text.splitlines() if l.strip()]
                first_error = lines[-1][:220] if lines else "no output"
            print("FAIL " + m["kernel"] + " rc=" + str(r.returncode)
                  + " :: " + first_error)
        valid = "yes" if status == "ok" else "no"
        giveup = "M2V-GIVEUP" if status == "refused" else ""
        cause = ""
        if status == "refused":
            cause = "refused: metal feature known untranslatable"
        elif status != "ok":
            low = first_error.lower()
            if r.returncode == 5:
                cause = "spirv validation"
            elif "bfloat" in low or "fp16" in low and "type" in low:
                cause = "unsupported type"
            elif "clspv" in low:
                cause = "clspv limit"
            elif "builtin" in low or "undeclared" in low or "no member" in low:
                cause = "missing builtin"
            else:
                cause = "compile error"
        rows.append([m["kernel"], m["origin"], m["sha256"], valid,
                     first_error.replace("\t", " "), giveup,
                     m["representative"].replace("\t", " "),
                     "yes" if status == "ok" else
                     ("spirv-val fail" if r.returncode == 5 else ""),
                     cause])
    hdr = ["kernel", "origin", "source_sha256", "valid", "first_error",
           "m2v_giveup", "representative_call", "spirv_val", "cause"]
    with (HERE / "results-vlmA.tsv").open("w") as f:
        f.write("\t".join(hdr) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    print("compiled ok:", okc, "failed:", len(rows) - okc)


if __name__ == "__main__":
    {"gen": gen, "check": check}[sys.argv[1]]()
