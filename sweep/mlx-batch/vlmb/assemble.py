#!/usr/bin/env python3
"""Static MSL assembler for mlx fast.metal_kernel call sites.

Executes a pinned upstream module with a recording fake ``mlx`` (no MLX
install needed), calls each kernel factory (or replays module-level
registrations), and emits the exact Metal source MLX would generate at the
representative call. The signature builder is a faithful port of
mlx/backend/common/metal_kernel.cpp write_signature: buffer order, shape /
strides / ndim buffers when the source mentions them, constant address space
and the ``c``/``s`` name suffixes for small / scalar inputs, the metal
attribute arguments, and the ``template [[host_name(...)]]`` instantiation.

Usage: assemble.py spec.json   (writes <captured name>.metal next to it)
"""
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
import types

HERE = pathlib.Path(__file__).resolve().parent

# mlx/common/compiled.cpp get_type_string
TYPE_STRINGS = {
    "float32": "float", "float16": "float16_t", "bfloat16": "bfloat16_t",
    "float64": "double", "complex64": "complex64_t", "bool": "bool",
    "int8": "int8_t", "int16": "int16_t", "int32": "int32_t", "int64": "int64_t",
    "uint8": "uint8_t", "uint16": "uint16_t", "uint32": "uint32_t", "uint64": "uint64_t",
}

# metal_kernel.cpp metal_attributes order
METAL_ATTRIBUTES = [
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

MAX_CONSTANT_ARRAY_SIZE = 8

# Pinned upstream trees live under this root (mlx-vlm/, mlx/, ... siblings).
PINNED_ROOT = pathlib.Path(
    os.environ.get("MLX_BATCH_SRC_ROOT",
                   pathlib.Path.home() / "scratch/sweep/mlx-batch-src"))


def dtype_str(name):
    if name in TYPE_STRINGS:
        return TYPE_STRINGS[name]
    raise SystemExit(f"unknown dtype {name!r}; known: {sorted(TYPE_STRINGS)}")


def template_arg_str(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str):
        return dtype_str(value)
    return str(value)


def make_template_hash(template_def):
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
    return "".join(out)[:-1] if out else ""


class Recorder:
    """Stands in for mx.fast.metal_kernel; records every registration."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)

        class _Kernel:
            def __call__(self, *a, **k):
                raise SystemExit("assembled kernel launched: capture-only harness")

            def __getitem__(self, _):
                return self

        return _Kernel()


def fake_mx():
    mx = types.ModuleType("mlx.core")

    class _Array:
        pass

    for name in TYPE_STRINGS:
        setattr(mx, name, _Dtype(name))
    mx.array = _Array
    mx.Dtype = type("Dtype", (), {})
    mx.fast = types.SimpleNamespace(metal_kernel=Recorder())
    mx.metal = types.SimpleNamespace(is_available=lambda: True)
    mx.compile = lambda fn=None, **kw: (lambda f: f) if fn is None else fn
    mx.__file__ = str(pathlib.Path("/nonexistent-mx-include/mx.py"))
    # any other mx.<attr> stays a permissive stand-in
    return mx


def _permissive(name):
    mod = types.ModuleType(name)

    def __getattr__(item):
        if item.startswith("__"):
            raise AttributeError(item)
        return type(item, (), {})

    mod.__getattr__ = __getattr__
    return mod


def capture_module(py_path, mx_file=None):
    """Exec a pinned module with mlx stubbed out; returns (namespace, recorder)."""
    mx = fake_mx()
    if mx_file:  # pretend mx is installed here so mlx include lookup works
        mx.__file__ = str(mx_file)
    elif py_path and "mlx-batch-src" in str(py_path):
        # nemotron's steel header expansion reads Path(mx.__file__).parent/include;
        # build that layout on the fly with include/mlx -> the pinned mlx source tree.
        pinned_mlx_pkg = PINNED_ROOT / "mlx" / "mlx"
        if pinned_mlx_pkg.is_dir():
            shim = pathlib.Path(tempfile.mkdtemp(prefix="m2v-mxinc"))
            (shim / "include").mkdir()
            (shim / "include" / "mlx").symlink_to(pinned_mlx_pkg)
            mx.__file__ = str(shim / "mx.py")
    mlx = types.ModuleType("mlx")
    mlx.core = mx
    sys.modules["mlx"] = mlx
    sys.modules["mlx.core"] = mx
    sys.modules["mlx.nn"] = _permissive("mlx.nn")
    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.modules["numpy"] = _permissive("numpy")

    text = pathlib.Path(py_path).read_text()
    # stub intra-package imports: "from .x import a, b as c" -> dummy classes
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^from \.\S* import (.*)$", line)
        if m:
            names = [m.group(1)]
            depth = line.count("(") - line.count(")")
            while depth > 0:
                i += 1
                names.append(lines[i])
                depth += lines[i].count("(") - lines[i].count(")")
            body = " ".join(names).strip().rstrip(")")
            body = re.sub(r"[()\\]", " ", body)
            for item in body.split(","):
                item = item.strip()
                if not item:
                    continue
                real = item.split(" as ")[-1].strip()
                if real and real.isidentifier():
                    out.append(f"{real} = type({real!r}, (), {{}})\n")
            i += 1
            continue
        out.append(line)
        i += 1
    text = "".join(out)
    mlx = sys.modules["mlx"]
    mlx.__path__ = []  # allow "from mlx.utils import ..." style imports
    sys.modules.setdefault("mlx.utils", _permissive("mlx.utils"))
    mod = types.ModuleType("pinned_capture")
    mod.__dict__["__name__"] = "pinned_capture"
    exec(compile(text, str(py_path), "exec"), mod.__dict__)
    return mod, mx.fast.metal_kernel


def call_factory(mod_ns, recorder, expr, args):
    fn = mod_ns
    for part in expr.split("."):
        fn = getattr(fn, part)
    args = [_dtype_sentinel(tuple(a) if isinstance(a, list) else a) for a in args]
    fn(*args)
    if not recorder.calls:
        raise SystemExit(f"factory {expr}({args}) recorded nothing (returned None?)")
    return recorder.calls[-1]


class _Dtype:
    _by_name = {}

    def __new__(cls, name):
        if name not in cls._by_name:
            cls._by_name[name] = object.__new__(cls)
        return cls._by_name[name]

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


def _dtype_sentinel(value):
    if isinstance(value, str) and value in TYPE_STRINGS:
        return _Dtype(value)
    return value


def build(reg, inputs, outputs, template_args):
    """reg: recorded kwargs. inputs: list of dicts(name, dtype, ndim, size).
    outputs: list of dtype strings. template_args: list of (name, value).
    Returns full MSL text and the kernel name."""
    header = reg.get("header", "") or ""
    source = reg["source"]
    input_names = reg["input_names"]
    output_names = reg["output_names"]
    assert len(input_names) == len(inputs), (input_names, inputs)
    assert len(output_names) == len(outputs)

    kernel_name = "custom_kernel_" + reg["name"]
    template_def = ""
    if template_args:
        template_def = "<" + ", ".join(template_arg_str(v) for _, v in template_args) + ">"
        kernel_name += "_" + make_template_hash(template_def)
    for inp in inputs:
        kernel_name += "_" + dtype_str(inp["dtype"])
        if inp["ndim"] == 0:
            kernel_name += "s"
        elif inp["size"] < MAX_CONSTANT_ARRAY_SIZE:
            kernel_name += "c"
    for odtype in outputs:
        kernel_name += "_" + dtype_str(odtype)

    shape_infos = [
        (
            source.find(n + "_shape") != -1,
            source.find(n + "_strides") != -1,
            source.find(n + "_ndim") != -1,
        )
        for n in input_names
    ]
    attributes = [
        (t, a) for a, t in METAL_ATTRIBUTES if source.find(a) != -1
    ]

    ks = [header]
    if template_args:
        parts = []
        for name, value in template_args:
            t = "int" if isinstance(value, int) and not isinstance(value, bool) else (
                "bool" if isinstance(value, bool) else "typename")
            parts.append(f"{t} {name}")
        ks.append("template <" + ", ".join(parts) + ">\n")
    ks.append(f"[[kernel]] void {kernel_name}(\n")
    index = 0
    for i, (name, inp) in enumerate(zip(input_names, inputs)):
        loc = "constant" if inp["size"] < MAX_CONSTANT_ARRAY_SIZE else "device"
        ref = "&" if inp["ndim"] == 0 else "*"
        ks.append(f"  const {loc} {dtype_str(inp['dtype'])}{ref} {name} [[buffer({index})]],\n")
        index += 1
        if inp["ndim"] > 0:
            if shape_infos[i][0]:
                ks.append(f"  const constant int* {name}_shape [[buffer({index})]],\n")
                index += 1
            if shape_infos[i][1]:
                ks.append(f"  const constant int64_t* {name}_strides [[buffer({index})]],\n")
                index += 1
            if shape_infos[i][2]:
                ks.append(f"  const constant int& {name}_ndim [[buffer({index})]],\n")
                index += 1
    for i, name in enumerate(output_names):
        ks.append(f"  device {dtype_str(outputs[i])}* {name} [[buffer({index})]]")
        if index < len(inputs) + len(output_names) - 1 or attributes:
            ks.append(",\n")
        else:
            ks.append(") {\n")
        index += 1
    for i, (t, a) in enumerate(attributes):
        ks.append(f"  {t} {a} [[{a}]]")
        if i < len(attributes) - 1:
            ks.append(",\n")
        else:
            ks.append(") {\n")
    ks.append(source)
    ks.append("\n}\n")
    if template_args:
        full = kernel_name + template_def
        ks.append(
            f"\ntemplate [[host_name(\"{kernel_name}\")]] [[kernel]] decltype({full}) {full};\n"
        )
    return "".join(ks), kernel_name


def main():
    spec_path = pathlib.Path(sys.argv[1])
    spec = json.loads(spec_path.read_text())
    recorder = None
    reg = None
    mod_ns = None
    module = spec["module"]
    if not os.path.isabs(module):  # spec paths are relative to the pinned root
        module = str(PINNED_ROOT / module)
    if spec.get("capture") is not None:  # module-level registration by index
        mod_ns, recorder = capture_module(module, spec.get("mx_file"))
        reg = recorder.calls[spec["capture"]]
    else:
        mod_ns, recorder = capture_module(module, spec.get("mx_file"))
        reg = call_factory(mod_ns, recorder, spec["factory"], spec.get("args", []))
    inputs = spec["inputs"]
    if reg["name"] != spec.get("expect_name"):
        print(f"NOTE name: captured {reg['name']!r} vs spec {spec.get('expect_name')!r}")
    msl, kernel_name = build(reg, inputs, spec["outputs"], spec["template"])
    out = HERE / f"{spec.get('out') or reg['name']}.metal"
    origin = spec["origin"]
    call = spec["call"]
    text = f"// Origin: {origin}.\n// Representative call: {call}.\n{msl}"
    out.write_text(text)
    print(json.dumps({"file": out.name, "kernel": kernel_name, "sha256": hashlib.sha256(text.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
