#!/usr/bin/env python3
"""Emit assemble.py spec JSONs for every vlmB metal_kernel call site."""
import json
import pathlib
import sys

SPEC_DIR = pathlib.Path(__file__).resolve().parent / "specs"
# pinned tree root is resolved by assemble.py (MLX_BATCH_SRC_ROOT); store paths
# relative to it so committed specs carry no machine-specific prefixes
ORIGIN = "mlx-vlm@8f5dc3d"

F16 = "float16"
F32 = "float32"
BF16 = "bfloat16"
U32 = "uint32"
I32 = "int32"
U8 = "uint8"


def arr(name, dtype, shape):
    size = 1
    for s in shape:
        size *= s
    return {"name": name, "dtype": dtype, "ndim": len(shape), "size": size}


def spec(module, kind, key, origin_line, call, inputs, outputs, template,
         expect_name=None, factory=None, args=None):
    path = {"tq": "mlx-vlm/mlx_vlm/turboquant.py",
            "qv": "mlx-vlm/mlx_vlm/models/quantized_verifier.py",
            "nem": "mlx-vlm/mlx_vlm/models/nemotron_labs_diffusion/language.py",
            "rope": "mlx-vlm/mlx_vlm/models/rope_utils.py",
            "q35l": "mlx-vlm/mlx_vlm/models/qwen3_5/language.py",
            "q35g": "mlx-vlm/mlx_vlm/models/qwen3_5/gated_delta.py",
            "1bit": "mlx-vlm/mlx_vlm/quantization/one_bit.py",
            "rwkv7": "mlx-vlm/mlx_vlm/models/rwkv7/language.py",
            "sparse": "mlx-vlm/mlx_vlm/models/sparse_attention.py",
            "qsa": "mlx-vlm/mlx_vlm/models/qwen4_exp/qsa_kernel.py"}[module]
    s = {"module": path, "origin": f"{ORIGIN}, {origin_line}", "call": call,
         "inputs": inputs, "outputs": outputs,
         "template": [[n, v] for n, v in template]}
    if kind == "capture":
        s["capture"] = key
    else:
        s["factory"] = factory
        s["args"] = args
    if expect_name:
        s["expect_name"] = expect_name
    return s


S = []

# ---------------- rwkv7 ----------------
S.append(spec("rwkv7", "capture", 0, "mlx_vlm/models/rwkv7/language.py:96",
    "r/w/k/v/a/b (1,2,2,64) float32, state (1,2,64,64) float32, T scalar int32; "
    "InT=float32,H=2,D=64; grid=(32,64,2), threadgroup=(32,4,1); y (1,2,2,64), state_out (1,2,64,64)",
    [arr(n, F32, (1, 2, 2, 64)) for n in ("r", "w", "k", "v", "a", "b")]
    + [arr("state_in", F32, (1, 2, 64, 64)), arr("T", I32, ())],
    [F32, F32],
    [("InT", F32), ("H", 2), ("D", 64)], expect_name="wkv7_kernel"))

# ---------------- sparse_attention ----------------
S.append(spec("sparse", "factory", None, "mlx_vlm/models/sparse_attention.py:131",
    "queries/keys/values (1,2,4,64)/(1,1,8,64)/(1,1,8,64) float16, indices (1,4,2) int32, "
    "scale (1) float32, q_size/k_size (1) int32; T=float16,QK_DIM=64,V_DIM=64,TOPK=2,"
    "NUM_Q_HEADS=2,NUM_KV_HEADS=1,GQA_FACTOR=2; grid=(1024,8,1), threadgroup=(1024,1,1); out (1,2,4,64)",
    [arr("queries", F16, (1, 2, 4, 64)), arr("keys", F16, (1, 1, 8, 64)),
     arr("values", F16, (1, 1, 8, 64)), arr("indices", I32, (1, 4, 2)),
     arr("scale", F32, (1,)), arr("q_size", I32, (1,)), arr("k_size", I32, (1,))],
    [F16],
    [("T", F16), ("QK_DIM", 64), ("V_DIM", 64), ("TOPK", 2),
     ("NUM_Q_HEADS", 2), ("NUM_KV_HEADS", 1), ("GQA_FACTOR", 2)],
    factory="_indexed_sparse_attention_kernel", args=[F16, 64, 64, 2, 2, 1]))

# ---------------- qsa ----------------
S.append(spec("qsa", "factory", None, "mlx_vlm/models/qwen4_exp/qsa_kernel.py:180",
    "queries/keys/values (1,2,4,64)/(1,1,8,64)/(1,1,8,64) float16, block_indices (1,4,1) int32, "
    "query_ends (1,4) int32, scale (1) float32, k_size (1) int32; T=float16,D_SIZE=64,Q_LEN=4,"
    "NUM_Q_HEADS=2,NUM_KV_HEADS=1,GQA_FACTOR=2,BLOCK_SIZE=8,TOPK_BLOCKS=1,SELECTED_LENGTH=8; "
    "grid=(1024,8,1), threadgroup=(1024,1,1); out (1,2,4,64)",
    [arr("queries", F16, (1, 2, 4, 64)), arr("keys", F16, (1, 1, 8, 64)),
     arr("values", F16, (1, 1, 8, 64)), arr("block_indices", I32, (1, 4, 1)),
     arr("query_ends", I32, (1, 4)), arr("scale", F32, (1,)), arr("k_size", I32, (1,))],
    [F16],
    [("T", F16), ("D_SIZE", 64), ("Q_LEN", 4), ("NUM_Q_HEADS", 2),
     ("NUM_KV_HEADS", 1), ("GQA_FACTOR", 2), ("BLOCK_SIZE", 8),
     ("TOPK_BLOCKS", 1), ("SELECTED_LENGTH", 8)],
    factory="_qsa_sparse_attention_kernel",
    args=[F16, 64, 8, 8, 1, 2, 1]))

# ---------------- rope_utils ----------------
S.append(spec("rope", "factory", None, "mlx_vlm/models/rope_utils.py:647",
    "x (1,2,4,32) float32, position_ids (1,4) int32, inv_freq (16) float32, "
    "position_selector (16) int32; T=float32, rotary_dim=32, position_ndim=2, half_split; "
    "grid=(128,1,1), threadgroup=(256,1,1); x_out (1,2,4,32)",
    [arr("x", F32, (1, 2, 4, 32)), arr("position_ids", I32, (1, 4)),
     arr("inv_freq", F32, (16,)), arr("position_selector", I32, (16,))],
    [F32], [("T", F32)],
    factory="_mrope_apply_kernel", args=[32, 2, "half_split"]))
S.append(spec("rope", "factory", None, "mlx_vlm/models/rope_utils.py:863",
    "x (1,2,4,32) float32, cos/sin (1,4,16) float32; T=float32, rotary_dim=32, "
    "half_split, cos_ndim=3, not sectioned, half cos layout; grid=(128,1,1), "
    "threadgroup=(256,1,1); x_out (1,2,4,32)",
    [arr("x", F32, (1, 2, 4, 32)), arr("cos", F32, (1, 4, 16)), arr("sin", F32, (1, 4, 16))],
    [F32], [("T", F32)],
    factory="_rotary_apply_kernel", args=[32, "half_split", 3, False, "half"]))

# ---------------- qwen3_5/language.py ----------------
Q35_IN = lambda: [arr("queries", F16, (1, 2, 1, 64)), arr("keys", F16, (1, 1, 8, 64)),
                  arr("values", F16, (1, 1, 8, 64)), arr("pads", I32, (1,)),
                  arr("scale", F32, (1,)), arr("k_size", I32, (1,))]
S.append(spec("q35l", "factory", None, "mlx_vlm/models/qwen3_5/language.py:570",
    "decode: queries (1,2,1,64) float16, keys/values (1,1,8,64), pads (1) int32, "
    "scale (1) float32, k_size (1) int32; T=float16,D_SIZE=64,V_SIZE=64,NUM_Q_HEADS=2,"
    "NUM_KV_HEADS=1,GQA_FACTOR=2; grid=(1024,2,1), threadgroup=(1024,1,1); out (1,2,1,64)",
    Q35_IN(), [F16],
    [("T", F16), ("D_SIZE", 64), ("V_SIZE", 64), ("NUM_Q_HEADS", 2),
     ("NUM_KV_HEADS", 1), ("GQA_FACTOR", 2)],
    factory="_qwen3_5_ragged_sdpa_one_pass_kernel", args=[F16, 64, 64]))
S.append(spec("q35l", "factory", None, "mlx_vlm/models/qwen3_5/language.py:582",
    "decode two-pass 1: queries (1,2,1,64) float16, keys/values (1,1,8,64), pads (1) int32, "
    "scale (1) float32, k_size (1) int32; T=float16,D_SIZE=64,V_SIZE=64,NUM_Q_HEADS=2,"
    "NUM_KV_HEADS=1,GQA_FACTOR=2,BLOCKS=2; grid=(32,2,2), threadgroup=(32,2,1); "
    "partials (1,2,1,2,64) float16, sums/maxs (1,2,1,2) float32",
    Q35_IN(), [F16, F32, F32],
    [("T", F16), ("D_SIZE", 64), ("V_SIZE", 64), ("NUM_Q_HEADS", 2),
     ("NUM_KV_HEADS", 1), ("GQA_FACTOR", 2), ("BLOCKS", 2)],
    factory="_qwen3_5_ragged_sdpa_two_pass_1_kernel", args=[F16, 64, 64, 2]))
S.append(spec("q35l", "factory", None, "mlx_vlm/models/qwen3_5/language.py:596",
    "decode two-pass 2: partials (1,2,1,8,64) float16, sums/maxs (1,2,1,8) float32; "
    "T=float16,D_SIZE=64,BLOCKS=8; grid=(1024,2,1), threadgroup=(1024,1,1); out (1,2,1,64)",
    [arr("partials", F16, (1, 2, 1, 8, 64)), arr("sums", F32, (1, 2, 1, 8)),
     arr("maxs", F32, (1, 2, 1, 8))],
    [F16],
    [("T", F16), ("D_SIZE", 64), ("BLOCKS", 8)],
    factory="_qwen3_5_ragged_sdpa_two_pass_2_kernel", args=[F16, 64, 8]))

# ---------------- qwen3_5/gated_delta.py ----------------
GD_IN_VEC = [arr(n, F32, (1, 2, 1, 32)) for n in ("q", "k", "v", "g")]
GD_IN_SCALAR_G = [arr("q", F32, (1, 2, 1, 32)), arr("k", F32, (1, 2, 1, 32)),
                  arr("v", F32, (1, 2, 1, 32)), arr("g", F32, (1, 2, 1))]
GD_T = [("InT", F32), ("StT", F32), ("Dk", 32), ("Dv", 32), ("Hk", 1), ("Hv", 1)]
GD_STATE = arr("state_in", F32, (1, 1, 32, 32))
GD_TS = arr("T", I32, ())
GD_OUT = [F32, F32]
for idx, (vname, ins, tmpl, note) in enumerate([
    (0, GD_IN_SCALAR_G, GD_T, "g (1,2,1) float32"),
    (1, GD_IN_SCALAR_G, GD_T, "g (1,2,1), mask (1,2) bool"),
    (2, GD_IN_VEC, GD_T, "vectorized g (1,2,1,32) float32"),
    (3, GD_IN_VEC, GD_T, "vectorized g (1,2,1,32), mask (1,2) bool"),
]):
    suffix = ("_vec" if idx >= 2 else "") + ("_mask" if idx % 2 else "")
    extra = [arr("mask", "bool", (1, 2))] if idx % 2 else []
    S.append(spec("q35g", "capture", idx, "mlx_vlm/models/qwen3_5/gated_delta.py:109",
        f"q/k/v float32 (1,2,1,32), {note}, beta (1,2,1) float32, state (1,1,32,32) "
        f"float32, T scalar int32; grid=(32,32,1), threadgroup=(32,4,1); "
        f"y (1,2,1,32), state_out (1,1,32,32)",
        ins + [arr("beta", F32, (1, 2, 1)), GD_STATE, GD_TS] + extra, GD_OUT, tmpl,
        expect_name="gated_delta_step" + suffix))
for idx, mask in ((4, False), (5, True)):
    suffix = "_mask" if mask else ""
    ins = GD_IN_SCALAR_G + [arr("beta", F32, (1, 2, 1)), GD_STATE, GD_TS]
    if mask:
        ins.append(arr("mask", "bool", (1, 2)))
    S.append(spec("q35g", "capture", idx, "mlx_vlm/models/qwen3_5/gated_delta.py:540",
        f"q/k/v float32 (1,2,1,32), g (1,2,1) float32, beta (1,2,1), state (1,1,32,32) "
        f"float32, T scalar int32{', mask (1,2) bool' if mask else ''}; "
        f"InT=StT=float32,Dk=Dv=32,Hk=Hv=1,StateT=2; grid=(32,32,1), threadgroup=(32,4,1); "
        f"y (1,2,1,32), state_out (1,1,32,32), states (1,2,1,32,32)",
        ins, [F32, F32, F32], GD_T + [("StateT", 2)],
        expect_name="qwen3_5_gated_delta_with_states" + suffix))

# ---------------- one_bit.py ----------------
OB_IN = [arr("x", F16, (1, 512)), arr("weight", U32, (64, 16)),
         arr("scales", F16, (64, 8)), arr("biases", F16, (64, 8))]
S.append(spec("1bit", "factory", None, "mlx_vlm/quantization/one_bit.py:224",
    "x (1,512) float16, weight uint32 (64,16), scales/biases (64,8) float16; "
    "T=float16, group_size=64, 4 outputs per simdgroup, row-aligned; grid=(512,1,1), "
    "threadgroup=(64,1,1); out (1,64) float16",
    OB_IN, [F16], [("T", F16)],
    factory="_one_bit_qmv_kernel", args=[64, True, 4]))
S.append(spec("1bit", "factory", None, "mlx_vlm/quantization/one_bit.py:240",
    "x (16,512) float16, weight uint32 (64,16), scales/biases (64,8) float16; "
    "T=float16, group_size=64; grid=(128,1,1), threadgroup=(128,1,1); out (1024) float16",
    [arr("x", F16, (16, 512))] + OB_IN[1:], [F16], [("T", F16)],
    factory="_one_bit_qmm_kernel", args=[64]))

# ---------------- nemotron ----------------
S.append(spec("nem", "capture", 0, "mlx_vlm/models/nemotron_labs_diffusion/language.py:56",
    "x (32,64) bfloat16, weight (128,64) bfloat16; T=bfloat16,K=64,O=128; "
    "grid=(256,1,1), threadgroup=(128,1,1); out (32,128)",
    [arr("x", BF16, (32, 64)), arr("weight", BF16, (128, 64))],
    [BF16], [("T", BF16), ("K", 64), ("O", 128)],
    expect_name="nemotron_bm32_steel_linear_nt"))
S.append(spec("nem", "capture", 1, "mlx_vlm/models/nemotron_labs_diffusion/language.py:203",
    "x (8,64) float16, weight (64,64) float16; T=float16,K=64,O=64,R=8; "
    "grid=(32,16,8), threadgroup=(32,1,8); out (8,64)",
    [arr("x", F16, (8, 64)), arr("weight", F16, (64, 64))],
    [F16], [("T", F16), ("K", 64), ("O", 64), ("R", 8)],
    expect_name="nemotron_small_row_gemv"))
S.append(spec("nem", "capture", 2, "mlx_vlm/models/nemotron_labs_diffusion/language.py:435",
    "x (8,64) float16, gate/up weights (64,64) float16; T=float16,K=64,O=64,R=8; "
    "grid=(32,16,8), threadgroup=(32,1,8); out (8,64)",
    [arr("x", F16, (8, 64)), arr("gate_weight", F16, (64, 64)),
     arr("up_weight", F16, (64, 64))],
    [F16], [("T", F16), ("K", 64), ("O", 64), ("R", 8)],
    expect_name="nemotron_small_row_swiglu"))

# ---------------- quantized_verifier ----------------
def qv_qmm_in(t_rows):
    return [arr("x", F16, (1, t_rows, 512)),
            arr("w", U32, (64, 64)), arr("scales", F16, (64, 8)), arr("biases", F16, (64, 8))]

QV_T2 = [("T", F16), ("VERIFY_T", 2), ("K_SIZE", 512), ("N_SIZE", 64)]
S.append(spec("qv", "factory", None, "mlx_vlm/models/quantized_verifier.py:1457",
    "x (1,2,512) float16, w uint32 (64,64), scales/biases (64,8) float16; "
    "T=float16,VERIFY_T=2,K_SIZE=512,N_SIZE=64; grid=(32,16,1), threadgroup=(32,2,1); y (1,2,64)",
    qv_qmm_in(2), [F16], QV_T2,
    factory="_target_verify_qmv_kernel", args=[4, 64, F16, 2, 512, 64]))
S.append(spec("qv", "factory", None, "mlx_vlm/models/quantized_verifier.py:1472",
    "as qmv; outputs tile_values (1,2,8) float16, tile_indices (1,2,8) int32; NUM_TILES=8; grid=(32,16,1)",
    qv_qmm_in(2), [F16, I32], QV_T2 + [("NUM_TILES", 8)],
    factory="_target_verify_qargmax_kernel", args=[4, 64, F16, 2, 512, 64]))
S.append(spec("qv", "factory", None, "mlx_vlm/models/quantized_verifier.py:1489",
    "as qargmax plus packed mask (2,2) int32 (mask_shape buffer); grid=(32,16,1)",
    qv_qmm_in(2) + [arr("mask", I32, (2, 2))], [F16, I32], QV_T2 + [("NUM_TILES", 8)],
    factory="_target_verify_masked_qargmax_kernel", args=[4, 64, F16, 2, 512, 64]))
S.append(spec("qv", "factory", None, "mlx_vlm/models/quantized_verifier.py:1506",
    "token-tiled: x (1,6,512) float16; VERIFY_T=6; grid=(32,16,3), threadgroup=(32,2,1); y (1,6,64)",
    qv_qmm_in(6), [F16],
    [("T", F16), ("VERIFY_T", 6), ("K_SIZE", 512), ("N_SIZE", 64)],
    factory="_target_verify_qmv_token_tiled_kernel", args=[4, 64, F16, 6, 512, 64]))
S.append(spec("qv", "factory", None, "mlx_vlm/models/quantized_verifier.py:1523",
    "token-tiled qargmax: x (1,6,512) float16; VERIFY_T=6,NUM_TILES=8; grid=(32,16,3); "
    "tile_values (1,6,8) float16, tile_indices (1,6,8) int32",
    qv_qmm_in(6), [F16, I32],
    [("T", F16), ("VERIFY_T", 6), ("K_SIZE", 512), ("N_SIZE", 64), ("NUM_TILES", 8)],
    factory="_target_verify_qargmax_token_tiled_kernel", args=[4, 64, F16, 6, 512, 64]))
S.append(spec("qv", "factory", None, "mlx_vlm/models/quantized_verifier.py:1540",
    "token-tiled masked: x (1,6,512) float16, mask (6,2) int32; grid=(32,16,3)",
    qv_qmm_in(6) + [arr("mask", I32, (6, 2))], [F16, I32],
    [("T", F16), ("VERIFY_T", 6), ("K_SIZE", 512), ("N_SIZE", 64), ("NUM_TILES", 8)],
    factory="_target_verify_masked_qargmax_token_tiled_kernel", args=[4, 64, F16, 6, 512, 64]))
S.append(spec("qv", "factory", None, "mlx_vlm/models/quantized_verifier.py:1557",
    "streamed: x (1,8,512) float16; VERIFY_T=8, one result per simdgroup; grid=(32,64,1), "
    "threadgroup=(32,2,1); y (1,8,64)",
    qv_qmm_in(8), [F16],
    [("T", F16), ("VERIFY_T", 8), ("K_SIZE", 512), ("N_SIZE", 64)],
    factory="_target_verify_qmv_streamed_kernel", args=[4, 64, F16, 8, 512, 64]))
S.append(spec("qv", "factory", None, "mlx_vlm/models/quantized_verifier.py:1574",
    "streamed qargmax: x (1,8,512) float16; VERIFY_T=8,NUM_TILES=32; grid=(32,64,1); "
    "tile_values (1,8,32) float16, tile_indices (1,8,32) int32",
    qv_qmm_in(8), [F16, I32],
    [("T", F16), ("VERIFY_T", 8), ("K_SIZE", 512), ("N_SIZE", 64), ("NUM_TILES", 32)],
    factory="_target_verify_qargmax_streamed_kernel", args=[4, 64, F16, 8, 512, 64]))
S.append(spec("qv", "factory", None, "mlx_vlm/models/quantized_verifier.py:1591",
    "streamed masked: x (1,8,512) float16, mask (8,2) int32; grid=(32,64,1)",
    qv_qmm_in(8) + [arr("mask", I32, (8, 2))], [F16, I32],
    [("T", F16), ("VERIFY_T", 8), ("K_SIZE", 512), ("N_SIZE", 64), ("NUM_TILES", 32)],
    factory="_target_verify_masked_qargmax_streamed_kernel", args=[4, 64, F16, 8, 512, 64]))
FUSED_IN = [arr("x", F16, (1, 2, 512))]
for i in (0, 1):
    FUSED_IN += [arr(f"w{i}", U32, (64, 64)), arr(f"scales{i}", F16, (64, 8)),
                 arr(f"biases{i}", F16, (64, 8))]
FUSED_T = [("T", F16), ("VERIFY_T", 2), ("K_SIZE", 512), ("N_SIZE", 128)]
S.append(spec("qv", "factory", None, "mlx_vlm/models/quantized_verifier.py:1610",
    "fused two linears: x (1,2,512) float16, w0/w1 uint32 (64,64), scales/biases (64,8) "
    "float16; T=float16,VERIFY_T=2,K_SIZE=512,N_SIZE=128; grid=(32,32,1), "
    "threadgroup=(32,2,1); y (1,2,128)",
    FUSED_IN, [F16], FUSED_T,
    factory="_target_verify_fused_qmv_kernel", args=[4, 64, F16, 2, 512, [64, 64]]))
S.append(spec("qv", "factory", None, "mlx_vlm/models/quantized_verifier.py:1631",
    "fused two linears streamed: x (1,6,512) float16; VERIFY_T=6,N_SIZE=128; "
    "grid=(32,128,1), threadgroup=(32,2,1); y (1,6,128)",
    [arr("x", F16, (1, 6, 512))] + FUSED_IN[1:], [F16],
    [("T", F16), ("VERIFY_T", 6), ("K_SIZE", 512), ("N_SIZE", 128)],
    factory="_target_verify_fused_qmv_streamed_kernel", args=[4, 64, F16, 6, 512, [64, 64]]))
S.append(spec("qv", "factory", None, "mlx_vlm/models/quantized_verifier.py:477",
    "fused affine MoE+HC: x (1,2,2,512) float16, indices (1,2,2) int32, route_weights "
    "(1,2,2) float32, w uint32 (32,64), scales/biases (32,8) float16, shared (1,2,32), "
    "residual (1,2,2,32) float16, post/comb (1,2,2) float32; T=float16,BITS=4,GROUP_SIZE=64,"
    "K_SIZE=512,N_SIZE=32,TOP_K=2,HC=2; grid=(32,16,2), threadgroup=(32,4,1); out (1,2,2,32)",
    [arr("x", F16, (1, 2, 2, 512)), arr("indices", I32, (1, 2, 2)),
     arr("route_weights", F32, (1, 2, 2)), arr("w", U32, (32, 64)),
     arr("scales", F16, (32, 8)), arr("biases", F16, (32, 8)),
     arr("shared", F16, (1, 2, 32)), arr("residual", F16, (1, 2, 2, 32)),
     arr("post", F32, (1, 2, 2)), arr("comb", F32, (1, 2, 2))],
    [F16],
    [("T", F16), ("BITS", 4), ("GROUP_SIZE", 64), ("K_SIZE", 512),
     ("N_SIZE", 32), ("TOP_K", 2), ("HC", 2)],
    factory="_fused_quantized_moe_hc_kernel",
    args=["affine", 4, 64, F16, 2, 512, 32, 2, 2]))
S.append(spec("qv", "factory", None, "mlx_vlm/models/quantized_verifier.py:604",
    "MoE+HC epilogue: routed (1,2,2,64) float16, route_weights (1,2,2) float32, "
    "shared (1,2,64) float16, residual (1,2,2,64) float16, post/comb (1,2,2) float32; "
    "T=float16,ROWS=2,HC=2,D=64,SIMDS=8,TOP_K=2; grid=(64,8,1), threadgroup=(32,8,1); "
    "out (1,2,2,64)",
    [arr("routed", F16, (1, 2, 2, 64)), arr("route_weights", F32, (1, 2, 2)),
     arr("shared", F16, (1, 2, 64)), arr("residual", F16, (1, 2, 2, 64)),
     arr("post", F32, (1, 2, 2)), arr("comb", F32, (1, 2, 2))],
    [F16],
    [("T", F16), ("ROWS", 2), ("HC", 2), ("D", 64), ("SIMDS", 8), ("TOP_K", 2)],
    factory="_quantized_moe_hc_kernel", args=[F16, 2, 2, 64, 2]))
S.append(spec("qv", "factory", None, "mlx_vlm/models/quantized_verifier.py:1974",
    "nvfp4 argmax: x (1,2,64) float16, w uint32 (64,8), scales uint8 (64,4); "
    "T=float16,VERIFY_T=2,K_SIZE=64,N_SIZE=64,NUM_TILES=8; grid=(32,16,1), "
    "threadgroup=(32,2,1); tile_values (1,2,8) float16, tile_indices (1,2,8) int32",
    [arr("x", F16, (1, 2, 64)), arr("w", U32, (64, 8)), arr("scales", U8, (64, 4))],
    [F16, I32],
    [("T", F16), ("VERIFY_T", 2), ("K_SIZE", 64), ("N_SIZE", 64), ("NUM_TILES", 8)],
    factory="_nvfp4_argmax_kernel", args=[F16, 2, 64, 64]))

# ---------------- turboquant ----------------
def tq_in(extra=None, d=64, t=2, r=1, mse_pw=8, sign_pw=2, codebook=16):
    ins = [arr("q_rot", F16, (1, 1, r, d))]
    if extra in ("qjl", "prod", "prod_repeat", "polar_turbo", "fused", "split"):
        ins.append(arr("q_proj", F16, (1, 1, r, d)))
    ins.append(arr("norms", F16, (1, 1, t)))
    if extra == "polar_turbo":
        ins.append(arr("radii", U32, (1, 1, t, 1)))
    if extra in ("prod", "prod_repeat", "fused", "mq", "single_tile_fused",
                 "fully_fused", "split"):
        ins.append(arr("mse_packed", U32, (1, 1, t, mse_pw)) if extra != "fused"
                   else arr("key_mse", U32, (1, 1, t, mse_pw)))
    if extra in ("qjl", "prod", "prod_repeat", "polar_turbo", "fused", "split"):
        ins.append(arr("residual_norms", F16, (1, 1, t)))
    if extra in ("qjl", "prod", "prod_repeat", "polar_turbo", "fused", "split"):
        ins.append(arr("signs", U32, (1, 1, t, sign_pw)) if extra != "fused"
                   else arr("key_signs", U32, (1, 1, t, sign_pw)))
    return ins

scores_note = "grid=(32,1,2), threadgroup=(32,1,1)"

S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:66",
    "q_rot (1,1,1,64) float16, norms (1,1,2) float16, packed (1,1,2,8) uint32, "
    "codebook (16) float32; Dim=64,Bits=4,PackedWidth=8; " + scores_note + "; out (1,1,1,2) float32",
    [arr("q_rot", F16, (1, 1, 1, 64)), arr("norms", F16, (1, 1, 2)),
     arr("packed", U32, (1, 1, 2, 8)), arr("codebook", F32, (16,))],
    [F32], [("Dim", 64), ("Bits", 4), ("PackedWidth", 8)],
    factory="_mse_score_kernel", args=[]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:161",
    "tiled: q_rot (1,1,1,64) float16, norms (1,1,2) float16, packed (1,1,2,8) uint32, "
    "codebook (16) float32; Dim=64,DimsPerLane=2,Bits=4,PackedWidth=8,RepeatCount=1,"
    "TokTileSize=64; grid=(32,1,1), threadgroup=(32,1,1); out (1,1,1,2) float32",
    [arr("q_rot", F16, (1, 1, 1, 64)), arr("norms", F16, (1, 1, 2)),
     arr("packed", U32, (1, 1, 2, 8)), arr("codebook", F32, (16,))],
    [F32], [("Dim", 64), ("DimsPerLane", 2), ("Bits", 4), ("PackedWidth", 8),
            ("RepeatCount", 1), ("TokTileSize", 64)],
    factory="_mse_score_tiled_kernel", args=[1]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:205",
    "values uint32 (4,8); Bits=4,Length=8,PackedWidth=2; grid=(2,4,1), "
    "threadgroup=(2,1,1); out (4,2) uint32",
    [arr("values", U32, (4, 8))], [U32],
    [("Bits", 4), ("Length", 8), ("PackedWidth", 2)],
    factory="_pack_lowbit_kernel", args=[]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:237",
    "packed uint32 (4,2); Bits=4,Length=8,PackedWidth=2; grid=(8,4,1), "
    "threadgroup=(32,1,1); out (4,8) uint32",
    [arr("packed", U32, (4, 2))], [U32],
    [("Bits", 4), ("Length", 8), ("PackedWidth", 2)],
    factory="_unpack_lowbit_kernel", args=[]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:289",
    "q_proj (1,1,1,64) float16, norms/residual_norms (1,1,2) float16, signs (1,1,2,2) "
    "uint32, scale (1) float32; Dim=64,PackedWidth=2; " + scores_note + "; out (1,1,1,2) float32",
    [arr("q_proj", F16, (1, 1, 1, 64)), arr("norms", F16, (1, 1, 2)),
     arr("residual_norms", F16, (1, 1, 2)), arr("signs", U32, (1, 1, 2, 2)),
     arr("scale", F32, (1,))],
    [F32], [("Dim", 64), ("PackedWidth", 2)],
    factory="_qjl_score_kernel", args=[]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:356",
    "q_rot/q_proj (1,1,1,64) float16, norms/residual_norms (1,1,2) float16, "
    "mse_packed (1,1,2,8) uint32, signs (1,1,2,2) uint32, codebook (16) float32, "
    "scale (1) float32; Dim=64,MseBits=4,MsePackedWidth=8,SignPackedWidth=2; "
    + scores_note + "; out (1,1,1,2) float32",
    [arr("q_rot", F16, (1, 1, 1, 64)), arr("q_proj", F16, (1, 1, 1, 64)),
     arr("norms", F16, (1, 1, 2)), arr("residual_norms", F16, (1, 1, 2)),
     arr("mse_packed", U32, (1, 1, 2, 8)), arr("signs", U32, (1, 1, 2, 2)),
     arr("codebook", F32, (16,)), arr("scale", F32, (1,))],
    [F32], [("Dim", 64), ("MseBits", 4), ("MsePackedWidth", 8), ("SignPackedWidth", 2)],
    factory="_prod_score_kernel", args=[]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:421",
    "weights (1,1,1,2) float32, norms (1,1,2) float16, indices (1,1,2,8) uint32, "
    "codebook (16) float32; Dim=64,Bits=4,PackedWidth=8; grid=(32,64,1), "
    "threadgroup=(32,1,1); out (1,1,1,64) float32",
    [arr("weights", F32, (1, 1, 1, 2)), arr("norms", F16, (1, 1, 2)),
     arr("indices", U32, (1, 1, 2, 8)), arr("codebook", F32, (16,))],
    [F32], [("Dim", 64), ("Bits", 4), ("PackedWidth", 8)],
    factory="_mse_weighted_rot_kernel", args=[]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:523",
    "q_rot/q_proj (1,1,2,64) float16, norms/residual_norms (1,1,2) float16, "
    "mse_packed (1,1,2,8) uint32, signs (1,1,2,2) uint32, codebook (16) float32, "
    "scale (1) float32; Dim=64,RepeatCount=2,MseBits=4,MsePackedWidth=8,"
    "SignPackedWidth=2; " + scores_note + "; out (1,1,2,2) float32",
    [arr("q_rot", F16, (1, 1, 2, 64)), arr("q_proj", F16, (1, 1, 2, 64)),
     arr("norms", F16, (1, 1, 2)), arr("residual_norms", F16, (1, 1, 2)),
     arr("mse_packed", U32, (1, 1, 2, 8)), arr("signs", U32, (1, 1, 2, 2)),
     arr("codebook", F32, (16,)), arr("scale", F32, (1,))],
    [F32], [("Dim", 64), ("RepeatCount", 2), ("MseBits", 4),
            ("MsePackedWidth", 8), ("SignPackedWidth", 2)],
    factory="_prod_score_repeat_kernel", args=[2]))

POLAR_LEVELS = [(1, 1, 2, 2), (1, 1, 2, 1), (1, 1, 2, 1), (1, 1, 2, 1)]
polar_in = [arr("q_rot", F16, (1, 1, 1, 64)), arr("norms", F16, (1, 1, 2)),
            arr("radii", U32, (1, 1, 2, 1))]
for i, sh in enumerate(POLAR_LEVELS, 1):
    polar_in.append(arr(f"angles_{i}", U32, sh))
for i in range(1, 5):
    polar_in += [arr(f"cos_{i}", F32, (4,)), arr(f"sin_{i}", F32, (4,))]
polar_turbo_in = [arr("q_rot", F16, (1, 1, 1, 64)), arr("q_proj", F16, (1, 1, 1, 64)),
                  arr("norms", F16, (1, 1, 2)), arr("radii", U32, (1, 1, 2, 1))]
for i, sh in enumerate(POLAR_LEVELS, 1):
    polar_turbo_in.append(arr(f"angles_{i}", U32, sh))
polar_turbo_in += [arr("residual_norms", F16, (1, 1, 2)), arr("signs", U32, (1, 1, 2, 2)),
                   arr("scale", F32, (1,))]
for i in range(1, 5):
    polar_turbo_in += [arr(f"cos_{i}", F32, (4,)), arr(f"sin_{i}", F32, (4,))]
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:610",
    "polar levels (2,2,2,2): q_rot (1,1,1,64) float16, norms (1,1,2) float16, "
    "radii (1,1,2,1) uint32, angle levels uint32 (1,1,2,2)/(1,1,2,1)x3, cos/sin tables "
    "(4) float32 x8; Dim=64,Levels=4,BlockCount=1,PackedWidth1=2,PackedWidth2..4=1; " + scores_note + "; out (1,1,1,2) float32",
    polar_in, [F32],
    [("Dim", 64), ("Levels", 4), ("BlockCount", 1), ("PackedWidth1", 2),
     ("PackedWidth2", 1), ("PackedWidth3", 1), ("PackedWidth4", 1)],
    factory="_polar_prod_score_kernel", args=[[2, 2, 2, 2]]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:802",
    "polar turbo levels (2,2,2,2): q_rot/q_proj (1,1,1,64) float16, norms (1,1,2) float16, "
    "radii uint32 (1,1,2,1), angle levels uint32, residual_norms (1,1,2) float16, "
    "signs (1,1,2,2) uint32, scale (1) float32, cos/sin tables (4) float32 x8; "
    "Dim=64,Levels=4,RepeatCount=1,BlockCount=1,SignPackedWidth=2,PackedWidth1=2,"
    "PackedWidth2..4=1; " + scores_note + "; out (1,1,1,2) float32",
    polar_turbo_in, [F32],
    [("Dim", 64), ("Levels", 4), ("RepeatCount", 1), ("BlockCount", 1),
     ("SignPackedWidth", 2), ("PackedWidth1", 2), ("PackedWidth2", 1),
     ("PackedWidth3", 1), ("PackedWidth4", 1)],
    factory="_polar_turbo_score_repeat_kernel", args=[[2, 2, 2, 2], 1]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:879",
    "weights (1,1,2,2) float32, norms (1,1,2) float16, indices (1,1,2,8) uint32, "
    "codebook (16) float32; Dim=64,RepeatCount=2,Bits=4,PackedWidth=8; grid=(32,64,1), "
    "threadgroup=(32,1,1); out (1,1,2,64) float32",
    [arr("weights", F32, (1, 1, 2, 2)), arr("norms", F16, (1, 1, 2)),
     arr("indices", U32, (1, 1, 2, 8)), arr("codebook", F32, (16,))],
    [F32], [("Dim", 64), ("RepeatCount", 2), ("Bits", 4), ("PackedWidth", 8)],
    factory="_mse_weighted_rot_repeat_kernel", args=[2]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:967",
    "scores (1,1,2,2) float32, norms (1,1,2) float16, packed (1,1,2,8) uint32, "
    "codebook (16) float32, max_scores (1,1,2) float32; Dim=64,Bits=4,PackedWidth=8; "
    "grid=(32,64,1), threadgroup=(32,1,1); out (1,1,2,64) float32",
    [arr("scores", F32, (1, 1, 2, 2)), arr("norms", F16, (1, 1, 2)),
     arr("packed", U32, (1, 1, 2, 8)), arr("codebook", F32, (16,)),
     arr("max_scores", F32, (1, 1, 2))],
    [F32], [("Dim", 64), ("Bits", 4), ("PackedWidth", 8)],
    factory="_mse_scores_weighted_rot_repeat_kernel", args=[2]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:1051",
    "as scores-weighted-rot; sum variant: same inputs; Dim=64,Bits=4,PackedWidth=8; "
    "grid=(32,64,1), threadgroup=(32,1,1); out (1,1,2,64) float32",
    [arr("scores", F32, (1, 1, 2, 2)), arr("norms", F16, (1, 1, 2)),
     arr("packed", U32, (1, 1, 2, 8)), arr("codebook", F32, (16,)),
     arr("max_scores", F32, (1, 1, 2))],
    [F32], [("Dim", 64), ("Bits", 4), ("PackedWidth", 8)],
    factory="_mse_scores_weighted_rot_sum_repeat_kernel", args=[2]))

FUSED_IN_TQ = [arr("q_rot", F16, (1, 1, 1, 64)), arr("q_proj", F16, (1, 1, 1, 64)),
               arr("key_norms", F16, (1, 1, 2)), arr("key_mse", U32, (1, 1, 2, 8)),
               arr("key_res_norms", F16, (1, 1, 2)), arr("key_signs", U32, (1, 1, 2, 2)),
               arr("val_norms", F16, (1, 1, 2)), arr("val_packed", U32, (1, 1, 2, 8)),
               arr("key_codebook", F32, (16,)), arr("key_scale", F32, (1,)),
               arr("val_codebook", F32, (16,))]
FUSED_T_TQ = [("Dim", 64), ("RepeatCount", 1), ("TokTileSize", 1024),
              ("KMsePackedWidth", 8), ("KSignPackedWidth", 2), ("VPackedWidth", 8)]
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:1713",
    "q_rot/q_proj (1,1,1,64) float16, key_norms/val_norms (1,1,2) float16, "
    "key_mse/val_packed (1,1,2,8) uint32, key_res_norms (1,1,2) float16, "
    "key_signs (1,1,2,2) uint32, codebooks (16) float32, key_scale (1) float32; "
    "Dim=64,RepeatCount=1,TokTileSize=1024,KMsePackedWidth=8,KSignPackedWidth=2,"
    "VPackedWidth=8,ValBits=4; grid=(32,1,1), threadgroup=(32,1,1); out_acc (1,1,1,64) float32, "
    "out_sum/out_max (1,1,1) float32",
    FUSED_IN_TQ, [F32, F32, F32], FUSED_T_TQ + [("ValBits", 4)],
    factory="_fused_integer_decode_kernel", args=[4, 1, 4]))
MQ_IN = [arr("q_rot", F16, (2, 2, 64)), arr("q_proj", F16, (2, 2, 64)),
         arr("key_norms", F16, (1, 1, 2)), arr("key_mse", U32, (1, 1, 2, 8)),
         arr("key_res_norms", F16, (1, 1, 2)), arr("key_signs", U32, (1, 1, 2, 2)),
         arr("key_codebook", F32, (16,)), arr("key_scale", F32, (1,))]
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:1744",
    "multi-query: q_rot/q_proj (2,2,64) float16, key_norms/key_res_norms (1,1,2) "
    "float16, key_mse (1,1,2,8) uint32, key_signs (1,1,2,2) uint32, codebook (16) "
    "float32, scale (1) float32; Dim=64,KMsePackedWidth=8,KSignPackedWidth=2; "
    "grid=(32,2,2), threadgroup=(32,1,1); out (2,2,2) float32",
    MQ_IN, [F32],
    [("Dim", 64), ("KMsePackedWidth", 8), ("KSignPackedWidth", 2)],
    factory="_multi_query_prod_score_kernel", args=[4, 2, 2, 2]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:1859",
    "single-tile value: weights (1,2,2) float32, norms (1,1,2) float16, "
    "indices (1,1,2,8) uint32, codebook (16) float32; Dim=64,RepeatCount=2,"
    "TokTileSize=1024,DimsPerLane=2,PackedWidth=8; grid=(64,1,1), threadgroup=(64,1,1); "
    "out (1,2,64) float32",
    [arr("weights", F32, (1, 2, 2)), arr("norms", F16, (1, 1, 2)),
     arr("indices", U32, (1, 1, 2, 8)), arr("codebook", F32, (16,))],
    [F32],
    [("Dim", 64), ("RepeatCount", 2), ("TokTileSize", 1024), ("DimsPerLane", 2),
     ("PackedWidth", 8)],
    factory="_single_tile_value_weighted_sum_kernel", args=[4, 2, 2]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:2000",
    "single-tile fused: as fused integer decode; Dim=64,RepeatCount=1,DimsPerLane=2,"
    "TokTileSize=1024,KMsePackedWidth=8,KSignPackedWidth=2,VPackedWidth=8,ValBits=4; "
    "grid=(32,2,1), threadgroup=(32,1,1); out_acc (1,1,64) float32, out_sum/out_max (1,) float32",
    FUSED_IN_TQ, [F32, F32, F32],
    FUSED_T_TQ + [("DimsPerLane", 2), ("ValBits", 4)],
    factory="_fused_integer_decode_single_tile_kernel", args=[4, 1, 2, 4]))
FF_IN = FUSED_IN_TQ + [arr("rot_input", F32, (64, 64))]
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:2209",
    "fully fused: as fused integer decode plus rot_input (64,64) float32; "
    "Dim=64,RepeatCount=1,DimsPerLane=2,KMsePackedWidth=8,KSignPackedWidth=2,"
    "VPackedWidth=8; grid=(32,1,1), threadgroup=(32,1,1); out (1,1,1,64) float32",
    FF_IN, [F32],
    [("Dim", 64), ("RepeatCount", 1), ("DimsPerLane", 2), ("KMsePackedWidth", 8),
     ("KSignPackedWidth", 2), ("VPackedWidth", 8)],
    factory="_fully_fused_decode_kernel", args=[4, 1, 2, 4, False]))
SDPA_IN = [arr("q_rot_flat", F16, (2, 64)), arr("key_norms", F16, (1, 1, 2)),
           arr("key_packed", U32, (1, 1, 2, 8)), arr("key_codebook", F32, (16,)),
           arr("val_norms", F16, (1, 1, 2)), arr("val_packed", U32, (1, 1, 2, 8)),
           arr("val_codebook", F32, (16,))]
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:2421",
    "fused mse sdpa single pass: q_rot_flat (2,64) float16, key/val norms (1,1,2) "
    "float16, key/val packed (1,1,2,8) uint32, codebooks (16) float32; "
    "Dim=64,RepeatCount=2,KPackedWidth=8,VPackedWidth=8; grid=(2048,1,1), "
    "threadgroup=(1024,1,1); out (2,64) float32",
    SDPA_IN, [F32],
    [("Dim", 64), ("RepeatCount", 2), ("KPackedWidth", 8), ("VPackedWidth", 8)],
    factory="_fused_mse_decode_kernel", args=[4, 4, 64]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:2525",
    "fused mse sdpa two-pass 1: same inputs as single pass; Dim=64,RepeatCount=2,"
    "Blocks=64,KPackedWidth=8,VPackedWidth=8; grid=(32,2,64), threadgroup=(32,2,1); "
    "out_acc (128,64) float32, out_sums/out_maxs (128) float32",
    SDPA_IN, [F32, F32, F32],
    [("Dim", 64), ("RepeatCount", 2), ("Blocks", 64), ("KPackedWidth", 8),
     ("VPackedWidth", 8)],
    factory="_fused_mse_decode_2pass_1_kernel", args=[4, 4, 64]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:2617",
    "fused mse sdpa two-pass 2: out_acc (128,64) float32, out_sums/out_maxs (128) "
    "float32; Dim=64,Blocks=64; grid=(2048,1,1), threadgroup=(1024,1,1); out (2,64) float32",
    [arr("partials", F32, (128, 64)), arr("sums", F32, (128,)),
     arr("maxs", F32, (128,))],
    [F32], [("Dim", 64), ("Blocks", 64)],
    factory="_fused_mse_decode_2pass_2_kernel", args=[]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:2800",
    "fused kv quantize: key/vectors (1,64) float16, rotations (64,64) float32, "
    "midpoints (15) float32; Dim=64,KPackedWidth=VPackedWidth=2; grid=(64,2,1), "
    "threadgroup=(64,1,1); out key/val norms (1) float16, key/val packed (1,2) uint32",
    [arr("key_vectors", F16, (1, 64)), arr("val_vectors", F16, (1, 64)),
     arr("key_rotation", F32, (64, 64)), arr("val_rotation", F32, (64, 64)),
     arr("key_midpoints", F32, (15,)), arr("val_midpoints", F32, (15,))],
    [F16, U32, F16, U32],
    [("Dim", 64), ("KPackedWidth", 2), ("VPackedWidth", 2)],
    factory="_fused_kv_quantize_kernel", args=[4, 4]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:2889",
    "no-rotation quantize: rotated (1,64) float16, midpoints (15) float32; "
    "Dim=64,PackedWidth=2; grid=(64,1,1), threadgroup=(64,1,1); out (1,2) uint32",
    [arr("rotated", F16, (1, 64)), arr("midpoints", F32, (15,))],
    [U32], [("Dim", 64), ("PackedWidth", 2)],
    factory="_fused_norot_quantize_kernel", args=[4]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:2981",
    "fused mse quantize: flat (1,64) float16, rotation (64,64) float32, midpoints (15) "
    "float32; Dim=64,DimsPerLane=2,PackedWidth=8; grid=(64,1,1), threadgroup=(64,1,1); "
    "out norms (1) float16, packed (1,8) uint32",
    [arr("flat", F16, (1, 64)), arr("rot_input", F32, (64, 64)),
     arr("midpoints", F32, (15,))],
    [F16, U32],
    [("Dim", 64), ("DimsPerLane", 2), ("PackedWidth", 8)],
    factory="_fused_mse_quantize_kernel", args=[4, False]))
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:3122",
    "fused prod quantize: flat (1,64) float16, rotation (64,64) float32, midpoints (15) "
    "float32, codebook (16) float32, combined proj (64,64) float32; Dim=64,DimsPerLane=2,"
    "MsePackedWidth=8,SignPackedWidth=2; grid=(32,1,1), threadgroup=(32,1,1); "
    "out norms/res_norms (1) float16, mse_packed (1,8) uint32, signs (1,2) uint32",
    [arr("flat", F16, (1, 64)), arr("rot_input", F32, (64, 64)),
     arr("midpoints", F32, (15,)), arr("codebook", F32, (16,)),
     arr("combined_proj_t", F32, (64, 64))],
    [F16, U32, F16, U32],
    [("Dim", 64), ("DimsPerLane", 2), ("MsePackedWidth", 8), ("SignPackedWidth", 2)],
    factory="_fused_prod_quantize_kernel", args=[4, False]))
SPLIT_IN = [arr("q_rot_low", F16, (1, 1, 1, 64)), arr("q_proj_low", F16, (1, 1, 1, 64)),
            arr("q_rot_high", F16, (1, 1, 1, 64)), arr("q_proj_high", F16, (1, 1, 1, 64)),
            arr("key_low_norms", F16, (1, 1, 2)), arr("key_low_mse", U32, (1, 1, 2, 8)),
            arr("key_low_res_norms", F16, (1, 1, 2)), arr("key_low_signs", U32, (1, 1, 2, 2)),
            arr("key_high_norms", F16, (1, 1, 2)), arr("key_high_mse", U32, (1, 1, 2, 8)),
            arr("key_high_res_norms", F16, (1, 1, 2)), arr("key_high_signs", U32, (1, 1, 2, 2)),
            arr("val_low_norms", F16, (1, 1, 2)), arr("val_low_indices", U32, (1, 1, 2, 8)),
            arr("val_high_norms", F16, (1, 1, 2)), arr("val_high_indices", U32, (1, 1, 2, 8)),
            arr("key_low_codebook", F32, (16,)), arr("key_high_codebook", F32, (16,)),
            arr("key_low_scale", F32, (1,)), arr("key_high_scale", F32, (1,)),
            arr("val_low_codebook", F32, (16,)), arr("val_high_codebook", F32, (16,))]
S.append(spec("tq", "factory", None, "mlx_vlm/turboquant.py:3346",
    "split decode (4,4): q_rot/q_proj low+high (1,1,1,64) float16, key low/high norms "
    "and residual norms (1,1,2) float16, mse/indices (1,1,2,8) uint32, signs (1,1,2,2) "
    "uint32, codebooks (16) float32, scales (1) float32; DimLow=DimHigh=64,"
    "RepeatCount=1,TokTileSize=1024, packed widths 8/2; grid=(32,2,1), "
    "threadgroup=(32,1,1); out_acc (1,1,1,64) float32, out_sum/out_max (1,1,1) float32",
    SPLIT_IN, [F32, F32, F32],
    [("DimLow", 64), ("DimHigh", 64), ("RepeatCount", 1), ("TokTileSize", 1024),
     ("KLMsePackedWidth", 8), ("KLSignPackedWidth", 2), ("KHMsePackedWidth", 8),
     ("KHSignPackedWidth", 2), ("VLPackedWidth", 8), ("VHPackedWidth", 8),
     ("VLBits", 4), ("VHBits", 4)],
    factory="_fused_split_decode_kernel", args=[4, 4, 1]))

out_dir = SPEC_DIR
out_dir.mkdir(exist_ok=True)
for s in S:
    stem = s.get("expect_name") or s.get("factory", "").split(".")[-1].lstrip("_")
    p = out_dir / f"{stem}.json"
    p.write_text(json.dumps(s, indent=1) + "\n")
    print(p.name)
print(f"{len(S)} specs", file=sys.stderr)
