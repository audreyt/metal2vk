"""Route ConvertValue's pointer conversions through the pointer-width integer.

The four kimi/experts kernels (k3_xp_up_r1/r4, k3_xp_down_r1/r4) load a runtime
table entry of six raw device addresses (`struct ExpertPtrs`, 6 x ptr) and copy
the row by value into thread-private memory; SROA leaves the copy as
mixed-address-space memcpys, which sweep/memcpy_mixed_as.py expands to byte
load/store pairs. ReplacePointerBitcastPass then regroups those bytes into
32-bit <2 x i16> chunks, assembles one <4 x i16> per 8-byte pointer through
ConvertInto/ConvertScalarIntoScalar/BitcastValues, and hands it to
BitcastUtils::ConvertValue with the destination type ptr addrspace(1). The
scalar branch of ConvertValue ends in a plain CreateBitCast, and LLVM has no
bitcast between a vector (or an integer) and a pointer: the module fails the
verifier and clspv aborts with

  Invalid bitcast
    %185 = bitcast <4 x i16> %184 to ptr addrspace(1)
  LLVM ERROR: Broken module found, compilation aborted!

(proven by instrumenting every CreateBitCast site in lib/; the only fired one
is ConvertValue's scalar branch, once per copied pointer). The same blind
CreateBitCast is equally wrong in the pointer-to-value direction.

Fix: in the scalar branch, when exactly one side is a pointer, route through
the integer of the pointer's store size - bitcast the non-pointer side to iN
and inttoptr (value -> pointer), or ptrtoint and bitcast (pointer -> value).
The bits are unchanged, so the produced pointer is exactly the raw address the
bytes encode, which is what --physical-storage-buffers addressing wants: the
table entries ARE device addresses. Same-width pointer-to-pointer bitcasts and
every non-pointer conversion keep the original path. A size mismatch keeps the
original path too (it was an invalid bitcast before and still is; no shape in
this sweep reaches it).

Tested against clspv f2b01dd6 plus the 19 CI-order patches
(apply-coopmat-lowering.py through apply-bool-vector-bitcast.py). The anchor
is the final else of ConvertValue's scalar branch, untouched by those patches.
Idempotent.

Usage: python3 apply-ptr-convert-route.py /path/to/clspv
"""
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
utils = root / "lib" / "BitcastUtils.cpp"

sha = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
).strip()
if not sha.startswith("f2b01dd"):
    print(f"NOTE: clspv HEAD is {sha[:8]} (tested f2b01dd6); anchors may need adjustment",
          file=sys.stderr)

s = utils.read_text()
old = """  } else {
    return builder.CreateBitCast(src, dst_type);
  }

  return nullptr;
}"""
new = """  } else {
    // metal2vk (apply-ptr-convert-route): LLVM has no bitcast between a
    // pointer and a non-pointer type, so a plain CreateBitCast here breaks
    // the module whenever one side is a pointer (vector chunks assembled by
    // ReplacePointerBitcastPass -> ptr addrspace(1), and the reverse). Route
    // those conversions through the pointer-width integer: the bits are
    // unchanged, so the result is exactly the pointer the source bytes (or
    // the source pointer) encode. In --physical-storage-buffers mode that
    // raw address is precisely what the target wants.
    unsigned ptr_bits = DL.getTypeStoreSizeInBits(
        dst_type->isPointerTy() ? dst_type : src_type);
    if (DL.getTypeStoreSizeInBits(src_type) == ptr_bits &&
        DL.getTypeStoreSizeInBits(dst_type) == ptr_bits) {
      if (dst_type->isPointerTy() && !src_type->isPointerTy()) {
        return builder.CreateIntToPtr(
            builder.CreateBitCast(src, builder.getIntNTy(ptr_bits)), dst_type);
      }
      if (src_type->isPointerTy() && !dst_type->isPointerTy()) {
        return builder.CreateBitCast(
            builder.CreatePtrToInt(src, builder.getIntNTy(ptr_bits)),
            dst_type);
      }
    }
    return builder.CreateBitCast(src, dst_type);
  }

  return nullptr;
}"""
old2 = """      // Create a reusable cast to an integer type for this element.
      if (!ele_int_cast || cast<User>(ele_int_cast)->getOperand(0) != ele) {
        ele_int_cast =
            builder.CreateBitCast(ele, IntegerType::get(context, ele_width));
      }"""
new2 = """      // Create a reusable cast to an integer type for this element.
      // metal2vk (apply-ptr-convert-route): a pointer element has no bitcast
      // to an integer in LLVM; ptrtoint is the same bits.
      if (!ele_int_cast || cast<User>(ele_int_cast)->getOperand(0) != ele) {
        if (ele->getType()->isPointerTy()) {
          ele_int_cast = builder.CreatePtrToInt(
              ele, IntegerType::get(context, ele_width));
        } else {
          ele_int_cast = builder.CreateBitCast(
              ele, IntegerType::get(context, ele_width));
        }
      }"""
if new in s and new2 in s:
    print("already patched")
    sys.exit(0)
if s.count(old) != 1:
    raise SystemExit(f"anchor count mismatch: {s.count(old)} (want 1)")
if s.count(old2) != 1:
    raise SystemExit(f"anchor2 count mismatch: {s.count(old2)} (want 1)")
utils.write_text(s.replace(old, new, 1).replace(old2, new2, 1))
print(f"patched (clspv HEAD {sha[:8]})")
