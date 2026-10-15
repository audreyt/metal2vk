"""The T to kernel mapping must follow MLX's own tiling rule, and the compiled MSL must keep the AOT-key contract."""
import hashlib
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import qmv_corpus as corpus


class MappingTest(unittest.TestCase):
    def test_wide_tiles_follow_mlx(self):
        # mlx/backend/metal/quantized.cpp qmv_wide: n_tiles = ceil(M / 5), vecs_per_tg = ceil(M / n_tiles)
        got = [corpus.variant_for("qmvw", t) for t in range(2, 9)]
        self.assertEqual(got, ["qmvw2", "qmvw3", "qmvw4", "qmvw5", "qmvw3", "qmvw4", "qmvw4"])

    def test_every_mapped_variant_exists(self):
        ids = {v["id"] for v in corpus.variants()}
        for kind in ("qmvf", "qmvw", "qmmt", "qmmt16"):
            for t in range(1, 33):
                if kind == "qmvw" and t < 2:
                    continue
                self.assertIn(corpus.variant_for(kind, t), ids)

    def test_grid_covers_every_output_and_row(self):
        for t in range(2, 9):
            v = {x["id"]: x for x in corpus.variants()}[corpus.variant_for("qmvw", t)]
            (gx, gy, _), (tx, ty, _) = corpus.grid("qmvw", v["nv"], t, 10240)
            self.assertGreaterEqual((gx // tx) * v["nv"], t)
            self.assertEqual((gy // ty) * 8, 10240)
        (gx, gy, _), (tx, _, _) = corpus.grid("qmvf", 1, 5, 17408)
        self.assertEqual((gx // tx, gy // 2 * 8), (5, 17408))
        (gx, gy, _), (tx, _, _) = corpus.grid("qmmt", 32, 8, 248320)
        self.assertEqual((gx // tx * 32, gy), (248320, 1))

    def test_row_tiles_cover_the_17_to_32_case(self):
        by_id = {x["id"]: x for x in corpus.variants()}
        for t in range(1, 33):
            for kind, rows in (("qmmt", 32), ("qmmt16", 16)):
                v = by_id[corpus.variant_for(kind, t)]
                (gx, gy, _), (tx, ty, _) = corpus.grid(kind, v["nv"], t, 10240)
                self.assertEqual(v["nv"], rows)
                self.assertEqual((gx // tx) * 32, 10240)
                self.assertGreaterEqual((gy // ty) * rows, t)
                self.assertLess((gy // ty - 1) * rows, t)  # no empty tile row
        self.assertEqual(corpus.grid("qmmt16", 16, 17, 10240)[0][1], 2)
        self.assertEqual(corpus.grid("qmmt", 32, 17, 10240)[0][1], 1)

    def test_wide_tiling_up_to_32_rows(self):
        by_id = {x["id"]: x for x in corpus.variants()}
        for t in range(2, 33):
            tiles = -(-t // 5)
            v = by_id[corpus.variant_for("qmvw", t)]
            self.assertLessEqual(v["nv"], 5)
            self.assertGreaterEqual(tiles * v["nv"], t)
            self.assertEqual(corpus.grid("qmvw", v["nv"], t, 10240)[0][0] // 32, tiles)


class FlattenTest(unittest.TestCase):
    def test_inlines_once_and_drops_what_the_shim_provides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            base = root / "mlx/backend/metal/kernels"
            (base / "steel/gemm").mkdir(parents=True)
            (base / "utils.h").write_text('#pragma once\n#include "mlx/backend/metal/kernels/complex.h"\nint u;\n')
            (base / "complex.h").write_text("int complex_h;\n")
            (base / "steel/gemm/gemm.h").write_text('#include "mlx/backend/metal/kernels/utils.h"\nint g;\n')
            (base / "quantized_utils.h").write_text("int qu;\n")
            (base / "quantized.h").write_text("instantiate_metal_simd_reduction_funcs(bfloat16_t, bfloat16_t, float);\nint q;\n")
            text = corpus.flatten(root)
        self.assertEqual(text.split(), ["int", "u;", "int", "g;", "int", "qu;", "int", "q;"])


class KeyTest(unittest.TestCase):
    def test_entry_name_matches_mlx_naming(self):
        asm = corpus.assemble_module()
        v = {x["id"]: x for x in corpus.variants()}["qmvw4"]
        msl, entry = corpus.msl_for(asm, v, "")
        self.assertEqual(entry, "custom_kernel_qmv_wide__bfloat16_t_64_4_4_8_uint32_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_ts_int32_ts_int32_ts_bfloat16_t")
        self.assertIn(f'[[host_name("{entry}")]]', msl)
        self.assertEqual(len(hashlib.sha256(msl.encode() + b"\0" + entry.encode()).hexdigest()), 64)


if __name__ == "__main__":
    unittest.main()
