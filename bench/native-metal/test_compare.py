#!/usr/bin/env python3
"""Unit tests for compare.py (native Metal vs translated Linux bench tables)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CASES = [
    "activation_f32", "activation_f16", "softmax_f32", "softmax_f16",
    "tilematmul_f32_256", "tilematmul_f32_1024",
    "rt2x2_256", "rt4x2_256", "rt4x4_256", "rt2x2_1024", "rt4x2_1024", "rt4x4_1024",
]


def native_doc(chip="Apple M2 Max", load=0.1, ok=True, drop=(), gpu_us=100.0):
    rows = {}
    for c in CASES:
        if c in drop:
            continue
        rows[c] = {"us": gpu_us + 1, "gpu_us": gpu_us, "iters": 500,
                   "err": 1e-7, "tol": 1e-5, "ok": ok, "metric": "GFLOPs", "value": 1000.0}
    return {"date": "2026-10-12", "device": chip, "macos": "Version 26.6.2 (Build 25G83)",
            "load": load, "rows": rows}


def linux_doc(tag="g14c", ok=True, drop=(), us=50.0, hand_us=41.1):
    rows = {}
    for c in CASES:
        if c in drop:
            continue
        rows[c] = {"us": us, "err": 1e-7, "tol": 1e-5, "ok": ok,
                   "metric": "GFLOPs", "value": 1000.0, "dtype": "f32", "n": 256,
                   "hand": {"fused": {"us": hand_us, "GFLOPs": 800.0},
                            "unfused": {"us": hand_us * 2, "GFLOPs": 400.0}}}
    return {"device": tag, "device_name": None, "driver": "", "icd": "M2V_ICD",
            "date": "2026-10-12", "runner": "m2v-run", "spvdir": "spv", "rows": rows}


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, doc):
        p = self.dir / name
        p.write_text(json.dumps(doc))
        return str(p)

    def run_compare(self, native, linux, meta=()):
        return subprocess.run(
            [sys.executable, str(Path(__file__).parent / "compare.py"),
             "--native", *native, "--linux", *linux,
             *[a for m in meta for a in ("--meta", m)]],
            capture_output=True, text=True)

    def test_median_ratio_and_header(self):
        nat = [self.write(f"n{i}.json", native_doc(gpu_us=100.0 * (i + 1))) for i in range(3)]
        lin = [self.write(f"l{i}.json", linux_doc(us=50.0 + 25.0 * i)) for i in range(3)]
        r = self.run_compare(nat, lin, meta=["kernel=7.1.12-2-11.38-sep-ARCH",
                                            "driver=69bde15340", "governor=performance", "ane=idle"])
        self.assertEqual(r.returncode, 0, r.stderr)
        # medians: native GPU 200.0 us, translated 75.0 us, ratio 0.38; spreads 200.0 / 50.0
        for want in ["200.0", "75.0", "0.38", "41.1/82.2"]:
            self.assertIn(want, r.stdout)
        self.assertIn("Apple M2 Max", r.stdout)
        self.assertIn("69bde15340", r.stdout)
        self.assertIn("performance", r.stdout)
        self.assertIn("41.1/82.2", r.stdout)          # hand fused/unfused context
        self.assertNotIn("WARNING", r.stdout)          # loads 0.1, quiet
        self.assertIn("| case |", r.stdout)            # markdown table

    def test_chip_mismatch_refused(self):
        nat = self.write("n.json", native_doc(chip="Apple M1 Max"))
        lin = self.write("l.json", linux_doc(tag="g14c"))
        r = self.run_compare([nat], [lin])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("chip", r.stderr.lower())

    def test_missing_case_refused(self):
        nat = self.write("n.json", native_doc())
        lin = self.write("l.json", linux_doc(drop=["rt4x4_1024"]))
        r = self.run_compare([nat], [lin])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("rt4x4_1024", r.stderr)

    def test_failed_check_refused(self):
        nat = self.write("n.json", native_doc(ok=True))
        lin = self.write("l.json", linux_doc(ok=False))
        r = self.run_compare([nat], [lin])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("OK", r.stderr)

    def test_native_missing_load_refused(self):
        doc = native_doc()
        del doc["load"]
        nat = self.write("n.json", doc)
        lin = self.write("l.json", linux_doc())
        r = self.run_compare([nat], [lin])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("load", r.stderr)

    def test_native_missing_chip_refused(self):
        doc = native_doc()
        del doc["device"]
        nat = self.write("n.json", doc)
        lin = self.write("l.json", linux_doc())
        r = self.run_compare([nat], [lin])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("chip", r.stderr.lower())

    def test_load_warning_macos(self):
        nat = self.write("n.json", native_doc(load=0.9))
        lin = self.write("l.json", linux_doc())
        r = self.run_compare([nat], [lin])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("WARNING", r.stdout)
        self.assertIn("0.9", r.stdout)

    def test_load_warning_linux_meta(self):
        nat = self.write("n.json", native_doc())
        lin = self.write("l.json", linux_doc())
        r = self.run_compare([nat], [lin], meta=["load=2.10"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("WARNING", r.stdout)
        self.assertIn("2.1", r.stdout)


if __name__ == "__main__":
    unittest.main()
