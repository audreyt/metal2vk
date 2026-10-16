"""qmuse_compare.py must tell agreement from disagreement and refuse dumps that did not see the same inputs."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def dump(path, out, ref=None, inputs=("aa",), platform="linux", case="c@T1"):
    meta = {"platform": platform, "device": "d", "mlx": "x", "route": "r"}
    np.savez(path, meta=np.array(json.dumps(meta)), **{f"{case}:out0": out, f"{case}:ref0": out if ref is None else ref,
                                                       f"{case}:inputs": np.array(json.dumps(list(inputs)))})


def compare(a_kwargs, b_kwargs, tol="2e-2"):
    with tempfile.TemporaryDirectory() as tmp:
        t = pathlib.Path(tmp)
        dump(t / "a.npz", **a_kwargs)
        dump(t / "b.npz", platform="darwin", **b_kwargs)
        p = subprocess.run([sys.executable, str(HERE / "qmuse_compare.py"), str(t / "a.npz"), str(t / "b.npz"), "--tol", tol],
                           capture_output=True, text=True, check=False)
        return p.returncode, p.stdout


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(1).standard_normal((4, 64)).astype("float32")

    def test_identical_outputs_agree(self):
        code, out = compare({"out": self.x}, {"out": self.x})
        self.assertEqual(code, 0)
        self.assertIn("1.0000", out)

    def test_small_difference_agrees_large_disagrees(self):
        near = self.x * (1 + 1e-3)
        self.assertEqual(compare({"out": self.x}, {"out": near})[0], 0)
        far = self.x.copy()
        far[2, 5] += 1.0
        code, out = compare({"out": self.x}, {"out": far})
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out)

    def test_different_inputs_are_not_comparable(self):
        self.assertEqual(compare({"out": self.x, "inputs": ("aa",)}, {"out": self.x, "inputs": ("bb",)})[0], 2)

    def test_different_case_sets_are_not_comparable(self):
        self.assertEqual(compare({"out": self.x}, {"out": self.x, "case": "other@T1"})[0], 2)

    def test_nan_output_fails(self):
        bad = self.x.copy()
        bad[0, 0] = np.nan
        self.assertEqual(compare({"out": self.x}, {"out": bad})[0], 1)

    def test_missing_file_is_not_comparable(self):
        p = subprocess.run([sys.executable, str(HERE / "qmuse_compare.py"), "/nonexistent/a.npz", "/nonexistent/b.npz"],
                           capture_output=True, text=True, check=False)
        self.assertEqual(p.returncode, 2)


if __name__ == "__main__":
    unittest.main()
