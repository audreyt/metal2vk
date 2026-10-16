"""qmuse_compare.py must tell agreement from disagreement and refuse dumps that did not see the same inputs."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def dump(path, out, ref, inputs=("aa",), platform="linux", case="c@T1", tol=None, requested=None, drop_ref=False):
    meta = {"platform": platform, "device": "d", "mlx": "x", "route": "r",
            "requested": {case: "ok"} if requested is None else requested}
    arrays = {f"{case}:out0": out, f"{case}:inputs": np.array(json.dumps(list(inputs)))}
    if not drop_ref:
        arrays[f"{case}:ref0"] = ref
    if tol is not None:
        arrays[f"{case}:tol0"] = np.array(tol)
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)


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

    def side(self, out=None, **kwargs):
        """One dump: its output (the reference by default) and the shared reference self.x."""
        return {"out": self.x if out is None else out, "ref": self.x, **kwargs}

    def test_identical_outputs_agree(self):
        code, out = compare(self.side(), self.side())
        self.assertEqual(code, 0)
        self.assertIn("1.0000", out)

    def test_small_difference_agrees_large_disagrees(self):
        self.assertEqual(compare(self.side(), self.side(self.x * (1 + 1e-3)))[0], 0)
        far = self.x.copy()
        far[2, 5] += 1.0
        code, out = compare(self.side(), self.side(far))
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out)

    def test_different_inputs_are_not_comparable(self):
        self.assertEqual(compare(self.side(inputs=("aa",)), self.side(inputs=("bb",)))[0], 2)

    def test_different_case_sets_are_not_comparable(self):
        self.assertEqual(compare(self.side(), self.side(case="other@T1"))[0], 2)

    def test_nan_output_fails(self):
        bad = self.x.copy()
        bad[0, 0] = np.nan
        self.assertEqual(compare(self.side(), self.side(bad))[0], 1)

    def test_missing_file_is_not_comparable(self):
        p = subprocess.run([sys.executable, str(HERE / "qmuse_compare.py"), "/nonexistent/a.npz", "/nonexistent/b.npz"],
                           capture_output=True, text=True, check=False)
        self.assertEqual(p.returncode, 2)

    def test_both_sides_zero_while_references_are_not_must_fail(self):
        zeros = np.zeros_like(self.x)
        code, out = compare(self.side(zeros), self.side(zeros))
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out)

    def test_one_side_off_its_own_reference_fails_even_if_the_other_matches(self):
        self.assertEqual(compare(self.side(), self.side(self.x * 1.1))[0], 1)

    def test_references_that_differ_are_not_comparable(self):
        code, out = compare(self.side(), {"out": self.x, "ref": self.x * 1.5})
        self.assertEqual(code, 2)
        self.assertIn("CPU references", out)

    def test_a_stored_exact_tolerance_makes_one_wrong_index_fail(self):
        idx = np.arange(256, dtype="float32").reshape(1, 256)
        wrong = idx.copy()
        wrong[0, 7] = 8.0
        side = {"ref": idx}
        self.assertEqual(compare({"out": idx, "tol": 0.0, **side}, {"out": wrong, "tol": 0.0, **side})[0], 1)
        self.assertEqual(compare({"out": idx, **side}, {"out": wrong, **side})[0], 0)  # 2e-2 alone hides it

    def test_a_case_that_failed_on_both_platforms_is_not_a_pass(self):
        failed = {"c@T1": "ok", "gone@T1": "ERROR"}
        code, out = compare(self.side(requested=failed), self.side(requested=failed))
        self.assertEqual(code, 2)
        self.assertIn("gone@T1", out)
        self.assertIn("ERROR", out)

    def test_a_dump_without_a_request_list_is_not_comparable(self):
        self.assertEqual(compare(self.side(requested={}), self.side())[0], 2)

    def test_a_requested_case_missing_its_reference_is_incomplete(self):
        code, out = compare(self.side(drop_ref=True), self.side())
        self.assertEqual(code, 2)
        self.assertIn("c@T1", out)


if __name__ == "__main__":
    unittest.main()
