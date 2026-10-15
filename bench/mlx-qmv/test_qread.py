"""qread.py must fail closed: an unmeasured cell, an overlapping round, a dirty route total or a wrong driver never passes."""
import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
BAR = {"m": [2, 4], "n": [10240], "arms": ["qmvf"], "min_gain": 0.10, "max_row_err": 0.01,
       "driver_sha16": "aaaa", "baseline_drift": 0.15}
CLEAN = "[omarchy] metal_kernel route summary: m2v=10 translator=0 fallbacks=0 refusals=0\n"


def row(arm, t, ms, err=0.002):
    return {"arm": arm, "t": t, "ms_median": ms, "ms_q1": ms * 0.98, "ms_q3": ms * 1.02, "row_err_vs_f64": err}


def result(qmvf_ms=(0.3, 0.3)):
    rows = [row("dep", 2, 0.5), row("dep", 4, 0.5), row("qmvf", 2, qmvf_ms[0]), row("qmvf", 4, qmvf_ms[1])]
    return {"k": 5120, "device": "gpu", "mlx": "x", "dep_trace": {"10240": {"2": []}}, "dead": {},
            "driver_libs": {"/usr/lib/a.so": "aaaa"}, "shapes": [{"n": 10240, "rows": rows}]}


def run(res, bar=BAR, route=CLEAN, baseline=None):
    with tempfile.TemporaryDirectory() as tmp:
        t = pathlib.Path(tmp)
        (t / "r.json").write_text(json.dumps(res))
        (t / "b.json").write_text(json.dumps(bar))
        cmd = [sys.executable, str(HERE / "qread.py"), str(t / "r.json"), "--bar", str(t / "b.json")]
        if route is not None:
            (t / "route.err").write_text(route)
            cmd += ["--route", str(t / "route.err")]
        if baseline is not None:
            (t / "s.json").write_text(json.dumps(baseline))
            cmd += ["--baseline", str(t / "s.json")]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return p.returncode, p.stdout


class ReaderTest(unittest.TestCase):
    def test_pass_needs_every_cell(self):
        self.assertEqual(run(result((0.3, 0.3)))[0], 0)
        code, out = run(result((0.3, 0.49)))  # M=4 gains under 10 percent
        self.assertEqual(code, 1)
        self.assertIn("gain short", out)

    def test_unmeasured_cell_fails(self):
        res = result()
        res["shapes"][0]["rows"] = [r for r in res["shapes"][0]["rows"] if not (r["arm"] == "qmvf" and r["t"] == 4)]
        code, out = run(res)
        self.assertEqual(code, 1)
        self.assertIn("not measured", out)

    def test_overlapping_rounds_fail(self):
        res = result()
        res["shapes"][0]["rows"][2]["ms_q3"] = 0.6  # q3 above dep q1
        self.assertEqual(run(res)[0], 1)

    def test_row_error_fails(self):
        res = result()
        res["shapes"][0]["rows"][3]["row_err_vs_f64"] = 0.05
        self.assertEqual(run(res)[0], 1)

    def test_dirty_route_total_is_not_evidence(self):
        dirty = "[omarchy] metal_kernel route summary: m2v=10 translator=0 fallbacks=1 refusals=0\n" \
                "[omarchy]   qmv_fast: m2v=5 translator=0 fallbacks=0 refusals=0\n"
        self.assertEqual(run(result(), route=dirty)[0], 2)

    def test_wrong_driver_is_not_evidence(self):
        res = result()
        res["driver_libs"] = {"/usr/lib/a.so": "bbbb"}
        self.assertEqual(run(res)[0], 2)

    def test_baseline_drift_is_not_evidence(self):
        sealed = result()
        res = copy.deepcopy(result())
        for r in res["shapes"][0]["rows"]:
            if r["arm"] == "dep":
                r["ms_median"] *= 1.3
        self.assertEqual(run(res, baseline=sealed)[0], 2)
        self.assertEqual(run(result(), baseline=sealed)[0], 0)


if __name__ == "__main__":
    unittest.main()
