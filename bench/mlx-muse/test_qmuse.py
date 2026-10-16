"""The float32 references in qmuse.py must agree with mlx's own composed ops, or a kernel could pass against a wrong reference.
Needs an mlx install (CPU is enough); skipped without one."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:
    import mlx.core as mx
except ImportError:
    mx = None


@unittest.skipIf(mx is None, "mlx is not installed")
class ReferenceTest(unittest.TestCase):
    def setUp(self):
        import qmuse
        mx.set_default_device(mx.cpu)
        import numpy as np
        qmuse.RNG[0] = np.random.default_rng(7)
        self.qmuse = qmuse
        self.cases = qmuse.make_cases(mx, None, None)

    def ref(self, cid, t):
        return self.cases[cid](t)[1]()

    def close(self, got, want, tol):
        self.assertLessEqual(self.qmuse.row_error(got.astype(mx.float32), want.astype(mx.float32)), tol)

    def test_qk_norm_rope_matches_rms_norm_then_fast_rope(self):
        # the kernel's contract (mlx_lm/kernels/qk_norm_rope.py): qk_norm, then mx.fast.rope(traditional=True)
        t = 16
        _, ref, ins = self.cases["qk_norm_rope_D128_HQ32_HK8_os0"](t)
        (q, _), (k, _) = ref()
        qkvg = ins[0]
        hq, hk, hd = self.qmuse.HQ, self.qmuse.HK, self.qmuse.HD
        x = qkvg[..., : (hq + hk) * hd].astype(mx.float32).reshape(1, t, hq + hk, hd)
        y = mx.fast.rms_norm(x, mx.ones((hd,)), 1e-6)
        y_q = y[..., :hq, :] * (43.784 / hd ** 0.5)
        want_q = mx.fast.rope(y_q.transpose(0, 2, 1, 3), hd, traditional=True, base=1e6, scale=1.0, offset=0)
        want_k = mx.fast.rope(y[..., hq:, :].transpose(0, 2, 1, 3), hd, traditional=True, base=1e6, scale=1.0, offset=0)
        self.close(q.transpose(0, 2, 1, 3), want_q, 3e-2)
        self.close(k.transpose(0, 2, 1, 3), want_k, 3e-2)

    def test_scaled_rms_matches_fast_rms_norm_with_centered_weight(self):
        mx.random.seed(7)
        run, ref, ins = self.cases["omlx_scaled_rms_AXIS4096"](4)
        (got, _), = ref()
        x, w = ins
        want = mx.fast.rms_norm(x.astype(mx.float32), 1.0 + w.astype(mx.float32), 1e-6)
        self.close(got, want, 2e-2)

    def test_gate_matches_rms_norm_times_sigmoid(self):
        mx.random.seed(7)
        run, ref, ins = self.cases["rms_norm_sigmoid_gate_D128"](4)
        (got, _), = ref()
        x, gate = ins
        xn = mx.fast.rms_norm(x.astype(mx.float32).transpose(0, 2, 1, 3), mx.ones((self.qmuse.HD,)), 1e-6)
        want = (xn * mx.sigmoid(gate.astype(mx.float32).reshape(1, 4, self.qmuse.HQ, self.qmuse.HD))).reshape(
            1, 4, self.qmuse.HQ * self.qmuse.HD)
        self.close(got, want, 2e-2)

    def test_routing_returns_distinct_experts_and_normalised_weights(self):
        (inds, _), (w, _) = self.ref("expert_routing_E256_K8", 16)
        for row in inds.astype(mx.int32).tolist():
            self.assertEqual(len(set(row)), self.qmuse.K)
        # weights sum to route_scale (2.5) up to the epsilon
        self.assertLess(float(mx.abs(w.sum(-1) - 2.5).max()), 1e-4)


@unittest.skipIf(mx is None, "mlx is not installed")
class ExitCodeTest(unittest.TestCase):
    """qmuse.main returns 0 only when every case is ok: a MISMATCH or an ERROR row must not exit 0."""

    def run_main(self, builder, extra=()):
        import tempfile
        from unittest import mock

        import qmuse
        mx.set_default_device(mx.cpu)
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp)
            (src / "mlxlm").mkdir()
            (src / "mlxlm" / "__init__.py").write_text("")
            (src / "omlx_norm_kernels.py").write_text("")
            argv = ["qmuse.py", "--src", str(src), "--route", "native", "--reps", "1", "--batch", "1", "--tokens", "1",
                    *extra]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(qmuse, "make_cases", lambda *_: {"c": builder}):
                return qmuse.main()

    def test_ok_returns_0(self):
        one = mx.ones((2, 4))
        self.assertEqual(self.run_main(lambda _: (lambda: [one], lambda: [(one, 1e-4)], [one])), 0)

    def test_mismatch_returns_1(self):
        one = mx.ones((2, 4))
        self.assertEqual(self.run_main(lambda _: (lambda: [mx.zeros((2, 4))], lambda: [(one, 1e-4)], [one])), 1)

    def test_error_returns_1(self):
        def builder(_):
            raise RuntimeError("a kernel this device cannot run")
        self.assertEqual(self.run_main(builder), 1)

    def test_no_case_ran_returns_1(self):
        self.assertEqual(self.run_main(lambda _: None, extra=("--cases", "no_such_case")), 1)


if __name__ == "__main__":
    unittest.main()
