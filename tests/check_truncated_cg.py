"""Captured SPD solve and negative-curvature truncation."""

import json
from pathlib import Path
import numpy as np
import warp as wp
import warp.sparse as ws
from warp.optim import linear
from simulation.truncated_cg import TruncatedCG

wp.init()
rows = wp.array([0, 1, 2], dtype=int)
values = wp.array([2.0, 3.0, 4.0], dtype=wp.float64)
a = ws.bsr_from_triplets(3, 3, rows, rows, values)
op = linear.aslinearoperator(a)
identity = ws.bsr_from_triplets(3, 3, rows, rows, wp.ones(3, dtype=wp.float64))
m = linear.aslinearoperator(identity)
b = wp.array([1.0, 2.0, 3.0], dtype=wp.float64)
x = wp.zeros_like(b)
cg = TruncatedCG(op, b, x, m)
with wp.ScopedCapture() as warm:
    cg()
wp.capture_launch(warm.graph)
wp.synchronize()
with wp.ScopedCapture() as cap:
    cg()
wp.capture_launch(cap.graph)
wp.synchronize()
err = float(np.max(abs(x.numpy() - np.array([0.5, 2 / 3, 0.75]))))
assert err < 1.0e-12
# Reuse the SPD preconditioner, but make the mechanical operator negative definite.
wp.copy(a.values, wp.array([-2.0, -3.0, -4.0], dtype=wp.float64))
wp.capture_launch(cap.graph)
wp.synchronize()
assert cg.negative.numpy()[0] == 1 and cg.it.numpy()[0] == 0
assert np.array_equal(x.numpy(), np.zeros(3))
report = {
    "spd_error": err,
    "negative_curvature_detected": True,
    "first_iteration_truncated": True,
}
Path("validation/truncated-cg.json").write_text(json.dumps(report, indent=2))
print(report)
