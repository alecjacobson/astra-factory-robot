"""Captured refit commits valid factors and rejects an indefinite candidate."""

import json
from pathlib import Path
import numpy as np
import warp as wp
import warp.sparse as ws
from simulation.bac import I21
from simulation.device_refit import DeviceRefit

wp.init()
rows = wp.array(np.arange(21, dtype=np.int32))
a = ws.bsr_from_triplets(21, 21, rows, rows, wp.full(21, 2.0, dtype=wp.float64))
r = DeviceRefit(
    a, wp.array([tuple(range(21))], dtype=I21), wp.ones(21, dtype=wp.float64)
)
h = wp.array(np.eye(21)[None], dtype=wp.float64)
with wp.ScopedCapture() as warm:
    r.update(h)
wp.capture_launch(warm.graph)
wp.synchronize()
r.counts.zero_()
with wp.ScopedCapture() as capture:
    r.update(h)
wp.capture_launch(capture.graph)
wp.synchronize()
good = r.inverse.G.values.numpy().copy()
wp.copy(h, wp.array((-3 * np.eye(21))[None], dtype=wp.float64))
wp.capture_launch(capture.graph)
wp.synchronize()
assert np.array_equal(good, r.inverse.G.values.numpy())
assert r.counts.numpy().tolist() == [1, 1]
assert np.isfinite(good).all()
path = Path("build/refit-guard.dot")
wp.capture_debug_dot_print(capture.graph, str(path), verbose=True)
text = path.read_text()
assert not any(x in text for x in ["HtoD", "DtoH", "HOST"])
report = {
    "valid_commits": 1,
    "invalid_rejections": 1,
    "previous_factor_preserved": True,
    "host_nodes_or_transfers": False,
}
Path("validation/device-refit.json").write_text(json.dumps(report, indent=2))
print(report)
