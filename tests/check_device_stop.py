"""An unconverged Newton solve stops the entire captured timestep loop."""

import json
from pathlib import Path
import warp as wp
from simulation.model import Shell
from simulation.dynamics import Integrator

wp.init()
s = Shell(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0.1], [1, 1, 0.1]],
    [[0, 1, 2], [1, 3, 2]],
    patches={"1": [0, 0, 0]},
    patch_face_owner=[1, 0],
)
h = s.stiffness()
sim = Integrator(
    s, h, dt=0.1, newton_limit=1, tol=1e-12, absolute_tol=1e-12, max_steps=3
)
graph = sim.capture(3, 1)
wp.capture_launch(graph)
wp.synchronize()
r = {
    "completed_attempts": int(sim.clock.numpy()[0]),
    "device_failure_flag": int(sim.step_failed.numpy()[0]),
    "relative_gradient": float(sim.history.numpy()[0, 1]),
}
assert r["completed_attempts"] == 1 and r["device_failure_flag"] == 1, r
Path("validation/device-stop.json").write_text(json.dumps(r, indent=2))
print(r)
