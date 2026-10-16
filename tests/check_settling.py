"""The captured settling gate must pass before any transfer step executes."""

import json
from pathlib import Path
import warp as wp
from simulation.model import Shell
from simulation.dynamics import Integrator, prediction
from simulation.settling import Settlement, capture_transfer
from simulation.audit import graph_audit

wp.init()
s = Shell(
    [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0.01], [0.1, 0.1, 0.01]],
    [[0, 1, 2], [1, 3, 2]],
    patches={"1": [0, 0, 0]},
    patch_face_owner=[1, 0],
)
h = s.stiffness()
# Even past the normal robot start time, settling has no carrier acceleration.
target = wp.zeros_like(s.x0)
old = wp.zeros_like(s.x0)
velocity = wp.zeros_like(s.x0)
clock = wp.array([100], dtype=wp.int32)
wp.launch(prediction, s.nd, [s.x0, velocity, target, old, clock, 0.01, s.nv, 1, 0.0])
expected = s.x0.numpy()
expected[2 : 3 * s.nv : 3] -= 9.81 * 0.01**2
assert abs(target.numpy() - expected).max() < 1e-14
reports = {}
for label, cap in [("insufficient", 5), ("settled", 100)]:
    settle = Integrator(
        s,
        h,
        dt=0.05,
        damping=20,
        stationary=True,
        gravity_ramp=2,
        max_steps=cap,
        newton_limit=100,
    )
    settle.settlement = Settlement(settle)
    motion = Integrator(s, h, dt=0.01, max_steps=2, newton_limit=100, gravity_ramp=0)
    graph, children = capture_transfer(settle, motion, cap, 2, 1)
    wp.capture_launch(graph)
    wp.synchronize()
    success = int(settle.settlement.settled.numpy()[0])
    steps = int(motion.clock.numpy()[0])
    assert success == int(label == "settled"), (label, success)
    assert steps == (2 if success else 0), (label, steps)
    assert settle.step_failed.numpy()[0] == motion.step_failed.numpy()[0] == 0
    if success:
        assert abs(motion.frames.numpy()[0] - settle.x.numpy()).max() == 0
    if success:
        initial = motion.frames.numpy()[0].copy()
        wp.capture_launch(graph)
        wp.synchronize()
        assert int(motion.clock.numpy()[0]) == 2
        assert abs(motion.frames.numpy()[0] - initial).max() < 1e-9
    path = Path(f"build/settling-test-{label}.dot")
    wp.capture_debug_dot_print(graph, str(path), verbose=True)
    reports[label] = dict(
        settling_steps=int(settle.clock.numpy()[0]),
        transfer_steps=steps,
        settled=bool(success),
        graph=graph_audit(path),
    )
Path("validation/settling-gate.json").write_text(json.dumps(reports, indent=2) + "\n")
print(reports)
