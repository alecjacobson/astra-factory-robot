"""Check prescribed support motion against Blender's exported end-effector poses."""

import json, math
from pathlib import Path
import numpy as np

m = json.loads(Path("simulation/headliner_mesh.json").read_text())
errors = []
rotations = []
for pose in m["poses"]:
    t = (pose["frame"] - 1) / 24
    cycle = (t / 4) % 2
    u = max(0, min(1, ((cycle % 1) - 0.12) / 0.76))
    s = u * u * u * (10 + u * (-15 + 6 * u))
    if cycle >= 1:
        s = 1 - s
    p = np.array(
        [-1.9975 + 3.995 * s, -0.2975, 0.9775 + 1.3175 * math.sin(math.pi * s)]
    )
    a = np.array(pose["matrix"])
    errors.append(np.linalg.norm(a[:3, 3] - p))
    rotations.append(np.max(abs(a[:3, :3] - np.eye(3))))
r = {
    "poses_checked": len(errors),
    "max_translation_error_m": float(max(errors)),
    "max_rotation_matrix_error": float(max(rotations)),
}
assert r["max_translation_error_m"] < 5e-6 and r["max_rotation_matrix_error"] < 1e-6, r
Path("validation/carrier-motion.json").write_text(json.dumps(r, indent=2))
print(r)
