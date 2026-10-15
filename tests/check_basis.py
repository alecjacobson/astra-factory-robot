"""Independent post-setup checks of the saved affine skinning space."""

import json, sys
from pathlib import Path
import numpy as np

p = Path(sys.argv[1])
a = np.load(p / "rom.npz")
b = a["basis"]
fixed = a["fixed"].astype(bool)
m = np.repeat(a["mass"], 3)
nv = len(a["mass"])
info = json.loads((p / "modes.json").read_text())
nr = info.get("position_coordinates", b.shape[0])
bp = b[:nr, : 3 * nv]
gram = (bp * m) @ bp.T
report = dict(
    position_mass_orthogonality_max=float(np.max(abs(gram - np.eye(nr)))),
    maximum_fixed_entry=float(np.max(abs(b[:, fixed]))),
    position_coordinates=nr,
    total_coordinates=len(b),
)
assert report["position_mass_orthogonality_max"] < 1.0e-7, report
assert report["maximum_fixed_entry"] < 1.0e-10, report
rest = np.array(json.loads((p / "mesh.json").read_text())["vertices"]).ravel()
u = a["positions"].reshape(-1, len(m)) - rest
q = (u * m) @ bp.T
report["position_subspace_error_m"] = float(np.max(abs(q @ bp - u)))
assert report["position_subspace_error_m"] < 1.0e-8
if len(b) == nr:
    report["director_lift_error_rad"] = float(
        np.max(abs(q @ b[:, 3 * nv :] - a["directors"]))
    )
    assert report["director_lift_error_rad"] < 1.0e-7
(p / "basis-audit.json").write_text(json.dumps(report, indent=2))
print(report)
