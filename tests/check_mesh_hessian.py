"""Consistent assembled tangent at a saved deformed headliner checkpoint."""

import json, sys
from pathlib import Path
import numpy as np
import warp as wp
from simulation.model import load_headliner
from simulation import bac_hessian

wp.init()
s, _ = load_headliner(int(sys.argv[2]))
data = np.load(Path(sys.argv[1]) / "rom.npz")
j = len(data["positions"]) // 2
state = np.r_[data["positions"][j].ravel(), data["directors"][j]]
x = wp.array(state, dtype=wp.float64)
rng = np.random.default_rng(71)
v = rng.normal(size=s.nd)
v[np.array(s.fixed_host).astype(bool)] = 0
v /= np.linalg.norm(v)
h = wp.zeros((s.nf, 21, 21), dtype=wp.float64)
out = wp.zeros(s.nd, dtype=wp.float64)
wp.launch(
    bac_hessian.evaluate,
    h.shape,
    [x, s.ids, s.signs, s.interior, s.rest, s.young, s.nu, s.thickness, s.fixed, h],
)
wp.launch(
    bac_hessian.product, (s.nf, 21), [s.ids, h, wp.array(v, dtype=wp.float64), out]
)
hv = out.numpy()
gs = []
for sign in [-1, 1]:
    q = wp.array(state + sign * 1e-7 * v, dtype=wp.float64)
    s.evaluate(q)
    out.zero_()
    s.gradient(out)
    gs.append(out.numpy())
fd = (gs[1] - gs[0]) / 2e-7
err = float(np.linalg.norm(fd - hv) / np.linalg.norm(fd))
assert err < 1e-6, err
r = {"assembled_deformed_hessian_relative_error": err}
Path("validation/headliner-hessian.json").write_text(json.dumps(r, indent=2))
print(r)
