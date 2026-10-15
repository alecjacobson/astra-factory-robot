import json
from pathlib import Path
import numpy as np
import warp as wp
from simulation import bac_hessian as H
from simulation.model import Shell

wp.init()
a = iter(Path("tests/libshell_bac_golden.txt").read_text().split())
n, ne, nf = [int(next(a)) for _ in range(3)]
x0 = [[float(next(a)) for _ in range(3)] for _ in range(n)]
f = [[int(next(a)) for _ in range(3)] for _ in range(nf)]
x = [float(next(a)) for _ in range(3 * n + ne)]
e, nu, t = [float(next(a)) for _ in range(3)]
s = Shell(x0, f, young=e, nu=nu, thickness=t)
q = wp.array(x, dtype=wp.float64)
h = wp.zeros((nf, 21, 21), dtype=wp.float64)
wp.launch(
    H.evaluate,
    (nf, 21, 21),
    [q, s.ids, s.signs, s.interior, s.rest, e, nu, t, s.fixed, h],
)
d = np.random.default_rng(91).normal(size=s.nd)
d /= np.linalg.norm(d)
v = wp.array(d, dtype=wp.float64)
hp = wp.zeros_like(v)
wp.launch(H.product, (nf, 21), [s.ids, h, v, hp])
g = wp.zeros_like(v)
gs = []
for sign in [-1, 1]:
    s.evaluate(wp.array(np.array(x) + sign * 1.0e-7 * d, dtype=wp.float64))
    g.zero_()
    s.gradient(g)
    gs.append(g.numpy())
fd = (gs[1] - gs[0]) / 2.0e-7
report = dict(
    deformed_hessian_directional_relative_error=float(
        np.linalg.norm(fd - hp.numpy()) / np.linalg.norm(fd)
    ),
    local_symmetry_error=float(np.max(abs(h.numpy() - h.numpy().transpose(0, 2, 1)))),
)
print(report)
assert report["deformed_hessian_directional_relative_error"] < 1.0e-7
assert report["local_symmetry_error"] < 1.0e-8
Path("validation/bac-consistent-hessian.json").write_text(json.dumps(report, indent=2))
