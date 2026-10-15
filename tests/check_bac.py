"""Independent LibShell fixture, finite differences and rigid-motion invariance."""

import json
from pathlib import Path
import numpy as np
import warp as wp
from simulation.model import Shell

wp.init()
a = iter(Path("tests/libshell_bac_golden.txt").read_text().split())
n, ne, nf = [int(next(a)) for _ in range(3)]
x0 = [[float(next(a)) for _ in range(3)] for _ in range(n)]
faces = [[int(next(a)) for _ in range(3)] for _ in range(nf)]
x = [float(next(a)) for _ in range(3 * n + ne)]
young, nu, t, energy_ref = [float(next(a)) for _ in range(4)]
gref = np.array([float(next(a)) for _ in range(3 * n + ne)])
s = Shell(x0, faces, young=young, nu=nu, thickness=t)
q = wp.array(x, dtype=wp.float64)
out = wp.zeros(1, dtype=wp.float64)
g = wp.zeros(s.nd, dtype=wp.float64)
s.evaluate(q)
s.energy(q, out)
s.gradient(g)
e = float(out.numpy()[0])
gg = g.numpy()
report = dict(
    energy=e,
    reference_energy=energy_ref,
    energy_relative_error=abs(e - energy_ref) / abs(energy_ref),
    gradient_relative_error=float(np.linalg.norm(gg - gref) / np.linalg.norm(gref)),
)
assert report["energy_relative_error"] < 1.0e-11, report
assert report["gradient_relative_error"] < 1.0e-10, report
# curved rest state remains stress free and survives arbitrary rigid motion
s2 = Shell(
    np.array(x[: 3 * n]).reshape(n, 3).tolist(), faces, young=young, nu=nu, thickness=t
)
out.zero_()
g.zero_()
s2.energy(s2.x0, out)
s2.gradient(g)
report["curved_rest_energy"] = float(out.numpy()[0])
report["curved_rest_gradient"] = float(np.linalg.norm(g.numpy()))
assert report["curved_rest_energy"] < 1.0e-20
assert report["curved_rest_gradient"] < 1.0e-10
rot = np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0], [-0.48, 0.64, 0.6]])
trans = np.array(x[: 3 * n]).reshape(n, 3) @ rot.T + np.array([1, 2, 3])
y = np.r_[trans.ravel(), x[3 * n :]]
out.zero_()
s.energy(wp.array(y, dtype=wp.float64), out)
report["rigid_energy_relative_error"] = abs(float(out.numpy()[0]) - e) / abs(e)
assert report["rigid_energy_relative_error"] < 1.0e-11
rng = np.random.default_rng(17)
d = rng.normal(size=s.nd)
d /= np.linalg.norm(d)
eps = 1.0e-6
es = []
for sign in [-1, 1]:
    out.zero_()
    s.energy(wp.array(np.array(x) + sign * eps * d, dtype=wp.float64), out)
    es.append(float(out.numpy()[0]))
fd = (es[1] - es[0]) / (2 * eps)
report["directional_gradient_relative_error"] = abs(fd - np.dot(d, gg)) / max(
    1, abs(fd)
)
assert report["directional_gradient_relative_error"] < 1.0e-7
Path("validation/bac-derivatives.json").write_text(json.dumps(report, indent=2))
print(report)
# At stress-free curved rest, J^T J is the true Hessian, not just an approximation.
from warpack import SparseOperator

s2.evaluate(s2.x0)
h = s2.stiffness()
p = rng.normal(size=s2.nd)
p /= np.linalg.norm(p)
pd = wp.array(p.reshape(1, -1), dtype=wp.float64)
hp = wp.zeros_like(pd)
SparseOperator(h).apply(pd, hp)
gs = []
for sign in [-1, 1]:
    xx = wp.array(s2.x0.numpy() + sign * 1.0e-7 * p, dtype=wp.float64)
    s2.evaluate(xx)
    g.zero_()
    s2.gradient(g)
    gs.append(g.numpy())
fd = (gs[1] - gs[0]) / 2.0e-7
report["rest_hessian_directional_error"] = float(
    np.linalg.norm(fd - hp.numpy()[0]) / np.linalg.norm(fd)
)
assert report["rest_hessian_directional_error"] < 1.0e-7
Path("validation/bac-derivatives.json").write_text(json.dumps(report, indent=2))
print("HESSIAN", report["rest_hessian_directional_error"])
