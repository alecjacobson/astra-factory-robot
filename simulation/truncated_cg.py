"""Device-only preconditioned Newton-CG, truncated at nonpositive curvature.

The iterate accumulated before negative curvature is a descent direction.
If curvature is negative immediately, the caller uses its SPD-preconditioned
negative gradient. No indefinite CG iteration or host decision is required.
"""

import warp as wp
from warp._src.optim.linear import TiledDot


@wp.kernel
def initialize(
    rr: wp.array[wp.float64],
    tol: wp.array[wp.float64],
    it: wp.array[int],
    running: wp.array[int],
    negative: wp.array[int],
):
    it[0] = 0
    negative[0] = 0
    tol[0] = wp.max(wp.float64(1.0e-20), wp.float64(1.0e-8) * rr[0])
    running[0] = int(rr[0] > tol[0])


@wp.kernel
def curvature(
    pap: wp.array[wp.float64],
    valid: wp.array[int],
    running: wp.array[int],
    negative: wp.array[int],
    total: wp.array[int],
):
    valid[0] = int(wp.isfinite(pap[0]) and pap[0] > wp.float64(0))
    if valid[0] == 0:
        running[0] = 0
        negative[0] = 1
        wp.atomic_add(total, 0, 1)


@wp.kernel
def advance(
    x: wp.array[wp.float64],
    r: wp.array[wp.float64],
    p: wp.array[wp.float64],
    ap: wp.array[wp.float64],
    rz: wp.array[wp.float64],
    pap: wp.array[wp.float64],
):
    i = wp.tid()
    a = rz[0] / pap[0]
    x[i] += a * p[i]
    r[i] -= a * ap[i]


@wp.kernel
def direction(
    z: wp.array[wp.float64],
    p: wp.array[wp.float64],
    rz: wp.array[wp.float64],
    old: wp.array[wp.float64],
):
    i = wp.tid()
    p[i] = z[i] + rz[0] / old[0] * p[i]


@wp.kernel
def check(
    rr: wp.array[wp.float64],
    tol: wp.array[wp.float64],
    it: wp.array[int],
    running: wp.array[int],
):
    it[0] += 1
    running[0] = int(it[0] < 2000 and wp.isfinite(rr[0]) and rr[0] > tol[0])


class TruncatedCG:
    def __init__(self, operator, rhs, solution, preconditioner):
        self.a = operator
        self.b = rhs
        self.x = solution
        self.m = preconditioner
        self.n = rhs.size
        self.r = wp.empty_like(rhs)
        self.z = wp.empty_like(rhs)
        self.p = wp.empty_like(rhs)
        self.ap = wp.empty_like(rhs)
        self.dot = TiledDot(
            max_length=self.n,
            device=rhs.device,
            scalar_type=wp.float64,
            max_column_count=3,
        )
        self.rr = self.dot.col(0)
        self.rz = self.dot.col(1)
        self.pap = self.dot.col(2)
        self.old = wp.zeros(1, dtype=wp.float64)
        self.tol = wp.zeros(1, dtype=wp.float64)
        self.it = wp.zeros(1, dtype=int)
        self.running = wp.zeros(1, dtype=int)
        self.valid = wp.zeros(1, dtype=int)
        self.negative = wp.zeros(1, dtype=int)
        self.negative_count = wp.zeros(1, dtype=int)

    def step_positive(self):
        wp.launch(
            advance, self.n, [self.x, self.r, self.p, self.ap, self.old, self.pap]
        )
        self.m.matvec(self.r, self.z, self.z, 1.0, 0.0)
        self.dot.compute(self.r, self.r, col_offset=0)
        self.dot.compute(self.r, self.z, col_offset=1)
        wp.launch(direction, self.n, [self.z, self.p, self.rz, self.old])
        wp.launch(check, 1, [self.rr, self.tol, self.it, self.running])

    def step(self):
        wp.copy(self.old, self.rz)
        self.a.matvec(self.p, self.ap, self.ap, 1.0, 0.0)
        self.dot.compute(self.p, self.ap, col_offset=2)
        wp.launch(
            curvature,
            1,
            [self.pap, self.valid, self.running, self.negative, self.negative_count],
        )
        wp.capture_if(self.valid, self.step_positive)

    def __call__(self):
        self.x.zero_()
        wp.copy(self.r, self.b)
        self.z.zero_()
        self.ap.zero_()
        self.m.matvec(self.r, self.z, self.z, 1.0, 0.0)
        wp.copy(self.p, self.z)
        self.dot.compute(self.r, self.r, col_offset=0)
        self.dot.compute(self.r, self.z, col_offset=1)
        wp.launch(
            initialize, 1, [self.rr, self.tol, self.it, self.running, self.negative]
        )
        wp.capture_while(self.running, self.step)
        return self.it, self.rr, self.tol
