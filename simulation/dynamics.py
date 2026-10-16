"""Implicit Euler, consistent Newton tangents, Armijo and PCG entirely in Warp.

cuDSS is used ONLY in setup. Captured integration has no host numerics,
callbacks, host transfers, convergence readback, or allocations.
"""

import warp as wp
import warp.sparse as ws
from warp.optim import linear
from warpack import kernels as K
from warpack.linalg_tiles import gram_tiled
from . import bac_hessian
from .device_refit import DeviceRefit
from .truncated_cg import TruncatedCG


@wp.func
def carrier(t: wp.float64):
    phase = t / wp.float64(4)
    cycle = phase - wp.floor(phase / wp.float64(2)) * wp.float64(2)
    u = cycle - wp.floor(cycle)
    u = wp.clamp(
        (u - wp.float64(0.12)) / wp.float64(0.76), wp.float64(0), wp.float64(1)
    )
    s = u * u * u * (wp.float64(10) + u * (wp.float64(-15) + wp.float64(6) * u))
    if cycle >= wp.float64(1):
        s = wp.float64(1) - s
    return wp.vec3d(
        wp.float64(-1.9975) + wp.float64(3.995) * s,
        wp.float64(-0.2975),
        wp.float64(0.9775)
        + wp.float64(1.3175) * wp.sin(wp.float64(3.141592653589793) * s),
    )


@wp.kernel
def prediction(
    x: wp.array[wp.float64],
    v: wp.array[wp.float64],
    target: wp.array[wp.float64],
    old: wp.array[wp.float64],
    clock: wp.array[wp.int32],
    dt: wp.float64,
    nv: int,
    stationary: int,
    gravity_ramp: wp.float64,
):
    i = wp.tid()
    t = wp.float64(clock[0]) * dt
    delta = carrier(t + dt) - wp.float64(2) * carrier(t) + carrier(t - dt)
    if stationary != 0:
        delta = wp.vec3d(0.0, 0.0, 0.0)
    # A zero ramp means gravity is already fully applied after equilibration.
    u = wp.float64(1)
    if gravity_ramp > wp.float64(0):
        u = wp.clamp((t + dt) / gravity_ramp, wp.float64(0), wp.float64(1))
    ramp = u * u * (wp.float64(3) - wp.float64(2) * u)
    y = x[i] + dt * v[i]
    if i < 3 * nv:
        y -= delta[i % 3]
        if i % 3 == 2:
            y -= wp.float64(9.81) * dt * dt * ramp
    target[i] = y
    old[i] = x[i]


@wp.kernel
def diagonal(
    m: wp.array[wp.float64],
    fixed: wp.array[wp.int32],
    dt: wp.float64,
    damping: wp.float64,
    rows: wp.array[wp.int32],
    vals: wp.array[wp.float64],
):
    i = wp.tid()
    rows[i] = i
    v = m[i] * (wp.float64(1) / (dt * dt) + damping / dt) + wp.float64(1.0e-12)
    if fixed[i] != 0:
        v = wp.float64(1)
    vals[i] = v


@wp.kernel
def inertia_gradient(
    x: wp.array[wp.float64],
    target: wp.array[wp.float64],
    old: wp.array[wp.float64],
    mass: wp.array[wp.float64],
    fixed: wp.array[wp.int32],
    dt: wp.float64,
    damping: wp.float64,
    g: wp.array[wp.float64],
):
    i = wp.tid()
    if fixed[i] != 0:
        g[i] = wp.float64(0)
    else:
        g[i] += mass[i] * (
            (x[i] - target[i]) / (dt * dt) + damping * (x[i] - old[i]) / dt
        )


@wp.kernel
def inertia_energy(
    x: wp.array[wp.float64],
    target: wp.array[wp.float64],
    old: wp.array[wp.float64],
    mass: wp.array[wp.float64],
    fixed: wp.array[wp.int32],
    dt: wp.float64,
    damping: wp.float64,
    e: wp.array[wp.float64],
):
    i = wp.tid()
    if fixed[i] == 0:
        a = x[i] - target[i]
        b = x[i] - old[i]
        wp.atomic_add(
            e, 0, mass[i] * (a * a / (dt * dt) + damping * b * b / dt) / wp.float64(2)
        )


@wp.kernel
def add_tangent(
    out: wp.array[wp.float64],
    v: wp.array[wp.float64],
    diag: wp.array[wp.float64],
    y: wp.array[wp.float64],
    z: wp.array[wp.float64],
    alpha: wp.float64,
    beta: wp.float64,
):
    i = wp.tid()
    z[i] = alpha * (out[i] + diag[i] * v[i]) + beta * y[i]


@wp.kernel
def combine(
    a: wp.array[wp.float64],
    y: wp.array[wp.float64],
    z: wp.array[wp.float64],
    alpha: wp.float64,
    beta: wp.float64,
):
    i = wp.tid()
    z[i] = alpha * a[i] + beta * y[i]


@wp.kernel
def lift(b: wp.array2d[wp.float64], q: wp.array[wp.float64], x: wp.array[wp.float64]):
    i = wp.tid()
    v = wp.float64(0)
    for j in range(b.shape[0]):
        v += b[j, i] * q[j]
    x[i] = v


@wp.kernel
def project(
    b: wp.array2d[wp.float64], x: wp.array[wp.float64], q: wp.array[wp.float64]
):
    j, chunk = wp.tid()
    a = wp.tile_load(b[j], shape=256, offset=chunk * 256)
    v = wp.tile_load(x, shape=256, offset=chunk * 256)
    s = wp.tile_sum(wp.tile_map(wp.mul, a, v))
    wp.tile_atomic_add(q, s, offset=j)


@wp.kernel
def inverse_columns(l: wp.array2d[wp.float64], inv: wp.array2d[wp.float64]):
    j = wp.tid()
    n = l.shape[0]
    for i in range(n):
        v = wp.float64(0)
        if i == j:
            v = wp.float64(1)
        for k in range(i):
            v -= l[i, k] * inv[k, j]
        inv[i, j] = v / l[i, i]
    for ii in range(n):
        i = n - 1 - ii
        v = inv[i, j]
        for k in range(i + 1, n):
            v -= l[k, i] * inv[k, j]
        inv[i, j] = v / l[i, i]


@wp.kernel
def dense_mv(
    a: wp.array2d[wp.float64],
    x: wp.array[wp.float64],
    y: wp.array[wp.float64],
    z: wp.array[wp.float64],
    alpha: wp.float64,
    beta: wp.float64,
):
    i = wp.tid()
    v = wp.float64(0)
    for j in range(a.shape[1]):
        v += a[i, j] * x[j]
    z[i] = alpha * v + beta * y[i]


@wp.kernel
def norm2(a: wp.array[wp.float64], out: wp.array[wp.float64]):
    i = wp.tid()
    wp.atomic_add(out, 0, a[i] * a[i])


@wp.kernel
def dot(a: wp.array[wp.float64], b: wp.array[wp.float64], out: wp.array[wp.float64]):
    i = wp.tid()
    wp.atomic_add(out, 0, a[i] * b[i])


@wp.kernel
def negate(a: wp.array[wp.float64], out: wp.array[wp.float64]):
    i = wp.tid()
    out[i] = -a[i]


@wp.kernel
def init_newton(
    it: wp.array[wp.int32],
    running: wp.array[wp.int32],
    gn: wp.array[wp.float64],
    reference: wp.array[wp.float64],
):
    it[0] = 0
    running[0] = 1
    reference[0] = wp.max(gn[0], wp.float64(1.0e-20))


@wp.kernel
def check_newton(
    it: wp.array[wp.int32],
    running: wp.array[wp.int32],
    gn: wp.array[wp.float64],
    ref: wp.array[wp.float64],
    limit: int,
    tol: wp.float64,
    atol: wp.float64,
):
    it[0] += 1
    running[0] = int(
        it[0] < limit and gn[0] > tol * tol * ref[0] and gn[0] > atol * atol
    )


@wp.kernel
def init_line(
    step: wp.array[wp.float64], it: wp.array[wp.int32], running: wp.array[wp.int32]
):
    step[0] = wp.float64(1)
    it[0] = 0
    running[0] = 1


@wp.kernel
def trial(
    x: wp.array[wp.float64],
    dx: wp.array[wp.float64],
    step: wp.array[wp.float64],
    out: wp.array[wp.float64],
):
    i = wp.tid()
    out[i] = x[i] + step[0] * dx[i]


@wp.kernel
def accept_line(
    base: wp.array[wp.float64],
    candidate: wp.array[wp.float64],
    slope: wp.array[wp.float64],
    step: wp.array[wp.float64],
    it: wp.array[wp.int32],
    running: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    slack: wp.float64,
):
    it[0] += 1
    if candidate[0] <= base[0] + wp.float64(1.0e-4) * step[0] * slope[
        0
    ] + slack * wp.max(wp.float64(1), wp.abs(base[0])):
        running[0] = 0
    else:
        step[0] *= wp.float64(0.5)
        if it[0] >= 20:
            running[0] = 0
            step[0] = wp.float64(0)
            wp.atomic_add(failed, 0, 1)


@wp.kernel
def apply_step(
    x: wp.array[wp.float64], dx: wp.array[wp.float64], step: wp.array[wp.float64]
):
    i = wp.tid()
    x[i] += step[0] * dx[i]


@wp.kernel
def finish_velocity(
    x: wp.array[wp.float64],
    old: wp.array[wp.float64],
    v: wp.array[wp.float64],
    dt: wp.float64,
):
    i = wp.tid()
    v[i] = (x[i] - old[i]) / dt


@wp.kernel
def log_step(
    clock: wp.array[wp.int32],
    iters: wp.array[wp.int32],
    gn: wp.array[wp.float64],
    ref: wp.array[wp.float64],
    history: wp.array2d[wp.float64],
    failed: wp.array[wp.int32],
):
    i = clock[0]
    history[i, 0] = wp.float64(iters[0])
    history[i, 1] = wp.sqrt(gn[0] / ref[0])
    history[i, 2] = wp.sqrt(gn[0])
    history[i, 3] = wp.float64(failed[0])
    clock[0] += 1


@wp.kernel
def save_frame(
    x: wp.array[wp.float64],
    clock: wp.array[wp.int32],
    stride: int,
    frames: wp.array2d[wp.float64],
):
    i = wp.tid()
    step = clock[0]
    if step % stride == 0:
        frames[step // stride, i] = x[i]


@wp.kernel
def check_time(
    clock: wp.array[wp.int32],
    running: wp.array[wp.int32],
    total: wp.array[wp.int32],
    failed: wp.array[wp.int32],
):
    running[0] = int(clock[0] < total[0] and failed[0] == 0)


@wp.kernel
def check_step_failure(
    gn: wp.array[wp.float64],
    ref: wp.array[wp.float64],
    failed: wp.array[wp.int32],
    tol: wp.float64,
    atol: wp.float64,
):
    if not wp.isfinite(gn[0]) or (gn[0] > tol * tol * ref[0] and gn[0] > atol * atol):
        failed[0] = 1


@wp.kernel
def direction_check(
    slope: wp.array[wp.float64], fallback: wp.array[wp.int32], count: wp.array[wp.int32]
):
    fallback[0] = int(not wp.isfinite(slope[0]) or slope[0] >= wp.float64(0))
    if fallback[0] != 0:
        wp.atomic_add(count, 0, 1)


@wp.kernel
def log_linear(
    clock: wp.array[wp.int32],
    it: wp.array[wp.int32],
    res: wp.array[wp.float64],
    history: wp.array2d[wp.float64],
    fallback: wp.array[wp.int32],
):
    i = clock[0]
    history[i, 4] = wp.float64(it[0])
    history[i, 5] = wp.sqrt(res[0])
    history[i, 6] = wp.float64(fallback[0])


class Integrator:
    def __init__(
        self,
        shell,
        h,
        basis=None,
        pre_basis=None,
        dt=1 / 120,
        damping=1.0,
        newton_limit=60,
        tol=1.0e-6,
        absolute_tol=1.0e-5,
        energy_slack=1.0e-12,
        max_steps=960,
        linear_solver="truncated",
        stationary=False,
        gravity_ramp=0.4,
    ):
        self.s = shell
        self.basis = basis
        self.dt = dt
        self.damping = damping
        self.stationary = int(stationary)
        self.gravity_ramp = gravity_ramp
        self.settlement = None
        self.newton_limit = newton_limit
        self.tol = tol
        self.absolute_tol = absolute_tol
        self.energy_slack = energy_slack
        n = shell.nd
        self.n = n
        self.r = n if basis is None else basis.shape[0]
        r = self.r
        dev = shell.device

        def arr(n=n):
            return wp.zeros(n, dtype=wp.float64, device=dev)

        def integer():
            return wp.zeros(1, dtype=wp.int32, device=dev)

        self.x = wp.clone(shell.x0)
        self.v = arr()
        self.old = arr()
        self.target = arr()
        self.g = arr()
        self.dx = arr()
        self.trial = arr()
        self.hv = arr()
        self.lifted = arr()
        self.mapped = arr()
        self.rg = arr(r)
        self.rhs = arr(r)
        self.dq = arr(r)
        self.projected = arr(r)
        self.jv = wp.zeros((shell.nf, 8), dtype=wp.float64)
        self.hessian = wp.zeros((shell.nf, 21, 21), dtype=wp.float64)
        self.energy = arr(1)
        self.candidate = arr(1)
        self.slope = arr(1)
        self.alpha = arr(1)
        self.gnorm = arr(1)
        self.reference = arr(1)
        self.clock = integer()
        self.newton_it = integer()
        self.newton_running = integer()
        self.line_it = integer()
        self.line_running = integer()
        self.failed = integer()
        self.fallback = integer()
        self.fallback_count = integer()
        self.time_running = integer()
        self.step_failed = integer()
        self.history = wp.zeros((max_steps, 7), dtype=wp.float64)
        self.frames = None
        rows = wp.empty(n, dtype=wp.int32)
        self.diag = arr()
        wp.launch(diagonal, n, [shell.mass, shell.fixed, dt, damping, rows, self.diag])
        diag = ws.bsr_from_triplets(n, n, rows, rows, self.diag)
        a = ws.bsr_copy(h)
        ws.bsr_axpy(diag, a)
        a = ws.bsr_copy(a)
        pb = basis if basis is not None else pre_basis
        coarse = None
        if pb is not None:
            nr = pb.shape[0]
            hb = wp.empty_like(pb)
            gram = wp.zeros((nr, nr), dtype=wp.float64)
            l = wp.zeros_like(gram)
            self.inv = wp.zeros_like(gram)
            status = integer()
            op = __import__("warpack").SparseOperator(a)
            op.apply(pb, hb)
            wp.launch_tiled(
                gram_tiled,
                dim=((nr + 7) // 8, (nr + 7) // 8, (n + 255) // 256),
                inputs=[pb, hb, gram],
                block_dim=128,
            )
            wp.launch(K.cholesky, 1, [gram, l, status])
            wp.launch(inverse_columns, nr, [l, self.inv])
            assert status.numpy()[0] == 0
            coarse = arr(nr)
            sol = arr(nr)
            coarse_full = arr()
        if basis is None:
            self.refit = DeviceRefit(a, shell.ids, self.diag)
            fsai = self.refit.inverse
            if pb is None:
                pre = fsai
            else:

                def prec(x, y, z, alpha, beta):
                    fsai.matvec(x, y, z, alpha, beta)
                    coarse.zero_()
                    wp.launch_tiled(
                        project,
                        dim=(nr, (n + 255) // 256),
                        inputs=[pb, x, coarse],
                        block_dim=128,
                    )
                    wp.launch(dense_mv, nr, [self.inv, coarse, coarse, sol, 1.0, 0.0])
                    wp.launch(lift, n, [pb, sol, coarse_full])
                    wp.launch(combine, n, [coarse_full, z, z, alpha, 1.0])

                pre = linear.LinearOperator(
                    (r, r), wp.float64, wp.get_device(dev), prec
                )
        else:

            def prec(x, y, z, alpha, beta):
                wp.launch(dense_mv, r, [self.inv, x, y, z, alpha, beta])

            pre = linear.LinearOperator((r, r), wp.float64, wp.get_device(dev), prec)
        self.pre = pre
        operator = linear.LinearOperator(
            (r, r), wp.float64, wp.get_device(dev), self.matvec
        )
        self.linear_solver = linear_solver
        if linear_solver == "standard":
            self.cg = linear.cg(
                operator,
                self.rhs,
                self.dq,
                M=pre,
                tol=1.0e-4,
                atol=1.0e-10,
                maxiter=2000,
                check_every=0,
                run=False,
            )
        else:
            self.cg = TruncatedCG(operator, self.rhs, self.dq, pre)

    def project(self, x, out):
        out.zero_()
        wp.launch_tiled(
            project,
            dim=(self.r, (self.n + 255) // 256),
            inputs=[self.basis, x, out],
            block_dim=128,
        )

    def matvec(self, x, y, z, alpha, beta):
        full = x
        if self.basis is not None:
            wp.launch(lift, self.n, [self.basis, x, self.lifted])
            full = self.lifted
        self.hv.zero_()
        wp.launch(
            bac_hessian.product,
            (self.s.nf, 21),
            [self.s.ids, self.hessian, full, self.hv],
        )
        if self.basis is None:
            wp.launch(
                add_tangent, self.n, [self.hv, full, self.diag, y, z, alpha, beta]
            )
        else:
            wp.launch(
                add_tangent,
                self.n,
                [self.hv, full, self.diag, full, self.mapped, 1.0, 0.0],
            )
            self.project(self.mapped, self.projected)
            wp.launch(combine, self.r, [self.projected, y, z, alpha, beta])

    def objective(self, x, out):
        out.zero_()
        self.s.energy(x, out)
        wp.launch(
            inertia_energy,
            self.n,
            [
                x,
                self.target,
                self.old,
                self.s.mass,
                self.s.fixed,
                self.dt,
                self.damping,
                out,
            ],
        )

    def gradient(self):
        self.s.evaluate(self.x)
        self.g.zero_()
        self.s.gradient(self.g)
        wp.launch(
            inertia_gradient,
            self.n,
            [
                self.x,
                self.target,
                self.old,
                self.s.mass,
                self.s.fixed,
                self.dt,
                self.damping,
                self.g,
            ],
        )
        if self.basis is None:
            wp.copy(self.rg, self.g)
        else:
            self.project(self.g, self.rg)
        self.gnorm.zero_()
        wp.launch(norm2, self.r, [self.rg, self.gnorm])

    def line_body(self):
        wp.launch(trial, self.n, [self.x, self.dx, self.alpha, self.trial])
        self.objective(self.trial, self.candidate)
        wp.launch(
            accept_line,
            1,
            [
                self.energy,
                self.candidate,
                self.slope,
                self.alpha,
                self.line_it,
                self.line_running,
                self.failed,
                self.energy_slack,
            ],
        )

    def descent_fallback(self):
        self.pre.matvec(self.rhs, self.rhs, self.dq, 1.0, 0.0)
        if self.basis is None:
            wp.copy(self.dx, self.dq)
        else:
            wp.launch(lift, self.n, [self.basis, self.dq, self.dx])
        self.slope.zero_()
        wp.launch(dot, self.n, [self.g, self.dx, self.slope])

    def refresh_hessian(self):
        s = self.s
        wp.launch(
            bac_hessian.evaluate,
            (s.nf, 21, 21),
            [
                self.x,
                s.ids,
                s.signs,
                s.interior,
                s.rest,
                s.young,
                s.nu,
                s.thickness,
                s.fixed,
                self.hessian,
            ],
        )
        if self.basis is None:
            self.refit.update(self.hessian)

    def newton_body(self):
        self.refresh_hessian()
        wp.launch(negate, self.r, [self.rg, self.rhs])
        self.dq.zero_()
        self.cg_result = self.cg()
        if self.basis is None:
            wp.copy(self.dx, self.dq)
        else:
            wp.launch(lift, self.n, [self.basis, self.dq, self.dx])
        self.objective(self.x, self.energy)
        self.slope.zero_()
        wp.launch(dot, self.n, [self.g, self.dx, self.slope])
        wp.launch(direction_check, 1, [self.slope, self.fallback, self.fallback_count])
        wp.capture_if(self.fallback, self.descent_fallback)
        wp.launch(init_line, 1, [self.alpha, self.line_it, self.line_running])
        wp.capture_while(self.line_running, self.line_body)
        wp.launch(apply_step, self.n, [self.x, self.dx, self.alpha])
        self.gradient()
        wp.launch(
            check_newton,
            1,
            [
                self.newton_it,
                self.newton_running,
                self.gnorm,
                self.reference,
                self.newton_limit,
                self.tol,
                self.absolute_tol,
            ],
        )

    def step(self):
        wp.launch(
            prediction,
            self.n,
            [
                self.x,
                self.v,
                self.target,
                self.old,
                self.clock,
                self.dt,
                self.s.nv,
                self.stationary,
                self.gravity_ramp,
            ],
        )
        self.gradient()
        wp.launch(
            init_newton,
            1,
            [self.newton_it, self.newton_running, self.gnorm, self.reference],
        )
        wp.capture_while(self.newton_running, self.newton_body)
        wp.launch(
            check_step_failure,
            1,
            [self.gnorm, self.reference, self.step_failed, self.tol, self.absolute_tol],
        )
        if self.settlement is not None:
            wp.copy(self.settlement.previous_velocity, self.v)
        wp.launch(finish_velocity, self.n, [self.x, self.old, self.v, self.dt])
        wp.launch(
            log_linear,
            1,
            [
                self.clock,
                self.cg_result[0],
                self.cg_result[1],
                self.history,
                self.fallback_count,
            ],
        )
        wp.launch(
            log_step,
            1,
            [
                self.clock,
                self.newton_it,
                self.gnorm,
                self.reference,
                self.history,
                self.failed,
            ],
        )
        if self.frames is not None:
            wp.launch(
                save_frame, self.n, [self.x, self.clock, self.stride, self.frames]
            )
        wp.launch(
            check_time, 1, [self.clock, self.time_running, self.total, self.step_failed]
        )

        if self.settlement is not None:
            self.settlement.check(self)

    def capture(self, steps, stride=10):
        self.total = wp.array([steps], dtype=wp.int32)
        self.stride = stride
        self.frames = wp.zeros((steps // stride + 1, self.n), dtype=wp.float64)
        wp.copy(self.frames[0], self.s.x0)
        # Warm all body kernels/conditional graphs once outside the final capture.
        with wp.ScopedCapture() as warm:
            self.step()
        wp.capture_launch(warm.graph)
        wp.synchronize()
        self.reset()
        with wp.ScopedCapture() as capture:
            wp.launch(
                check_time,
                1,
                [self.clock, self.time_running, self.total, self.step_failed],
            )
            wp.capture_while(self.time_running, self.step)
        return capture.graph

    def reset(self):
        wp.copy(self.x, self.s.x0)
        self.v.zero_()
        self.clock.zero_()
        self.step_failed.zero_()
        self.failed.zero_()
        self.fallback_count.zero_()
        if self.linear_solver == "truncated":
            self.cg.negative_count.zero_()
        if self.basis is None:
            self.refit.counts.zero_()
        self.history.zero_()
        if self.settlement is not None:
            self.settlement.reset()
