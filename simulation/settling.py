"""Device-side equilibrium gate and captured settle-then-transfer sequence."""

import math
import warp as wp
from .dynamics import check_time


@wp.kernel
def measure(
    v: wp.array[wp.float64],
    previous: wp.array[wp.float64],
    dt: wp.float64,
    maxima: wp.array[wp.float64],
):
    i = wp.tid()
    speed = wp.vec3d(v[3 * i], v[3 * i + 1], v[3 * i + 2])
    old = wp.vec3d(previous[3 * i], previous[3 * i + 1], previous[3 * i + 2])
    wp.atomic_max(maxima, 0, wp.length(speed))
    wp.atomic_max(maxima, 1, wp.length(speed - old) / dt)


@wp.kernel
def gate(
    clock: wp.array[wp.int32],
    maxima: wp.array[wp.float64],
    quiet: wp.array[wp.int32],
    settled: wp.array[wp.int32],
    running: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    minimum: int,
    hold: int,
    speed_tol: wp.float64,
    acceleration_tol: wp.float64,
    history: wp.array2d[wp.float64],
):
    history[clock[0] - 1, 0] = maxima[0]
    history[clock[0] - 1, 1] = maxima[1]
    if (
        clock[0] > minimum
        and wp.isfinite(maxima[0])
        and wp.isfinite(maxima[1])
        and maxima[0] <= speed_tol
        and maxima[1] <= acceleration_tol
        and failed[0] == 0
    ):
        quiet[0] += 1
    else:
        quiet[0] = 0
    settled[0] = int(quiet[0] >= hold)
    if settled[0] != 0:
        running[0] = 0


class Settlement:
    def __init__(self, sim, ramp=2.0, hold=0.25, speed_tol=1e-5, acceleration_tol=1e-3):
        self.previous_velocity = wp.zeros_like(sim.v)
        self.maxima = wp.zeros(2, dtype=wp.float64)
        self.quiet = wp.zeros(1, dtype=wp.int32)
        self.settled = wp.zeros(1, dtype=wp.int32)
        self.history = wp.zeros((sim.history.shape[0], 2), dtype=wp.float64)
        self.minimum = math.ceil(ramp / sim.dt)
        self.hold = math.ceil(hold / sim.dt)
        self.speed_tol = speed_tol
        self.acceleration_tol = acceleration_tol

    def check(self, sim):
        self.maxima.zero_()
        wp.launch(
            measure, sim.s.nv, [sim.v, self.previous_velocity, sim.dt, self.maxima]
        )
        wp.launch(
            gate,
            1,
            [
                sim.clock,
                self.maxima,
                self.quiet,
                self.settled,
                sim.time_running,
                sim.step_failed,
                self.minimum,
                self.hold,
                self.speed_tol,
                self.acceleration_tol,
                self.history,
            ],
        )

    def reset(self):
        self.previous_velocity.zero_()
        self.maxima.zero_()
        self.quiet.zero_()
        self.settled.zero_()
        self.history.zero_()


def capture_transfer(settler, motion, settle_steps, motion_steps, stride):
    """One graph; transfer is unreachable until the settling gate succeeds."""
    settling_graph = settler.capture(settle_steps, max(1, round(0.25 / settler.dt)))
    motion_graph = motion.capture(motion_steps, stride)

    def transfer():
        wp.copy(motion.x, settler.x)
        wp.copy(motion.v, settler.v)
        wp.copy(motion.frames[0], settler.x)
        wp.launch(
            check_time,
            1,
            [motion.clock, motion.time_running, motion.total, motion.step_failed],
        )
        wp.capture_while(motion.time_running, motion.step)

    with wp.ScopedCapture() as capture:
        settler.reset()
        motion.reset()
        wp.launch(
            check_time,
            1,
            [settler.clock, settler.time_running, settler.total, settler.step_failed],
        )
        wp.capture_while(settler.time_running, settler.step)
        wp.capture_if(settler.settlement.settled, transfer)
    # Retain the preparation graphs through execution as well.
    return capture.graph, (settling_graph, motion_graph)
