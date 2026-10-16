import argparse, json, time
from pathlib import Path
import numpy as np
import warp as wp
from .model import load_headliner
from .reduction import build_basis
from .dynamics import Integrator
from .settling import Settlement, capture_transfer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contacts", type=int, choices=[2, 4], default=4)
    p.add_argument(
        "--full-preconditioner", choices=["fsai", "two-level"], default="fsai"
    )
    p.add_argument("--energy-slack", type=float, default=1.0e-12)
    p.add_argument("--absolute-tol", type=float, default=1.0e-5)
    p.add_argument("--newton-limit", type=int, default=200)
    p.add_argument("--weights", type=int, default=40)
    p.add_argument("--steps", type=int, default=480)
    p.add_argument("--dt", type=float, default=1 / 120)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--model", choices=["full", "rom", "both"], default="both")
    p.add_argument("--output", default="build/dynamics")
    p.add_argument(
        "--linear-solver", choices=["standard", "truncated"], default="truncated"
    )
    p.add_argument("--debug-chunk", type=int, default=0)
    p.add_argument("--young", type=float, default=3.0e8)
    p.add_argument("--enrichment", type=int, default=2)
    p.add_argument("--free-directors", action="store_true")
    p.add_argument("--no-nonlinear-enrichment", action="store_true")
    p.add_argument("--no-settle", action="store_true", help="Legacy stress-free start")
    p.add_argument("--settle-dt", type=float, default=1 / 120)
    p.add_argument("--settle-limit", type=float, default=30.0)
    p.add_argument("--settle-damping", type=float, default=20.0)
    args = p.parse_args()
    if (
        args.settle_dt <= 0
        or args.settle_dt > args.settle_limit
        or args.settle_limit < 2.25
        or args.settle_damping <= 0
    ):
        p.error(
            "positive settling dt/damping and at least 2.25 s settling limit required"
        )
    if not args.no_settle and args.debug_chunk:
        p.error("debug chunks are only available for legacy --no-settle runs")
    if args.steps <= 0 or args.stride <= 0 or args.steps % args.stride or args.dt <= 0:
        p.error("positive steps/dt/stride required; steps must be divisible by stride")
    if (
        args.newton_limit <= 0
        or args.absolute_tol <= 0
        or args.weights <= 0
        or args.enrichment < 0
        or args.energy_slack < 0
    ):
        p.error(
            "positive solver limits/weights/tolerance and nonnegative enrichment required"
        )
    wp.init()
    wp.set_mempool_release_threshold("cuda:0", 0)
    s, data = load_headliner(args.contacts, young=args.young)
    print("MESH", s.nv, s.nf, s.nd, flush=True)
    b, h, modes = build_basis(
        s,
        args.weights,
        enrichment=args.enrichment,
        free_directors=args.free_directors,
        nonlinear=not args.no_nonlinear_enrichment,
    )
    print("MODES", modes, flush=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mesh.json").write_text(json.dumps(data))
    (out / "modes.json").write_text(json.dumps(modes, indent=2))
    for name, basis in [("full", None), ("rom", b)]:
        if args.model not in ("both", name):
            continue
        start = time.perf_counter()
        s.evaluate(s.x0)
        sim = Integrator(
            s,
            h,
            basis,
            pre_basis=b if args.full_preconditioner == "two-level" else None,
            dt=args.dt,
            max_steps=args.steps,
            linear_solver=args.linear_solver,
            newton_limit=args.newton_limit,
            absolute_tol=args.absolute_tol,
            energy_slack=args.energy_slack,
            gravity_ramp=0.4 if args.no_settle else 0.0,
        )
        settler = None
        children = None
        if args.no_settle:
            graph = sim.capture(args.steps, args.stride)
        else:
            settle_steps = round(args.settle_limit / args.settle_dt)
            settler = Integrator(
                s,
                h,
                basis,
                dt=args.settle_dt,
                damping=args.settle_damping,
                max_steps=settle_steps,
                linear_solver="truncated",
                newton_limit=args.newton_limit,
                absolute_tol=args.absolute_tol,
                energy_slack=args.energy_slack,
                stationary=True,
                gravity_ramp=2.0,
            )
            settler.settlement = Settlement(settler)
            graph, children = capture_transfer(
                settler, sim, settle_steps, args.steps, args.stride
            )
        wp.synchronize()
        setup = time.perf_counter() - start
        wp.capture_debug_dot_print(
            graph, str(out / (name + "-graph.dot")), verbose=True
        )
        print("CAPTURED", name, setup, flush=True)

        def ensure_converged(simulator):
            if simulator.step_failed.numpy()[0]:
                completed = int(simulator.clock.numpy()[0])
                np.savez_compressed(
                    out / (name + "-failed.npz"),
                    history=simulator.history.numpy()[:completed],
                    state=simulator.x.numpy(),
                    velocity=simulator.v.numpy(),
                    frames=simulator.frames.numpy()[
                        : completed // simulator.stride + 1
                    ],
                )
                raise RuntimeError(
                    f"{name}: unconverged timestep {completed}; device loop stopped and diagnostic checkpoint saved"
                )

        start = time.perf_counter()
        if args.debug_chunk:
            for limit in range(
                args.debug_chunk, args.steps + args.debug_chunk, args.debug_chunk
            ):
                sim.total.fill_(min(limit, args.steps))
                wp.capture_launch(graph)
                wp.synchronize()
                ensure_converged(sim)
                last = min(limit, args.steps) - 1
                checkpoint = sim.history.numpy()[: last + 1]
                np.savez_compressed(
                    out / (name + "-debug.npz"),
                    history=checkpoint,
                    state=sim.x.numpy(),
                    velocity=sim.v.numpy(),
                )
                print(
                    "DEBUG",
                    name,
                    last + 1,
                    checkpoint[last].tolist(),
                    flush=True,
                )
                assert np.isfinite(checkpoint).all()
                assert not (
                    (checkpoint[:, 1] > 1.0e-6) & (checkpoint[:, 2] > args.absolute_tol)
                ).any(), "Unconverged debug timestep; checkpoint saved"
        else:
            wp.capture_launch(graph)
            wp.synchronize()
        elapsed = time.perf_counter() - start
        settlement_report = None
        if settler is not None:
            ensure_converged(settler)
            sn = int(settler.clock.numpy()[0])
            sh = settler.history.numpy()[:sn]
            sm = settler.settlement.history.numpy()[:sn]
            np.savez_compressed(
                out / (name + "-settling.npz"),
                state=settler.x.numpy(),
                velocity=settler.v.numpy(),
                history=sh,
                measurements=sm,
                frames=settler.frames.numpy()[: sn // settler.stride + 1],
                frame_dt=settler.dt * settler.stride,
            )
            settlement_report = dict(
                converged=bool(settler.settlement.settled.numpy()[0]),
                steps=sn,
                duration_s=sn * settler.dt,
                dt=settler.dt,
                damping_per_s=settler.damping,
                gravity_ramp_s=2.0,
                required_quiet_duration_s=settler.settlement.hold * settler.dt,
                speed_tolerance_m_s=settler.settlement.speed_tol,
                acceleration_tolerance_m_s2=settler.settlement.acceleration_tol,
                final_max_speed_m_s=float(sm[-1, 0]),
                final_max_acceleration_m_s2=float(sm[-1, 1]),
                line_search_failures=int(sh[-1, 3]),
                unconverged_steps=int(
                    ((sh[:, 1] > 1e-6) & (sh[:, 2] > args.absolute_tol)).sum()
                ),
            )
            (out / (name + "-settling.json")).write_text(
                json.dumps(settlement_report, indent=2)
            )
            assert settlement_report["converged"], (
                "Settling limit reached; robot did not move"
            )
        ensure_converged(sim)
        assert int(sim.clock.numpy()[0]) == args.steps
        x = sim.frames.numpy()
        hist = sim.history.numpy()
        assert np.isfinite(x).all() and np.isfinite(hist).all()
        free = 1 - np.array(s.fixed_host)
        mass = s.mass.numpy()
        patch = np.array(s.fixed_host[: 3 * s.nv], dtype=bool)
        report = dict(
            model=name,
            settling=settlement_report,
            simulation_timing_includes_settling=settler is not None,
            preconditioner=args.full_preconditioner
            if name == "full"
            else "reduced-rest-inverse",
            linear_solver=args.linear_solver,
            newton_limit=args.newton_limit,
            absolute_tol=args.absolute_tol,
            energy_slack=args.energy_slack,
            contacts=args.contacts,
            steps=args.steps,
            dt=args.dt,
            stride=args.stride,
            setup_seconds=setup,
            simulation_seconds=elapsed,
            hessian_evaluations=int(hist[:, 0].sum()),
            mean_newton=float(hist[:, 0].mean()),
            max_newton=int(hist[:, 0].max()),
            last_cg_iterations=int(sim.cg_result[0].numpy()[0]),
            preconditioner_refits=sim.refit.counts.numpy().tolist()
            if name == "full"
            else None,
            negative_curvature_truncations=int(sim.cg.negative_count.numpy()[0])
            if args.linear_solver == "truncated"
            else None,
            non_descent_fallbacks=int(sim.fallback_count.numpy()[0]),
            debug_host_checkpoints=bool(args.debug_chunk),
            last_cg_residual=float(sim.cg_result[1].numpy()[0] ** 0.5),
            max_relative_stationarity=float(hist[:, 1].max()),
            max_absolute_stationarity=float(hist[:, 2].max()),
            line_search_failures=int(hist[-1, 3]),
            max_patch_error_m=float(
                np.max(
                    np.abs(x[:, : 3 * s.nv][:, patch] - s.x0.numpy()[: 3 * s.nv][patch])
                )
            ),
            max_displacement_m=float(
                np.linalg.norm(
                    (x[:, : 3 * s.nv] - s.x0.numpy()[: 3 * s.nv]).reshape(
                        len(x), s.nv, 3
                    ),
                    axis=2,
                ).max()
            ),
            mass_kg=float(mass.sum() / 3),
            free_dofs=int(free.sum()),
            material=dict(
                young_Pa=s.young,
                poisson=s.nu,
                thickness_m=s.thickness,
                density_kg_m3=s.density,
            ),
            basis_dofs=sim.r,
        )
        np.savez_compressed(
            out / (name + ".npz"),
            positions=x[:, : 3 * s.nv].reshape(len(x), s.nv, 3),
            directors=x[:, 3 * s.nv :],
            history=hist,
            mass=mass[: 3 * s.nv : 3],
            fixed=np.array(s.fixed_host),
            basis=b.numpy() if name == "rom" else np.zeros(0),
        )
        (out / (name + ".json")).write_text(json.dumps(report, indent=2))
        print("RESULT", report, flush=True)
        del graph, children, sim, settler


if __name__ == "__main__":
    main()
