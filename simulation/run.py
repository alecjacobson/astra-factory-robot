import argparse, json, time
from pathlib import Path
import numpy as np
import warp as wp
from .model import load_headliner
from .reduction import build_basis
from .dynamics import Integrator


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
    args = p.parse_args()
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
        )
        graph = sim.capture(args.steps, args.stride)
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
                    frames=simulator.frames.numpy()[: completed // args.stride + 1],
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
        del graph, sim


if __name__ == "__main__":
    main()
