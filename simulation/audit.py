"""Post-run audits and plots. NumPy here is validation/I/O, not simulation."""

import argparse, collections, hashlib, json
from pathlib import Path
import numpy as np


def graph_audit(path):
    text = Path(path).read_text()
    report = {
        k: text.count(k)
        for k in [
            "KERNEL",
            "CONDITIONAL",
            "MEMCPY",
            "MEMSET",
            "HOST",
            "HtoD",
            "DtoH",
            "DtoD",
        ]
    }
    report["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    report["accepted"] = (
        report["HOST"] == report["HtoD"] == report["DtoH"] == 0
        and report["CONDITIONAL"] >= 4
        and report["MEMCPY"] == report["DtoD"]
    )
    assert report["accepted"], report
    return report


def audit(folder):
    folder = Path(folder)
    mesh = json.loads((folder / "mesh.json").read_text())
    x = np.array(mesh["vertices"])
    faces = mesh["faces"]
    full = np.load(folder / "full.npz")
    rom = np.load(folder / "rom.npz")
    assert all(
        np.isfinite(a[k]).all()
        for a in [full, rom]
        for k in ["positions", "directors", "history"]
    )
    f = full["positions"]
    r = rom["positions"]
    m = full["mass"]
    df = f - f[0]
    dr = r - r[0]
    motion_error = np.linalg.norm(dr - df, axis=2)
    motion_rms = np.sqrt(np.sum(motion_error**2 * m, axis=1) / m.sum())
    motion_full = np.linalg.norm(df, axis=2)
    motion_rom = np.linalg.norm(dr, axis=2)
    error = np.linalg.norm(r - f, axis=2)
    deform = np.linalg.norm(f - x, axis=2)
    rms = np.sqrt(np.sum(error**2 * m, axis=1) / m.sum())
    frms = np.sqrt(np.sum(deform**2 * m, axis=1) / m.sum())
    freport = json.loads((folder / "full.json").read_text())
    rreport = json.loads((folder / "rom.json").read_text())
    dt = freport["dt"]
    stride = freport["stride"]
    time = np.arange(len(f)) * dt * stride
    e = collections.Counter(
        tuple(sorted((a, b))) for t in faces for a, b in zip(t, t[1:] + t[:1])
    )
    adj = collections.defaultdict(list)
    for (a, b), n in e.items():
        if n == 1:
            adj[a].append(b)
            adj[b].append(a)
    seen = set()
    loops = 0
    for v in adj:
        if v in seen:
            continue
        loops += 1
        todo = [v]
        while todo:
            q = todo.pop()
            if q not in seen:
                seen.add(q)
                todo += adj[q]
    assert loops == 12 and all(n <= 2 for n in e.values())
    report = dict(
        vertices=len(x),
        triangles=len(faces),
        boundary_loops=loops,
        patch_radius_m=0.05,
        patch_polygon_segments=mesh["patch_polygon_segments"],
        patch_max_radial_overreach_m=mesh["patch_max_radial_overreach_m"],
        max_vertex_error_m=float(error.max()),
        max_mass_rms_error_m=float(rms.max()),
        trajectory_relative_mass_L2_error=float(
            np.sqrt(np.sum(error**2 * m) / np.sum(deform**2 * m))
        ),
        peak_full_displacement_m=float(deform.max()),
        motion_peak_full_m=float(motion_full.max()),
        motion_peak_rom_m=float(motion_rom.max()),
        motion_max_mass_rms_error_m=float(motion_rms.max()),
        motion_relative_mass_L2_error=float(
            np.sqrt(np.sum(motion_error**2 * m) / np.sum(motion_full**2 * m))
        ),
        initial_equilibrium_max_vertex_difference_m=float(error[0].max()),
        full=freport,
        rom=rreport,
        graphs={
            model: graph_audit(folder / (model + "-graph.dot"))
            for model in ["full", "rom"]
        },
    )
    for model in ["full", "rom"]:
        h = full["history"] if model == "full" else rom["history"]
        failed = (h[:, 1] > 1.0e-6) & (
            h[:, 2] > report[model].get("absolute_tol", 1.0e-6)
        )
        report[model]["unconverged_steps"] = int(failed.sum())
        report[model]["iteration_cap_steps"] = int(
            (h[:, 0] >= report[model].get("newton_limit", 60)).sum()
        )
        assert report[model]["max_patch_error_m"] < 1.0e-10
        report[model]["accepted"] = not bool(failed.any())
        current_positions = full["positions"] if model == "full" else rom["positions"]
        report[model]["initial_peak_sag_m"] = float(
            np.linalg.norm(current_positions[0] - x, axis=1).max()
        )
        report[model]["initial_quiet_dwell_peak_change_m"] = float(
            np.linalg.norm(
                current_positions[time <= 0.4] - current_positions[0], axis=2
            ).max()
        )
        settling = report[model].get("settling")
        if settling:
            assert settling["converged"] and settling["unconverged_steps"] == 0
            a = np.load(folder / (model + "-settling.npz"))
            quiet = round(settling["required_quiet_duration_s"] / settling["dt"])
            assert np.all(
                a["measurements"][-quiet:, 0] <= settling["speed_tolerance_m_s"]
            )
            assert np.all(
                a["measurements"][-quiet:, 1] <= settling["acceleration_tolerance_m_s2"]
            )
            current = full if model == "full" else rom
            assert (
                np.max(
                    abs(
                        a["state"][: 3 * len(x)].reshape(-1, 3)
                        - current["positions"][0]
                    )
                )
                < 1e-12
            )
        assert report[model]["accepted"], (
            f"{model}: {failed.sum()} unconverged timesteps"
        )
    report["timing_ratio_full_over_rom"] = (
        freport["simulation_seconds"] / rreport["simulation_seconds"]
    )
    (folder / "comparison.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        folder / "comparison_curves.npz",
        time=time,
        rms_error=rms,
        full_rms=frms,
        max_error=error.max(axis=1),
        full_peak=deform.max(axis=1),
        motion_rms_error=motion_rms,
        motion_max_error=motion_error.max(axis=1),
        motion_full_peak=motion_full.max(axis=1),
        motion_rom_peak=motion_rom.max(axis=1),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    ax[0].plot(
        time, motion_full.max(axis=1) * 1000, label="Full BAC shell", color="#087f8c"
    )
    ax[0].plot(
        time,
        motion_rom.max(axis=1) * 1000,
        label="Skinning ROM",
        color="#d78a14",
    )
    ax[0].set(ylabel="Change from initial equilibrium (mm)", xlabel="Transfer time (s)")
    ax[0].legend()
    ax[0].grid(alpha=0.2)
    ax[1].plot(time, motion_rms * 1000, label="Mass-weighted RMS", color="#087f8c")
    ax[1].plot(
        time,
        motion_error.max(axis=1) * 1000,
        label="Maximum vertex error",
        color="#d78a14",
    )
    ax[1].set(ylabel="Motion-induced ROM discrepancy (mm)", xlabel="Transfer time (s)")
    ax[1].legend()
    ax[1].grid(alpha=0.2)
    fig.savefig(folder / "comparison.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("full", "rom")}, indent=2
        )
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    args = p.parse_args()
    audit(args.folder)
