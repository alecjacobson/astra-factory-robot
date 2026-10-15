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
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    ax[0].plot(time, deform.max(axis=1) * 1000, label="Full BAC shell", color="#087f8c")
    ax[0].plot(
        time,
        np.linalg.norm(r - x, axis=2).max(axis=1) * 1000,
        label="Skinning ROM",
        color="#d78a14",
    )
    ax[0].set(ylabel="Peak displacement (mm)", xlabel="Time (s)")
    ax[0].legend()
    ax[0].grid(alpha=0.2)
    ax[1].plot(time, rms * 1000, label="Mass-weighted RMS", color="#087f8c")
    ax[1].plot(
        time, error.max(axis=1) * 1000, label="Maximum vertex error", color="#d78a14"
    )
    ax[1].set(ylabel="ROM − full discrepancy (mm)", xlabel="Time (s)")
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
