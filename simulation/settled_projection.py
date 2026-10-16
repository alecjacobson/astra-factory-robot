"""Post-run best position fit and director-lift admissibility diagnostic.

NumPy performs a mass projection for validation only. The mechanical energy and
barrier checks are evaluated by the same Warp kernels as the simulation.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import warp as wp
from .model import load_headliner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contacts", type=int, choices=[2, 4], required=True)
    p.add_argument("--folder", required=True)
    args = p.parse_args()
    folder = Path(args.folder)
    full = np.load(folder / "full.npz")
    rom = np.load(folder / "rom.npz")
    report = json.loads((folder / "full.json").read_text())
    assert report.get("settling", {}).get("converged")
    x0 = np.asarray(json.loads((folder / "mesh.json").read_text())["vertices"]).ravel()
    n = len(x0)
    b = rom["basis"]
    u = full["positions"][0].ravel() - x0
    mass = np.repeat(full["mass"], 3)
    q = (u * mass) @ b[:, :n].T
    fit = q @ b
    dx = fit[:n] - u
    full_state = np.concatenate([full["positions"][0].ravel(), full["directors"][0]])
    position_fit = np.concatenate([x0 + fit[:n], full["directors"][0]])
    lifted_fit = np.concatenate([x0 + fit[:n], fit[n:]])
    wp.init()
    shell, _ = load_headliner(args.contacts, young=report["material"]["young_Pa"])
    energies = {}
    for label, state in [
        ("full", full_state),
        ("position_fit_full_directors", position_fit),
        ("position_fit_rom_director_lift", lifted_fit),
    ]:
        output = wp.zeros(1, dtype=wp.float64)
        shell.energy(wp.array(state, dtype=wp.float64), output)
        value = float(output.numpy()[0])
        rejected = value >= 1e99
        energies[label] = dict(
            barrier_rejected=rejected, elastic_energy_J=None if rejected else value
        )
    result = dict(
        contacts=args.contacts,
        state="full gravity-settled equilibrium",
        best_position_fit_mass_RMS_m=float(
            np.sqrt(np.sum(dx**2 * mass) / mass.sum() * 3)
        ),
        best_position_fit_max_m=float(np.linalg.norm(dx.reshape(-1, 3), axis=1).max()),
        director_coordinate_max_difference_rad=float(
            abs(fit[n:] - full["directors"][0]).max()
        ),
        energies=energies,
        interpretation="Position-mass projection only, not an energy minimizer or proof that no other ROM state can fit the reference.",
    )
    Path(f"validation/settled-projection-{args.contacts}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
