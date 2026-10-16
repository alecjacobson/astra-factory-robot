"""Rest-state Ritz stiffening diagnostic; Warp/cuDSS solves, NumPy reporting.

Eliminate massless directors implicitly through the mixed shifted inverse.
The eigenproblem itself has only free, mass-normalized position coordinates.
This is a same-mesh ROM diagnostic, not the nonlinear EigenFit algorithm.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import warp as wp
import warp.sparse as ws
from warpack import CuDSSInverse, KrylovSchur, SparseOperator
from warpack.linalg_tiles import gram_tiled
from .model import load_headliner
from .reduction import compliance_diag


@wp.kernel
def inject(
    x: wp.array2d[wp.float64],
    free: wp.array[wp.int32],
    mass: wp.array[wp.float64],
    rhs: wp.array2d[wp.float64],
):
    i = wp.tid()
    rhs[0, free[i]] = wp.sqrt(mass[free[i]]) * x[0, i]


@wp.kernel
def extract(
    x: wp.array2d[wp.float64],
    free: wp.array[wp.int32],
    mass: wp.array[wp.float64],
    y: wp.array2d[wp.float64],
):
    i = wp.tid()
    y[0, i] = wp.sqrt(mass[free[i]]) * x[0, free[i]]


@wp.kernel
def dense_entries(
    a: wp.array2d[wp.float64],
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    vals: wp.array[wp.float64],
):
    i, j = wp.tid()
    k = i * a.shape[0] + j
    rows[k] = i
    cols[k] = j
    vals[k] = a[i, j]
    if i == j:
        vals[k] += wp.float64(1)


class CondensedInverse:
    requires_static_outer_capture = True

    def __init__(self, inverse, free, mass):
        self.inverse, self.free, self.mass = inverse, free, mass
        self.n, self.device = free.size, inverse.device
        self.rhs = wp.zeros((1, inverse.n), dtype=wp.float64)
        self.solution = wp.zeros_like(self.rhs)

    def apply(self, x, y):
        for j in range(x.shape[0]):
            self.rhs.zero_()
            wp.launch(inject, self.n, [x[j : j + 1], self.free, self.mass, self.rhs])
            self.inverse.apply(self.rhs, self.solution)
            wp.launch(
                extract, self.n, [self.solution, self.free, self.mass, y[j : j + 1]]
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contacts", type=int, choices=[2, 4], required=True)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--folder")
    args = p.parse_args()
    wp.init()
    s, _ = load_headliner(args.contacts, young=3e8)
    h = s.stiffness()
    folder = Path(args.folder or f"results/dynamics-{args.contacts}")
    b = wp.array(np.load(folder / "rom.npz")["basis"], dtype=wp.float64)
    n, nr = s.nd, b.shape[0]
    free_host = np.flatnonzero(1 - np.asarray(s.fixed_host[: 3 * s.nv]))
    free = wp.array(free_host, dtype=wp.int32)
    rows = wp.empty(n, dtype=wp.int32)
    vals = wp.empty(n, dtype=wp.float64)
    wp.launch(compliance_diag, n, [s.mass, s.fixed, rows, vals])
    shifted = ws.bsr_copy(h)
    ws.bsr_axpy(ws.bsr_from_triplets(n, n, rows, rows, vals), shifted)
    inverse = CuDSSInverse(
        shifted, 1, mtype="general", matching=True, refinement_steps=2
    )
    op = CondensedInverse(inverse, free, s.mass)
    full = KrylovSchur(op, args.modes, ncv=64, which="LA", tol=1e-10)
    fg = full.capture(max_iterations=12)
    wp.capture_launch(fg)
    wp.synchronize()
    print("FULL", full.converged.numpy(), full.residuals.numpy(), flush=True)
    assert int(full.converged.numpy()[0]) == args.modes
    hb = wp.empty_like(b)
    SparseOperator(h).apply(b, hb)
    reduced = wp.zeros((nr, nr), dtype=wp.float64)
    wp.launch_tiled(
        gram_tiled,
        dim=((nr + 7) // 8, (nr + 7) // 8, (n + 255) // 256),
        inputs=[b, hb, reduced],
        block_dim=128,
    )
    rr, cc = wp.empty(nr * nr, dtype=wp.int32), wp.empty(nr * nr, dtype=wp.int32)
    vv = wp.empty(nr * nr, dtype=wp.float64)
    wp.launch(dense_entries, (nr, nr), [reduced, rr, cc, vv])
    kr = ws.bsr_from_triplets(nr, nr, rr, cc, vv)
    ri = CuDSSInverse(kr, 1, mtype="spd", refinement_steps=2)
    rom = KrylovSchur(ri, args.modes, ncv=64, which="LA", tol=1e-10)
    rg = rom.capture(max_iterations=12)
    wp.capture_launch(rg)
    wp.synchronize()
    print("ROM", rom.converged.numpy(), rom.residuals.numpy(), flush=True)
    assert int(rom.converged.numpy()[0]) == args.modes
    # Independent residuals, shape comparisons and the proposed per-mode ratios.
    lf, lr = 1 / full.eigenvalues.numpy() - 1, 1 / rom.eigenvalues.numpy() - 1
    uf = full.eigenvectors.numpy()
    qr = rom.eigenvectors.numpy()
    bn = b.numpy()
    mass = s.mass.numpy()[free_host]
    ur = (qr @ bn[:, free_host]) * np.sqrt(mass)
    mac = (uf @ ur.T) ** 2
    projection = uf @ (bn[:, free_host] * np.sqrt(mass)).T
    projection_loss = np.sqrt(np.maximum(0, 1 - (projection**2).sum(axis=1)))
    residual = np.linalg.norm(qr @ reduced.numpy() - lr[:, None] * qr, axis=1) / lr
    assert residual.max() < 1e-5, residual
    assert np.all(lr >= lf * (1 - 1e-6)), (lf, lr)
    physical = wp.zeros((args.modes, n), dtype=wp.float64)
    for j in range(args.modes):
        op.rhs.zero_()
        wp.launch(inject, op.n, [full.eigenvectors[j : j + 1], free, s.mass, op.rhs])
        inverse.apply(op.rhs, physical[j : j + 1])
    hu = wp.empty_like(physical)
    SparseOperator(h).apply(physical, hu)
    un = physical.numpy()
    hun = hu.numpy()
    mun = un * s.mass.numpy()
    fr = np.linalg.norm(hun - lf[:, None] * mun, axis=1) / (
        np.linalg.norm(hun, axis=1) + lf * np.linalg.norm(mun, axis=1)
    )
    assert fr.max() < 1e-5, fr
    report = dict(
        contacts=args.contacts,
        reference="same-mesh full shell; statically condensed massless directors",
        state="stress-free rest",
        modes=args.modes,
        full_eigenvalues_rad2_s2=lf.tolist(),
        rom_eigenvalues_rad2_s2=lr.tolist(),
        full_frequency_hz=(np.sqrt(lf) / (2 * np.pi)).tolist(),
        rom_frequency_hz=(np.sqrt(lr) / (2 * np.pi)).tolist(),
        proposed_eigenvalue_ratios=(lf / lr).tolist(),
        ordered_mode_MAC=np.diag(mac).tolist(),
        full_mode_mass_projection_loss=projection_loss.tolist(),
        full_inverse_eigen_residuals=full.residuals.numpy().tolist(),
        full_original_eigen_residuals=fr.tolist(),
        rom_original_eigen_residuals=residual.tolist(),
        nonlinear_correction_applied=False,
    )
    out = Path(f"validation/spectral-study-{args.contacts}.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    full.graph = rom.graph = None
    del fg, rg
    wp.synchronize()
    inverse.close()
    ri.close()


if __name__ == "__main__":
    main()
