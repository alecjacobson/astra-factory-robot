"""Constrained skinning weights and statically lifted BAC director coordinates.

The weight design uses the position block of the BAC rest Hessian (directors
held at rest). Directors in each affine displacement column are then relaxed
by H_tt D = -H_tx B. The simulation energy is always the complete BAC energy.
This is a declared basis-design choice, not a claim of condensed-shell modes.
"""

import time
import warp as wp
import warp.sparse as ws
from warpack import (
    ConstrainedKrylovSchur,
    CuDSSInverse,
    SparseOperator,
    saddle_point_matrix,
)
from warpack import kernels as K
from warpack.linalg_tiles import gram_tiled


@wp.kernel
def weight_entries(
    off: wp.array[wp.int32],
    col: wp.array[wp.int32],
    val: wp.array[wp.float64],
    nv: int,
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    vals: wp.array[wp.float64],
):
    i = wp.tid()
    for p in range(off[i], off[i + 1]):
        j = col[p]
        rows[p] = 0
        cols[p] = 0
        vals[p] = wp.float64(0)
        if i < 3 * nv and j < 3 * nv and i % 3 == j % 3:
            rows[p] = i // 3
            cols[p] = j // 3
            vals[p] = val[p]


@wp.kernel
def mass_entries(
    m: wp.array[wp.float64], rows: wp.array[wp.int32], vals: wp.array[wp.float64]
):
    i = wp.tid()
    rows[i] = i
    vals[i] = m[3 * i]


@wp.kernel
def affine_basis(
    w: wp.array2d[wp.float64],
    x: wp.array[wp.float64],
    fixed: wp.array[wp.int32],
    b: wp.array2d[wp.float64],
    nv: int,
):
    mode, i = wp.tid()
    weight = mode // 12
    axis = (mode % 12) // 4
    coord = mode % 4
    v = wp.float64(0)
    if i < 3 * nv and i % 3 == axis and fixed[i] == 0:
        v = w[weight, i // 3]
        if coord < 3:
            v *= x[3 * (i // 3) + coord]
    b[mode, i] = v


@wp.kernel
def weighted(
    b: wp.array2d[wp.float64], m: wp.array[wp.float64], out: wp.array2d[wp.float64]
):
    a, i = wp.tid()
    out[a, i] = b[a, i] * m[i]


@wp.kernel
def directors_matrix(
    off: wp.array[wp.int32],
    col: wp.array[wp.int32],
    val: wp.array[wp.float64],
    nv: int,
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    vals: wp.array[wp.float64],
):
    i = wp.tid()
    for p in range(off[i], off[i + 1]):
        j = col[p]
        rows[p] = 0
        cols[p] = 0
        vals[p] = wp.float64(0)
        if i >= 3 * nv and j >= 3 * nv:
            rows[p] = i - 3 * nv
            cols[p] = j - 3 * nv
            vals[p] = val[p]


@wp.kernel
def fixed_diagonal(
    fixed: wp.array[wp.int32],
    start: int,
    rows: wp.array[wp.int32],
    vals: wp.array[wp.float64],
):
    i = wp.tid()
    rows[i] = i
    vals[i] = wp.float64(fixed[start + i])


@wp.kernel
def director_rhs(
    off: wp.array[wp.int32],
    col: wp.array[wp.int32],
    val: wp.array[wp.float64],
    nv: int,
    b: wp.array2d[wp.float64],
    rhs: wp.array2d[wp.float64],
):
    k, e = wp.tid()
    i = 3 * nv + e
    v = wp.float64(0)
    for p in range(off[i], off[i + 1]):
        j = col[p]
        if j < 3 * nv:
            v -= val[p] * b[k, j]
    rhs[k, e] = v


@wp.kernel
def insert_directors(d: wp.array2d[wp.float64], b: wp.array2d[wp.float64], nv: int):
    k, e = wp.tid()
    b[k, 3 * nv + e] = d[k, e]


@wp.kernel
def compliance_diag(
    m: wp.array[wp.float64],
    fixed: wp.array[wp.int32],
    rows: wp.array[wp.int32],
    vals: wp.array[wp.float64],
):
    i = wp.tid()
    rows[i] = i
    vals[i] = m[i]
    if fixed[i] != 0:
        vals[i] = wp.float64(1)


@wp.kernel
def compliance_load(
    m: wp.array[wp.float64],
    fixed: wp.array[wp.int32],
    nv: int,
    rhs: wp.array2d[wp.float64],
):
    a, i = wp.tid()
    v = wp.float64(0)
    if i < 3 * nv and i % 3 == a and fixed[i] == 0:
        v = m[i]
    rhs[a, i] = v


@wp.kernel
def next_load(
    x: wp.array2d[wp.float64], m: wp.array[wp.float64], rhs: wp.array2d[wp.float64]
):
    a, i = wp.tid()
    rhs[a, i] = m[i] * x[a, i]


@wp.kernel
def append_weights(
    w: wp.array2d[wp.float64],
    response: wp.array2d[wp.float64],
    out: wp.array2d[wp.float64],
    start: int,
):
    a, i = wp.tid()
    out[start + a, i] = response[a // 3, 3 * i + a % 3]


@wp.kernel
def add_director_rows(
    off: wp.array[wp.int32],
    col: wp.array[wp.int32],
    val: wp.array[wp.float64],
    b: wp.array2d[wp.float64],
    start: int,
    nv: int,
):
    i = wp.tid()
    for p in range(off[i], off[i + 1]):
        b[start + i, 3 * nv + col[p]] = val[p]


@wp.kernel
def sample_response(
    x0: wp.array[wp.float64],
    response: wp.array2d[wp.float64],
    axis: int,
    sign: wp.float64,
    out: wp.array[wp.float64],
):
    i = wp.tid()
    out[i] = x0[i] + sign * response[axis, i]


@wp.kernel
def accumulate_response(
    g: wp.array[wp.float64], axis: int, out: wp.array2d[wp.float64]
):
    i = wp.tid()
    out[axis, i] -= wp.float64(0.5) * g[i]


def build_basis(shell, k=20, enrichment=2, free_directors=False, nonlinear=True):
    t = time.perf_counter()
    nv = shell.nv
    nd = shell.nd
    shell.evaluate(shell.x0)
    h = shell.stiffness()
    h.nnz_sync()
    rows = wp.empty(h.nnz, dtype=wp.int32)
    cols = wp.empty_like(rows)
    vals = wp.empty(h.nnz, dtype=wp.float64)
    wp.launch(
        weight_entries, nd, [h.offsets, h.columns, h.values, nv, rows, cols, vals]
    )
    hw = ws.bsr_from_triplets(nv, nv, rows, cols, vals)
    mr = wp.empty(nv, dtype=wp.int32)
    mv = wp.empty(nv, dtype=wp.float64)
    wp.launch(mass_entries, nv, [shell.mass, mr, mv])
    mw = ws.bsr_from_triplets(nv, nv, mr, mr, mv)
    pins = sorted({v for vs in shell.patch_vertices.values() for v in vs})
    cr = wp.array(list(range(len(pins))), dtype=wp.int32)
    cc = wp.array(pins, dtype=wp.int32)
    cv = wp.ones(len(pins), dtype=wp.float64)
    c = ws.bsr_from_triplets(len(pins), nv, cr, cc, cv)
    sigma = -1.0
    kkt = saddle_point_matrix(hw, mw, c, shift=sigma)
    inv = CuDSSInverse(
        kkt,
        1,
        mtype="general",
        matching=True,
        refinement_steps=1,
        cooperative_refinement=True,
    )
    eig = ConstrainedKrylovSchur(
        SparseOperator(hw),
        SparseOperator(mw),
        c,
        inv,
        k=k,
        ncv=min(nv, max(2 * k + 20, 64)),
        target=sigma,
        tol=1.0e-8,
    )
    graph = eig.capture(max_iterations=12)
    wp.capture_launch(graph)
    wp.synchronize()
    report = dict(
        weight_count=k,
        converged=int(eig.converged.numpy()[0]),
        max_eigen_residual=float(eig.residuals.numpy().max()),
        max_constraint_residual=float(eig.constraint_residuals.numpy().max()),
        eigenvalues=eig.eigenvalues.numpy().tolist(),
        weight_design="BAC position-block Hessian, rest directors held fixed; affine columns statically lifted into director space",
    )
    assert report["converged"] == k, report
    eigen_count = k
    weights = eig.eigenvectors
    if enrichment:
        ar = wp.empty(nd, dtype=wp.int32)
        av = wp.empty(nd, dtype=wp.float64)
        wp.launch(compliance_diag, nd, [shell.mass, shell.fixed, ar, av])
        ah = ws.bsr_copy(h)
        ws.bsr_axpy(ws.bsr_from_triplets(nd, nd, ar, ar, av), ah)
        ci = CuDSSInverse(ah, 3, mtype="general", matching=True, refinement_steps=2)
        load = wp.zeros((3, nd), dtype=wp.float64)
        response = wp.zeros_like(load)
        wp.launch(compliance_load, load.shape, [shell.mass, shell.fixed, nv, load])
        weights = wp.zeros(
            (k + 9 * enrichment + 9 * int(nonlinear), nv), dtype=wp.float64
        )
        wp.copy(weights, eig.eigenvectors, count=k * nv)
        first_response = wp.empty_like(response)
        for cycle in range(enrichment):
            ci.apply(load, response)
            if cycle == 0:
                wp.copy(first_response, response)
            wp.launch(
                append_weights,
                (9, nv),
                [eig.eigenvectors, response, weights, k + 9 * cycle],
            )
            wp.launch(next_load, load.shape, [response, shell.mass, load])
        if nonlinear:
            sample = wp.empty_like(shell.x0)
            sg = wp.zeros_like(shell.x0)
            load.zero_()
            for axis in range(3):
                for sign in [-1.0, 1.0]:
                    wp.launch(
                        sample_response,
                        nd,
                        [shell.x0, first_response, axis, sign, sample],
                    )
                    shell.evaluate(sample)
                    sg.zero_()
                    shell.gradient(sg)
                    wp.launch(accumulate_response, nd, [sg, axis, load])
            ci.apply(load, response)
            wp.launch(
                append_weights,
                (9, nv),
                [eig.eigenvectors, response, weights, k + 9 * enrichment],
            )
            shell.evaluate(shell.x0)
        wp.synchronize()
        ci.close()
        k += 9 * enrichment + 9 * int(nonlinear)
    b = wp.empty((12 * k, nd), dtype=wp.float64)
    wp.launch(affine_basis, b.shape, [weights, shell.x0, shell.fixed, b, nv])
    # Mass orthonormalization, in Warp. Curved geometry provides independent affine columns.
    mb = wp.empty_like(b)
    gram = wp.zeros((12 * k, 12 * k), dtype=wp.float64)
    factor = wp.zeros_like(gram)
    status = wp.zeros(1, dtype=wp.int32)
    for _ in range(2):
        wp.launch(weighted, b.shape, [b, shell.mass, mb])
        gram.zero_()
        wp.launch_tiled(
            gram_tiled,
            dim=((12 * k + 7) // 8, (12 * k + 7) // 8, (nd + 255) // 256),
            inputs=[b, mb, gram],
            block_dim=128,
        )
        wp.launch(K.cholesky, 1, [gram, factor, status])
        wp.launch(K.triangular, nd, [b, factor])
    assert status.numpy()[0] == 0, "dependent affine columns"
    # Exact linear director relaxation for each retained position coordinate.
    wp.launch(
        directors_matrix, nd, [h.offsets, h.columns, h.values, nv, rows, cols, vals]
    )
    ne = nd - 3 * nv
    htt = ws.bsr_from_triplets(ne, ne, rows, cols, vals)
    dr = wp.empty(ne, dtype=wp.int32)
    dv = wp.empty(ne, dtype=wp.float64)
    wp.launch(fixed_diagonal, ne, [shell.fixed, 3 * nv, dr, dv])
    diag = ws.bsr_from_triplets(ne, ne, dr, dr, dv)
    ws.bsr_axpy(diag, htt)
    rhs = wp.zeros((12 * k, ne), dtype=wp.float64)
    d = wp.zeros_like(rhs)
    wp.launch(director_rhs, rhs.shape, [h.offsets, h.columns, h.values, nv, b, rhs])
    dinv = CuDSSInverse(htt, 12 * k, mtype="spd", refinement_steps=1)
    dinv.apply(rhs, d)
    wp.launch(insert_directors, d.shape, [d, b, nv])
    director_count = 0
    if free_directors:
        from warpack import independent_constraints

        cb = independent_constraints(d, tolerance=1.0e-8)
        director_count = cb.nrow
        augmented = wp.zeros((12 * k + director_count, nd), dtype=wp.float64)
        wp.copy(augmented, b, count=12 * k * nd)
        wp.launch(
            add_director_rows,
            director_count,
            [cb.offsets, cb.columns, cb.values, augmented, 12 * k, nv],
        )
        b = augmented
    wp.synchronize()
    report.update(
        eigen_weight_count=eigen_count,
        compliance_weight_count=9 * enrichment,
        nonlinear_correction_weight_count=9 * int(nonlinear and enrichment > 0),
        total_skinning_weights=k,
        position_coordinates=12 * k,
        director_coordinates=director_count,
        reduced_dofs=b.shape[0],
        full_dofs=nd,
        setup_seconds=time.perf_counter() - t,
    )
    eig.graph = None
    del graph
    inv.close()
    dinv.close()
    return b, h, report
