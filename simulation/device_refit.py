"""Capture-safe numeric FSAI refit on frozen sparse supports.

Uses the pinned Warp FSAI kernels; replaces host validation/readbacks with a
GPU commit guard. An invalid candidate leaves both previous factors intact.
The exact mechanical operator is independent of this approximate inverse.
"""

import warp as wp
import warp.sparse as ws
from warp.optim import linear
from warp._src.optim import fsai as implementation
from .bac import I21


@wp.kernel
def locate(
    ids: wp.array[I21],
    off: wp.array[wp.int32],
    col: wp.array[wp.int32],
    mapping: wp.array3d[wp.int32],
):
    f, i, j = wp.tid()
    mapping[f, i, j] = ws.bsr_block_index(ids[f][i], ids[f][j], off, col)


@wp.kernel
def scatter(
    h: wp.array3d[wp.float64],
    mapping: wp.array3d[wp.int32],
    values: wp.array[wp.float64],
):
    f, i, j = wp.tid()
    p = mapping[f, i, j]
    if p >= 0:
        wp.atomic_add(values, p, h[f, i, j])


@wp.kernel
def add_diag(
    off: wp.array[wp.int32],
    col: wp.array[wp.int32],
    values: wp.array[wp.float64],
    diag: wp.array[wp.float64],
):
    i = wp.tid()
    p = ws.bsr_block_index(i, i, off, col)
    values[p] += diag[i]


@wp.kernel
def guard(
    bad: wp.array[wp.int32], okay: wp.array[wp.int32], counts: wp.array[wp.int32]
):
    okay[0] = int(bad[0] == 0)
    if okay[0] != 0:
        wp.atomic_add(counts, 0, 1)
    else:
        wp.atomic_add(counts, 1, 1)


class DeviceRefit:
    def __init__(self, matrix, ids, diag):
        self.matrix = matrix
        self.ids = ids
        self.diag = diag
        self.inverse = linear.FSAI(
            matrix,
            max_row_size=64,
            max_step_size=3,
            apply_lanes=8,
            factor_dtype=wp.float32,
            reuse_pattern=True,
        )
        self.plan = self.inverse._refit
        self.diagonal = implementation._build_kernels(
            self.inverse.scalar_type,
            self.inverse.max_row_size,
            self.inverse.max_step_size,
        )[0]
        self.mapping = wp.empty((ids.size, 21, 21), dtype=wp.int32)
        wp.launch(
            locate,
            self.mapping.shape,
            [ids, matrix.offsets, matrix.columns, self.mapping],
        )
        self.okay = wp.zeros(1, dtype=wp.int32)
        self.counts = wp.zeros(2, dtype=wp.int32)

    def commit(self):
        p = self.plan
        o = self.inverse
        wp.copy(o.G.values, p.candidate)
        wp.launch(
            p.scatter_transpose,
            p.transpose.size,
            [o.G.values, p.transpose, o.GT.values],
        )

    def update(self, h):
        a = self.matrix
        p = self.plan
        o = self.inverse
        s = p.scalar
        a.values.zero_()
        wp.launch(scatter, self.mapping.shape, [h, self.mapping, a.values])
        wp.launch(add_diag, a.nrow, [a.offsets, a.columns, a.values, self.diag])
        p.bad.zero_()
        wp.launch(p.gather, p.mapping.size, [a.values, p.mapping, p.values])
        wp.launch(
            self.diagonal, s.nrow, [s.offsets, s.columns, p.values, p.scale, p.bad]
        )
        wp.launch(
            p.refit,
            s.nrow,
            [
                s.offsets,
                s.columns,
                p.values,
                o.G.offsets,
                o.G.columns,
                p.scale,
                wp.float64(p.floor),
                p.candidate,
                p.bad,
            ],
        )
        wp.launch(guard, 1, [p.bad, self.okay, self.counts])
        wp.capture_if(self.okay, self.commit)
