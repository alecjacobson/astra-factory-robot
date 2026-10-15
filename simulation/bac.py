"""Warp reimplementation of BAC/MidedgeAngleTan + StVK.

First-order directional dual arithmetic computes exact residual Jacobians.
Energy is 1/2 ||r||²; JᵀJ is the exact rest Hessian and the positive
Gauss–Newton tangent away from rest. Directors are independent massless DOFs.
"""

import warp as wp

D = wp.vec2d
V8 = wp.types.vector(8, wp.float64)
V21 = wp.types.vector(21, wp.float64)
I21 = wp.types.vector(21, wp.int32)
M62 = wp.types.matrix((6, 2), wp.float64)
M82 = wp.types.matrix((8, 2), wp.float64)


@wp.func
def mul(a: D, b: D):
    return D(a[0] * b[0], a[1] * b[0] + a[0] * b[1])


@wp.func
def div(a: D, b: D):
    return D(a[0] / b[0], (a[1] * b[0] - a[0] * b[1]) / (b[0] * b[0]))


@wp.func
def sqrt(a: D):
    s = wp.sqrt(a[0])
    return D(s, a[1] / (wp.float64(2) * s))


@wp.func
def atan2(a: D, b: D):
    return D(
        wp.atan2(a[0], b[0]), (b[0] * a[1] - a[0] * b[1]) / (a[0] * a[0] + b[0] * b[0])
    )


@wp.func
def tan(a: D):
    t = wp.tan(a[0])
    return D(t, (wp.float64(1) + t * t) * a[1])


@wp.struct
class DV:
    v: wp.vec3d
    d: wp.vec3d


@wp.func
def sub(a: DV, b: DV):
    c = DV()
    c.v = a.v - b.v
    c.d = a.d - b.d
    return c


@wp.func
def cross(a: DV, b: DV):
    c = DV()
    c.v = wp.cross(a.v, b.v)
    c.d = wp.cross(a.d, b.v) + wp.cross(a.v, b.d)
    return c


@wp.func
def dot(a: DV, b: DV):
    return D(wp.dot(a.v, b.v), wp.dot(a.v, b.d) + wp.dot(a.d, b.v))


@wp.func
def norm(a: DV):
    return sqrt(dot(a, a))


@wp.func
def vertex(q: V21, i: int, direction: int):
    v = DV()
    v.v = wp.vec3d(q[3 * i], q[3 * i + 1], q[3 * i + 2])
    v.d = wp.vec3d(0.0)
    for c in range(3):
        if 3 * i + c == direction:
            v.d[c] = wp.float64(1)
    return v


@wp.func
def forms(q: V21, signs: wp.vec3i, interior: wp.vec3i, direction: int):
    a = vertex(q, 0, direction)
    b = vertex(q, 1, direction)
    c = vertex(q, 2, direction)
    e1 = sub(b, a)
    e2 = sub(c, a)
    a00 = dot(e1, e1)
    a01 = dot(e1, e2)
    a11 = dot(e2, e2)
    ii = wp.matrix(shape=(3, 2), dtype=wp.float64)
    for i in range(3):
        p = vertex(q, i, direction)
        u = vertex(q, (i + 1) % 3, direction)
        v = vertex(q, (i + 2) % 3, direction)
        n = cross(sub(u, p), sub(v, p))
        axis = sub(v, u)
        height = div(norm(n), norm(axis))
        half = D(0.0)
        if interior[i] != 0:
            other = vertex(q, 3 + i, direction)
            nb = cross(sub(v, other), sub(u, other))
            # Swapping global edge orientation swaps both normals and axis; half is unchanged.
            ss = div(dot(cross(n, nb), axis), norm(axis))
            cc = dot(n, nb) + mul(norm(n), norm(nb))
            half = atan2(ss, cc)
        dt = wp.float64(0)
        if direction == 18 + i:
            dt = wp.float64(1)
        alpha = half + D(q[18 + i], dt) * wp.float64(signs[i])
        val = mul(height, tan(alpha)) * wp.float64(2)
        ii[i, 0] = val[0]
        ii[i, 1] = val[1]
    out = M62()
    for j in range(2):
        out[0, j] = a00[j]
        out[1, j] = a01[j]
        out[2, j] = a11[j]
        out[3, j] = ii[0, j] + ii[1, j]
        out[4, j] = ii[0, j]
        out[5, j] = ii[0, j] + ii[2, j]
    return out


@wp.func
def residual(
    q: V21,
    signs: wp.vec3i,
    interior: wp.vec3i,
    rest: V8,
    young: wp.float64,
    nu: wp.float64,
    thickness: wp.float64,
    direction: int,
):
    f = forms(q, signs, interior, direction)
    # Inverse rest edge-coordinate matrix [[length(e1), dot(e1,e2)/length(e1)],[0,altitude]].
    a = rest[0]
    b = rest[1]
    c = rest[2]
    l = wp.sqrt(a)
    v = b / l
    h = wp.sqrt(c - v * v)
    t00 = wp.float64(1) / l
    t01 = -v / (l * h)
    t11 = wp.float64(1) / h
    area = l * h / wp.float64(2)
    lam = young * nu / (wp.float64(1) - nu * nu)
    mu = young / (wp.float64(2) * (wp.float64(1) + nu))
    out = M82()
    for part in range(2):
        off = 3 * part
        aa = D(f[off, 0] - rest[off], f[off, 1])
        ab = D(f[off + 1, 0] - rest[off + 1], f[off + 1, 1])
        bb = D(f[off + 2, 0] - rest[off + 2], f[off + 2, 1])
        s00 = aa * (t00 * t00)
        s01 = aa * (t00 * t01) + ab * (t00 * t11)
        s11 = aa * (t01 * t01) + ab * (wp.float64(2) * t01 * t11) + bb * (t11 * t11)
        coeff = thickness * area / wp.float64(4)
        if part == 1:
            coeff = thickness * thickness * thickness * area / wp.float64(12)
        rr0 = (s00 + s11) * wp.sqrt(coeff * lam)
        rr1 = s00 * wp.sqrt(wp.float64(2) * coeff * mu)
        rr2 = s11 * wp.sqrt(wp.float64(2) * coeff * mu)
        rr3 = s01 * wp.sqrt(wp.float64(4) * coeff * mu)
        for j in range(2):
            out[4 * part, j] = rr0[j]
            out[4 * part + 1, j] = rr1[j]
            out[4 * part + 2, j] = rr2[j]
            out[4 * part + 3, j] = rr3[j]
    return out


@wp.kernel
def rest_data(
    x: wp.array[wp.float64],
    ids: wp.array[I21],
    signs: wp.array[wp.vec3i],
    interior: wp.array[wp.vec3i],
    rest: wp.array[V8],
    mass: wp.array[wp.float64],
    rho: wp.float64,
    t: wp.float64,
):
    f = wp.tid()
    q = V21()
    for j in range(21):
        q[j] = x[ids[f][j]]
    ff = forms(q, signs[f], interior[f], -1)
    rr = V8()
    for j in range(6):
        rr[j] = ff[j, 0]
    rest[f] = rr
    area = wp.sqrt(rr[0] * rr[2] - rr[1] * rr[1]) / wp.float64(2)
    for j in range(9):
        wp.atomic_add(mass, ids[f][j], rho * t * area / wp.float64(3))


@wp.kernel
def evaluate(
    x: wp.array[wp.float64],
    ids: wp.array[I21],
    signs: wp.array[wp.vec3i],
    interior: wp.array[wp.vec3i],
    rest: wp.array[V8],
    young: wp.float64,
    nu: wp.float64,
    t: wp.float64,
    r: wp.array2d[wp.float64],
    jac: wp.array3d[wp.float64],
    fixed: wp.array[wp.int32],
):
    f, j = wp.tid()
    q = V21()
    for k in range(21):
        q[k] = x[ids[f][k]]
    rr = residual(q, signs[f], interior[f], rest[f], young, nu, t, j)
    for k in range(8):
        if j == 0:
            r[f, k] = rr[k, 0]
        val = rr[k, 1]
        if fixed[ids[f][j]] != 0:
            val = wp.float64(0)
        jac[f, k, j] = val


@wp.kernel
def energy(
    x: wp.array[wp.float64],
    ids: wp.array[I21],
    signs: wp.array[wp.vec3i],
    interior: wp.array[wp.vec3i],
    rest: wp.array[V8],
    young: wp.float64,
    nu: wp.float64,
    t: wp.float64,
    out: wp.array[wp.float64],
):
    f = wp.tid()
    q = V21()
    for j in range(21):
        q[j] = x[ids[f][j]]
    rr = residual(q, signs[f], interior[f], rest[f], young, nu, t, -1)
    e = wp.float64(0)
    for k in range(8):
        e += rr[k, 0] * rr[k, 0] / wp.float64(2)
    # Reject the other branch beyond the tan barrier, and collapsed triangles.
    for i in range(3):
        p = vertex(q, i, -1)
        u = vertex(q, (i + 1) % 3, -1)
        v = vertex(q, (i + 2) % 3, -1)
        n = cross(sub(u, p), sub(v, p))
        half = wp.float64(0)
        if interior[f][i] != 0:
            other = vertex(q, 3 + i, -1)
            nb = cross(sub(v, other), sub(u, other))
            ax = sub(v, u)
            half = wp.atan2(
                wp.dot(wp.cross(n.v, nb.v), ax.v) / wp.length(ax.v),
                wp.dot(n.v, nb.v) + wp.length(n.v) * wp.length(nb.v),
            )
        if wp.abs(half + wp.float64(signs[f][i]) * q[18 + i]) > wp.float64(
            1.570795
        ) or wp.length(n.v) < wp.float64(1.0e-14):
            e = wp.float64(1.0e100)
    if not wp.isfinite(e):
        e = wp.float64(1.0e100)
    wp.atomic_add(out, 0, e)


@wp.kernel
def gradient(
    ids: wp.array[I21],
    r: wp.array2d[wp.float64],
    jac: wp.array3d[wp.float64],
    out: wp.array[wp.float64],
):
    f, j = wp.tid()
    value = wp.float64(0)
    for k in range(8):
        value += jac[f, k, j] * r[f, k]
    wp.atomic_add(out, ids[f][j], value)


@wp.kernel
def tangent_local(
    ids: wp.array[I21],
    jac: wp.array3d[wp.float64],
    v: wp.array[wp.float64],
    jv: wp.array2d[wp.float64],
):
    f, k = wp.tid()
    value = wp.float64(0)
    for j in range(21):
        value += jac[f, k, j] * v[ids[f][j]]
    jv[f, k] = value


@wp.kernel
def triplets(
    ids: wp.array[I21],
    jac: wp.array3d[wp.float64],
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    vals: wp.array[wp.float64],
):
    f, i, j = wp.tid()
    v = wp.float64(0)
    for k in range(8):
        v += jac[f, k, i] * jac[f, k, j]
    p = f * 441 + i * 21 + j
    rows[p] = ids[f][i]
    cols[p] = ids[f][j]
    vals[p] = v
