"""Consistent BAC tangent via second-order mixed directional dual arithmetic.

Each lane computes one local Hessian entry; no finite differences or host AD.
"""

import warp as wp
from .bac import V8, V21, I21

D = wp.vec4d
M62 = wp.types.matrix((6, 4), wp.float64)
M82 = wp.types.matrix((8, 4), wp.float64)


@wp.func
def mul(a: D, b: D):
    return D(
        a[0] * b[0],
        a[1] * b[0] + a[0] * b[1],
        a[2] * b[0] + a[0] * b[2],
        a[3] * b[0] + a[1] * b[2] + a[2] * b[1] + a[0] * b[3],
    )


@wp.func
def chain(a: D, f: wp.float64, df: wp.float64, ddf: wp.float64):
    return D(f, df * a[1], df * a[2], df * a[3] + ddf * a[1] * a[2])


@wp.func
def div(a: D, b: D):
    inv = wp.float64(1) / b[0]
    return mul(a, chain(b, inv, -inv * inv, wp.float64(2) * inv * inv * inv))


@wp.func
def sqrt(a: D):
    s = wp.sqrt(a[0])
    return chain(a, s, wp.float64(0.5) / s, -wp.float64(0.25) / (s * s * s))


@wp.func
def atan2(a: D, b: D):
    x = a[0]
    y = b[0]
    r = x * x + y * y
    da = y / r
    db = -x / r
    daa = -wp.float64(2) * x * y / (r * r)
    dbb = -daa
    dab = (x * x - y * y) / (r * r)
    return D(
        wp.atan2(x, y),
        da * a[1] + db * b[1],
        da * a[2] + db * b[2],
        da * a[3]
        + db * b[3]
        + daa * a[1] * a[2]
        + dbb * b[1] * b[2]
        + dab * (a[1] * b[2] + a[2] * b[1]),
    )


@wp.func
def tan(a: D):
    t = wp.tan(a[0])
    df = wp.float64(1) + t * t
    return chain(a, t, df, wp.float64(2) * t * df)


@wp.struct
class DV:
    v: wp.vec3d
    d: wp.vec3d
    e: wp.vec3d
    h: wp.vec3d


@wp.func
def sub(a: DV, b: DV):
    c = DV()
    c.v = a.v - b.v
    c.d = a.d - b.d
    c.e = a.e - b.e
    c.h = a.h - b.h
    return c


@wp.func
def cross(a: DV, b: DV):
    c = DV()
    c.v = wp.cross(a.v, b.v)
    c.d = wp.cross(a.d, b.v) + wp.cross(a.v, b.d)
    c.e = wp.cross(a.e, b.v) + wp.cross(a.v, b.e)
    c.h = (
        wp.cross(a.h, b.v)
        + wp.cross(a.d, b.e)
        + wp.cross(a.e, b.d)
        + wp.cross(a.v, b.h)
    )
    return c


@wp.func
def dot(a: DV, b: DV):
    return D(
        wp.dot(a.v, b.v),
        wp.dot(a.d, b.v) + wp.dot(a.v, b.d),
        wp.dot(a.e, b.v) + wp.dot(a.v, b.e),
        wp.dot(a.h, b.v) + wp.dot(a.d, b.e) + wp.dot(a.e, b.d) + wp.dot(a.v, b.h),
    )


@wp.func
def norm(a: DV):
    return sqrt(dot(a, a))


@wp.func
def vertex(q: V21, i: int, direction: int, direction2: int):
    v = DV()
    v.v = wp.vec3d(q[3 * i], q[3 * i + 1], q[3 * i + 2])
    v.d = wp.vec3d(0.0)
    v.e = wp.vec3d(0.0)
    v.h = wp.vec3d(0.0)
    for c in range(3):
        if 3 * i + c == direction:
            v.d[c] = wp.float64(1)
        if 3 * i + c == direction2:
            v.e[c] = wp.float64(1)
    return v


@wp.func
def forms(q: V21, signs: wp.vec3i, interior: wp.vec3i, direction: int, direction2: int):
    a = vertex(q, 0, direction, direction2)
    b = vertex(q, 1, direction, direction2)
    c = vertex(q, 2, direction, direction2)
    e1 = sub(b, a)
    e2 = sub(c, a)
    a00 = dot(e1, e1)
    a01 = dot(e1, e2)
    a11 = dot(e2, e2)
    ii = wp.matrix(shape=(3, 4), dtype=wp.float64)
    for i in range(3):
        p = vertex(q, i, direction, direction2)
        u = vertex(q, (i + 1) % 3, direction, direction2)
        v = vertex(q, (i + 2) % 3, direction, direction2)
        n = cross(sub(u, p), sub(v, p))
        axis = sub(v, u)
        height = div(norm(n), norm(axis))
        half = D(0.0)
        if interior[i] != 0:
            other = vertex(q, 3 + i, direction, direction2)
            nb = cross(sub(v, other), sub(u, other))
            # Swapping global edge orientation swaps both normals and axis; half is unchanged.
            ss = div(dot(cross(n, nb), axis), norm(axis))
            cc = dot(n, nb) + mul(norm(n), norm(nb))
            half = atan2(ss, cc)
        dt = wp.float64(0)
        if direction == 18 + i:
            dt = wp.float64(1)
        alpha = half + D(
            q[18 + i], dt, wp.float64(direction2 == 18 + i), wp.float64(0)
        ) * wp.float64(signs[i])
        val = mul(height, tan(alpha)) * wp.float64(2)
        ii[i, 0] = val[0]
        ii[i, 1] = val[1]
        ii[i, 2] = val[2]
        ii[i, 3] = val[3]
    out = M62()
    for j in range(4):
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
    direction2: int,
):
    f = forms(q, signs, interior, direction, direction2)
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
        aa = D(f[off, 0] - rest[off], f[off, 1], f[off, 2], f[off, 3])
        ab = D(
            f[off + 1, 0] - rest[off + 1], f[off + 1, 1], f[off + 1, 2], f[off + 1, 3]
        )
        bb = D(
            f[off + 2, 0] - rest[off + 2], f[off + 2, 1], f[off + 2, 2], f[off + 2, 3]
        )
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
        for j in range(4):
            out[4 * part, j] = rr0[j]
            out[4 * part + 1, j] = rr1[j]
            out[4 * part + 2, j] = rr2[j]
            out[4 * part + 3, j] = rr3[j]
    return out


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
    fixed: wp.array[wp.int32],
    h: wp.array3d[wp.float64],
):
    f, i, j = wp.tid()
    if j < i:
        return
    value = wp.float64(0)
    if fixed[ids[f][i]] == 0 and fixed[ids[f][j]] == 0:
        q = V21()
        for k in range(21):
            q[k] = x[ids[f][k]]
        rr = residual(q, signs[f], interior[f], rest[f], young, nu, t, i, j)
        for k in range(8):
            value += rr[k, 1] * rr[k, 2] + rr[k, 0] * rr[k, 3]
    h[f, i, j] = value
    h[f, j, i] = value


@wp.kernel
def product(
    ids: wp.array[I21],
    h: wp.array3d[wp.float64],
    x: wp.array[wp.float64],
    out: wp.array[wp.float64],
):
    f, i = wp.tid()
    value = wp.float64(0)
    for j in range(21):
        value += h[f, i, j] * x[ids[f][j]]
    wp.atomic_add(out, ids[f][i], value)
