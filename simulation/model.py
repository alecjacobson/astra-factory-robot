"""Mesh topology is host preprocessing; all mechanical assembly is Warp."""

import json, math
from pathlib import Path
import warp as wp
import warp.sparse as ws
from . import bac


def connectivity(vertices, faces):
    n = len(vertices)
    edge_faces = {}
    for f, tri in enumerate(faces):
        for i in range(3):
            a, b = tri[(i + 1) % 3], tri[(i + 2) % 3]
            key = tuple(sorted((a, b)))
            side = int(a > b)
            if key not in edge_faces:
                edge_faces[key] = [-1, -1]
            assert edge_faces[key][side] == -1, "nonmanifold or inconsistent winding"
            edge_faces[key][side] = f
    edges = sorted(edge_faces)
    lookup = {e: i for i, e in enumerate(edges)}
    ids = []
    signs = []
    interior = []
    for f, tri in enumerate(faces):
        vs = list(tri)
        es = []
        ss = []
        ins = []
        for i in range(3):
            a, b = tri[(i + 1) % 3], tri[(i + 2) % 3]
            key = tuple(sorted((a, b)))
            e = lookup[key]
            side = int(a > b)
            other = edge_faces[key][1 - side]
            vs.append(
                next(v for v in faces[other] if v not in key) if other >= 0 else tri[i]
            )
            es.append(3 * n + e)
            ss.append(1 if side == 0 else -1)
            ins.append(int(other >= 0))
        ids.append([3 * v + c for v in vs for c in range(3)] + es)
        signs.append(ss)
        interior.append(ins)
    return ids, signs, interior, edges, edge_faces


def intersects_disk(tri, vertices, c, r):
    p = [vertices[i] for i in tri]
    for v in p:
        if (v[0] - c[0]) ** 2 + (v[1] - c[1]) ** 2 <= r * r:
            return True
    cross = []
    for a, b in zip(p, p[1:] + p[:1]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        den = dx * dx + dy * dy
        s = (
            max(0, min(1, ((c[0] - a[0]) * dx + (c[1] - a[1]) * dy) / den))
            if den
            else 0
        )
        if (a[0] + s * dx - c[0]) ** 2 + (a[1] + s * dy - c[1]) ** 2 <= r * r:
            return True
        cross.append(dx * (c[1] - a[1]) - dy * (c[0] - a[0]))
    return min(cross) >= 0 or max(cross) <= 0


def mesh_contact_disks(data, segments=48):
    """Conforming circumscribed circle polygons, <=0.108 mm radial overreach.

    Shapely is used only for offline geometry/topology, never mechanical numerics.
    Each patch replaces a small neighborhood with a constrained triangulation.
    """
    import shapely
    from shapely.geometry import Polygon

    vertices = [list(v) for v in data["vertices"]]
    faces = [list(t) for t in data["faces"]]
    owners = [0] * len(faces)
    for key, c in data["contacts"].items():
        selected = [
            i for i, t in enumerate(faces) if intersects_disk(t, vertices, c, 0.095)
        ]
        originals = [faces[i] for i in selected]
        region = shapely.union_all(
            [
                Polygon([(vertices[v][0], vertices[v][1]) for v in tri])
                for tri in originals
            ]
        )
        radius = 0.05 / math.cos(math.pi / segments)
        circle = Polygon(
            [
                (
                    c[0] + radius * math.cos(2 * math.pi * i / segments),
                    c[1] + radius * math.sin(2 * math.pi * i / segments),
                )
                for i in range(segments)
            ]
        )
        assert region.contains(circle)
        # All old neighborhood vertices are removed if no longer referenced.
        lut = {(round(v[0], 12), round(v[1], 12)): i for i, v in enumerate(vertices)}

        def add_point(x, y):
            k = (round(x, 12), round(y, 12))
            if k in lut:
                return lut[k]
            z = None
            for tri in originals:
                a, b, d = [vertices[i] for i in tri]
                den = (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0])
                u = ((x - a[0]) * (d[1] - a[1]) - (y - a[1]) * (d[0] - a[0])) / den
                v = ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])) / den
                if min(u, v, 1 - u - v) > -1.0e-8:
                    z = a[2] + u * (b[2] - a[2]) + v * (d[2] - a[2])
                    break
            assert z is not None, (x, y)
            lut[k] = len(vertices)
            vertices.append([x, y, z])
            return lut[k]

        omit = set(selected)
        new = [t for i, t in enumerate(faces) if i not in omit]
        newowners = [o for i, o in enumerate(owners) if i not in omit]
        for poly, owner in [(region.difference(circle), 0), (circle, int(key))]:
            triangles = shapely.constrained_delaunay_triangles(poly)
            for t in triangles.geoms:
                pts = list(t.exterior.coords)[:3]
                if (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1]) - (
                    pts[1][1] - pts[0][1]
                ) * (pts[2][0] - pts[0][0]) < 0:
                    pts.reverse()
                new.append([add_point(x, y) for x, y in pts])
                newowners.append(owner)
        faces, owners = new, newowners
    used = sorted({v for tri in faces for v in tri})
    remap = {v: i for i, v in enumerate(used)}
    return dict(
        data,
        vertices=[vertices[i] for i in used],
        faces=[[remap[i] for i in t] for t in faces],
        patch_face_owner=owners,
        patch_polygon_segments=segments,
        patch_max_radial_overreach_m=0.05 * (1 / math.cos(math.pi / segments) - 1),
    )


class Shell:
    def __init__(
        self,
        vertices,
        faces,
        young=3.0e8,
        nu=0.3,
        thickness=0.003,
        density=250.0,
        patches=None,
        patch_face_owner=None,
        device="cuda:0",
    ):
        if (
            not all(math.isfinite(v) and v > 0 for v in (young, thickness, density))
            or not 0 <= nu < 0.5
        ):
            raise ValueError("require E, thickness, density >0 and 0<=nu<0.5")
        self.device = device
        self.vertices = vertices
        self.faces = faces
        self.nv = len(vertices)
        self.nf = len(faces)
        ids, signs, interior, self.edges, self.edge_faces = connectivity(
            vertices, faces
        )
        self.nd = 3 * self.nv + len(self.edges)
        self.young = young
        self.nu = nu
        self.thickness = thickness
        self.density = density
        self.x0 = wp.array(
            [c for v in vertices for c in v] + [0.0] * len(self.edges),
            dtype=wp.float64,
            device=device,
        )
        self.ids = wp.array(ids, dtype=bac.I21, device=device)
        self.signs = wp.array(signs, dtype=wp.vec3i, device=device)
        self.interior = wp.array(interior, dtype=wp.vec3i, device=device)
        self.rest = wp.zeros(self.nf, dtype=bac.V8, device=device)
        self.mass = wp.zeros(self.nd, dtype=wp.float64, device=device)
        wp.launch(
            bac.rest_data,
            self.nf,
            [
                self.x0,
                self.ids,
                self.signs,
                self.interior,
                self.rest,
                self.mass,
                density,
                thickness,
            ],
            device=device,
        )
        self.fixed_host = [0] * self.nd
        self.patch_vertices = {}
        self.patch_faces = {}
        for name, c in (patches or {}).items():
            fs = (
                [f for f, owner in enumerate(patch_face_owner) if owner == int(name)]
                if patch_face_owner is not None
                else [
                    f
                    for f, tri in enumerate(faces)
                    if intersects_disk(tri, vertices, c, 0.05)
                ]
            )
            vs = sorted({v for f in fs for v in faces[f]})
            self.patch_vertices[name] = vs
            self.patch_faces[name] = fs
            for v in vs:
                for k in range(3):
                    self.fixed_host[3 * v + k] = 1
            # Directors wholly inside a rigid patch are prescribed too. Rim directors stay free.
            selected = set(fs)
            for e, key in enumerate(self.edges):
                if all(f in selected for f in self.edge_faces[key] if f >= 0):
                    self.fixed_host[3 * self.nv + e] = 1
        self.fixed = wp.array(self.fixed_host, dtype=wp.int32, device=device)
        self.r = wp.zeros((self.nf, 8), dtype=wp.float64, device=device)
        self.jac = wp.zeros((self.nf, 8, 21), dtype=wp.float64, device=device)
        self.evaluate(self.x0)

    def evaluate(self, x):
        wp.launch(
            bac.evaluate,
            (self.nf, 21),
            [
                x,
                self.ids,
                self.signs,
                self.interior,
                self.rest,
                self.young,
                self.nu,
                self.thickness,
                self.r,
                self.jac,
                self.fixed,
            ],
            device=self.device,
        )

    def energy(self, x, out):
        wp.launch(
            bac.energy,
            self.nf,
            [
                x,
                self.ids,
                self.signs,
                self.interior,
                self.rest,
                self.young,
                self.nu,
                self.thickness,
                out,
            ],
            device=self.device,
        )

    def gradient(self, out):
        wp.launch(
            bac.gradient,
            (self.nf, 21),
            [self.ids, self.r, self.jac, out],
            device=self.device,
        )

    def stiffness(self):
        count = self.nf * 441
        rows = wp.empty(count, dtype=wp.int32, device=self.device)
        cols = wp.empty_like(rows)
        vals = wp.empty(count, dtype=wp.float64, device=self.device)
        wp.launch(
            bac.triplets,
            (self.nf, 21, 21),
            [self.ids, self.jac, rows, cols, vals],
            device=self.device,
        )
        return ws.bsr_from_triplets(self.nd, self.nd, rows, cols, vals)


def load_headliner(contacts=4, refine=True, young=3.0e8):
    data = json.loads((Path(__file__).parent / "headliner_mesh.json").read_text())
    if refine:
        data = mesh_contact_disks(data)
    patches = {
        k: v for k, v in data["contacts"].items() if contacts == 4 or k in ("1", "4")
    }
    return Shell(
        data["vertices"],
        data["faces"],
        young=young,
        patches=patches,
        patch_face_owner=data.get("patch_face_owner"),
    ), data
