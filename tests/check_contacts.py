"""Geometry coverage and complete-region Dirichlet constraints in both layouts."""

import json
from pathlib import Path
import numpy as np
from shapely.geometry import Point, Polygon
from shapely import union_all
from simulation.model import load_headliner

reports = {}
for count in (4, 2):
    s, data = load_headliner(count)
    vertices = np.array(data["vertices"])
    faces = np.array(data["faces"])
    owners = np.array(data["patch_face_owner"])
    fixed = np.array(s.fixed_host[: 3 * s.nv]).reshape(-1, 3).astype(bool)
    report = {}
    for name, c in data["contacts"].items():
        owner = int(name)
        tris = faces[owners == owner]
        vs = np.unique(tris)
        region = union_all([Polygon(vertices[t, :2]) for t in tris])
        disk = Point(c[:2]).buffer(0.05, quad_segs=512)
        # Tangencies can differ by floating-point roundoff in GEOS predicates.
        assert disk.difference(region).area < 1.0e-18
        assert region.buffer(1.0e-12).covers(disk)
        active = count == 4 or owner in (1, 4)
        assert fixed[vs].all() if active else not fixed[vs].any()
        report[name] = {
            "active": active,
            "vertices": len(vs),
            "triangles": len(tris),
            "area_projected_m2": region.area,
            "covers_100mm_disk": True,
        }
    reports[str(count)] = report
Path("validation/suction-regions.json").write_text(json.dumps(reports, indent=2))
print(reports)
