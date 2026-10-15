"""Verify saved Blender playback against GPU samples and metric units."""

import argparse, json, sys
from pathlib import Path
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--contacts", type=int, required=True)
p.add_argument("--folder")
args = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
folder = Path(args.folder or ROOT / f"build/comparison-{args.contacts}")
reports = {}
for model, label in [("full", "FULL BAC shell"), ("rom", "SKINNING ROM")]:
    s = bpy.data.scenes[f"{args.contacts} contacts • {label}"]
    bpy.context.window.scene = s
    assert s.unit_settings.system == "METRIC" and s.unit_settings.scale_length == 1.0
    panel = next(
        o
        for o in s.objects
        if o.name.startswith(model.upper() + " • nonlinear BAC shell")
    )
    a = np.load(folder / (model + ".npz"))
    x = a["positions"]
    r = json.loads((folder / (model + ".json")).read_text())
    largest = 0.0
    for j in [0, len(x) // 2, len(x) - 1]:
        s.frame_set(round(1 + j * r["dt"] * r["stride"] * 24))
        ev = panel.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = ev.to_mesh()
        co = np.empty((len(mesh.vertices), 3), dtype="f")
        mesh.vertices.foreach_get("co", co.ravel())
        ev.to_mesh_clear()
        largest = max(largest, float(np.max(abs(co - x[j]))))
    assert largest < 1.0e-6, largest
    assert not panel.modifiers
    reports[model] = {
        "max_sample_playback_error_m": largest,
        "meters_per_blender_unit": 1.0,
        "sample_count": len(x),
        "simulation_duration_s": r["dt"] * r["steps"],
    }
(ROOT / f"validation/blender-dynamics-{args.contacts}.json").write_text(
    json.dumps(reports, indent=2)
)
print(reports)
