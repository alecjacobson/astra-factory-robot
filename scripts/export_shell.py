"""Export a simulation tessellation; Blender is only used for geometry I/O."""

import bpy, json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
s = bpy.data.scenes["01 • HEADLINER TRANSFER"]
bpy.context.window.scene = s
s.frame_set(49)
o = bpy.data.objects["HEADLINER • midsurface only"]
ob = o.copy()
ob.data = o.data.copy()
s.collection.objects.link(ob)
ob.parent = None
ob.matrix_world.identity()
for mod in ob.modifiers:
    mod.levels = 1
bpy.context.view_layer.objects.active = ob
ob.select_set(True)
for mod in list(ob.modifiers):
    bpy.ops.object.modifier_apply(modifier=mod.name)
mod = ob.modifiers.new("Simulation tessellation", "DECIMATE")
mod.ratio = 0.055
bpy.ops.object.modifier_apply(modifier=mod.name)
ob.data.calc_loop_triangles()
vertices = [list(v.co) for v in ob.data.vertices]
faces = [list(t.vertices) for t in ob.data.loop_triangles]
# Keep the exact contact geometry/units and the complete prescribed carrier path.
contacts = json.loads((root / "validation/contact-configurations.json").read_text())[
    "configurations"
]["4"]["contact_centers_in_headliner_coordinates_m"]
poses = []
for f in range(1, 194):
    s.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    m = o.evaluated_get(dg).matrix_world
    poses.append({"frame": f, "matrix": [list(r) for r in m]})
out = root / "simulation/headliner_mesh.json"
out.write_text(
    json.dumps(
        dict(
            vertices=vertices,
            faces=faces,
            contacts=contacts,
            poses=poses,
            meters_per_unit=1,
            source="Subdivision level 1, decimated to 5.5%; same tessellation for FOM and ROM",
        )
    )
)
print("EXPORTED", len(vertices), len(faces), out)
