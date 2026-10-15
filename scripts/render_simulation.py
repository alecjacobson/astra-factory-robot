"""Bake actual GPU trajectories into editable Blender shape keys and render them."""

import argparse, json, sys
from pathlib import Path
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--contacts", type=int, default=4)
p.add_argument("--folder")
p.add_argument("--render", action="store_true")
p.add_argument("--test", action="store_true")
args = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
folder = Path(args.folder or ROOT / f"build/comparison-{args.contacts}")
mesh = json.loads((folder / "mesh.json").read_text())
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "OPTIX"
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == "OPTIX"
source = bpy.data.scenes[
    "01 • HEADLINER TRANSFER" if args.contacts == 4 else "06 • TWO-CONTACT TRANSFER"
]
bpy.context.window.scene = source
source.frame_set(1)
panel_source = bpy.data.objects["HEADLINER • midsurface only"]


def material(name, color):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    bs = m.node_tree.nodes.get("Principled BSDF")
    bs.inputs["Base Color"].default_value = (*color, 1)
    bs.inputs["Roughness"].default_value = 0.68
    return m


patch_material = material("Rigid suction region • 100 mm", (0.025, 0.48, 0.34))
shell_material = material("BAC simulated headliner • ivory", (0.70, 0.66, 0.52))
scenes = []
for model in ["full", "rom"]:
    data = np.load(folder / (model + ".npz"))
    positions = data["positions"]
    report = json.loads((folder / (model + ".json")).read_text())
    s = bpy.data.scenes.new(
        f"{args.contacts} contacts • "
        + ("FULL BAC shell" if model == "full" else "SKINNING ROM")
    )
    s.use_fake_user = True
    s.unit_settings.system = "METRIC"
    s.unit_settings.scale_length = 1
    s.unit_settings.length_unit = "METERS"
    s.world = source.world
    s.render.resolution_x = 720
    s.render.resolution_y = 720
    s.render.resolution_percentage = 100
    s.render.engine = "CYCLES"
    s.cycles.device = "GPU"
    s.cycles.samples = 20
    s.cycles.use_denoising = True
    s.cycles.denoiser = "OPTIX"
    s.render.use_persistent_data = True
    s.view_settings.view_transform = "AgX"
    s.view_settings.exposure = -0.5
    s.render.fps = 24
    s.frame_start = 1
    s.frame_end = round(1 + (len(positions) - 1) * report["dt"] * report["stride"] * 24)
    mapping = {}
    for original in source.objects:
        if not original.visible_get(view_layer=source.view_layers[0]):
            continue
        o = original.copy()
        s.collection.objects.link(o)
        mapping[original] = o
    for original, o in mapping.items():
        if original.parent in mapping:
            o.parent = mapping[original.parent]
    panel = mapping[panel_source]
    panel.name = f"{model.upper()} • nonlinear BAC shell • {args.contacts} cups"
    panel.modifiers.clear()
    geo = bpy.data.meshes.new(panel.name)
    geo.from_pydata(mesh["vertices"], [], mesh["faces"])
    geo.update()
    panel.data = geo
    geo.materials.append(shell_material)
    geo.materials.append(patch_material)
    for f, poly in enumerate(geo.polygons):
        poly.use_smooth = True
        owner = mesh["patch_face_owner"][f]
        poly.material_index = int(
            owner != 0 and (args.contacts == 4 or owner in (1, 4))
        )
    panel.shape_key_add(name="Rest midsurface")
    step_frames = report["dt"] * report["stride"] * 24
    for j in range(1, len(positions)):
        key = panel.shape_key_add(
            name=f"t = {j * report['dt'] * report['stride']:.4f} s"
        )
        key.data.foreach_set("co", positions[j].astype("f").ravel())
        frame = 1 + j * step_frames
        for f, value in [
            (frame - step_frames, 0),
            (frame, 1),
            (frame + step_frames, 0),
        ]:
            key.value = value
            key.keyframe_insert("value", frame=f)
        key.value = 0
    # Linear temporal interpolation; every saved sample is an actual simulation frame.
    action = geo.shape_keys.animation_data.action
    for layer in action.layers:
        for strip in layer.strips:
            for slot in action.slots:
                bag = strip.channelbag(slot)
                if bag:
                    for curve in bag.fcurves:
                        for key in curve.keyframe_points:
                            key.interpolation = "LINEAR"
    camdata = bpy.data.cameras.new(model + " • following shell camera")
    cam = bpy.data.objects.new(model + " • following shell camera", camdata)
    s.collection.objects.link(cam)
    cam.parent = panel
    cam.location = (0.5, -1.75, 2.5)
    cam.rotation_euler = (-cam.location).to_track_quat("-Z", "Y").to_euler()
    camdata.type = "ORTHO"
    camdata.ortho_scale = 2.55
    s.camera = cam
    cell = mapping[source.camera]
    cell.name = f"{model.upper()} • entire robot cell camera"
    s["full_cell_camera"] = cell.name
    s["simulation"] = json.dumps(report)
    s["deformation_scale"] = 1.0
    s["contact_coupling"] = "Entire 100 mm circular region; green constrained triangles"
    s["shell_model"] = (
        "BAC MidedgeAngleTan + StVK, 3 mm, curved rest; no self/contact collision"
    )
    scenes.append((s, model))
# The saved file has both simulations and the original inspection scenes.
bpy.context.window.scene = scenes[0][0]
scenes[0][0].frame_set(49)
text = bpy.data.texts.new(f"READ ME • {args.contacts}-contact dynamic comparison")
text.write(
    "Select FULL BAC shell or SKINNING ROM from the Scene menu. Both use actual, unamplified GPU trajectories. Green faces are rigid suction patches. Each scene has a following shell camera and an entire robot cell camera. Shape keys store the 12 fps samples; playback interpolates linearly. The source kinematic scenes remain available. Physical parameters, solver convergence and graph audit are in the repository validation reports.\n"
    + json.dumps(json.loads((folder / "comparison.json").read_text()), indent=2)
)
bpy.ops.wm.save_as_mainfile(
    filepath=str(ROOT / f"scene/headliner_dynamics_{args.contacts}_contacts.blend")
)
if args.render:
    for s, model in scenes:
        bpy.context.window.scene = s
        s.render.resolution_x = 720
        s.render.resolution_y = 720
        s.render.resolution_percentage = 100
        output = folder / (model + "-frames")
        output.mkdir(exist_ok=True)
        indices = [min(48, len(positions) - 1)] if args.test else range(len(positions))
        for j in indices:
            path = output / f"{j:04d}.png"
            if path.exists():
                continue
            s.frame_set(round(1 + step_frames * j))
            s.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
