# HL / 04 — Robotic headliner transfer

A factory robot lifts an automobile headliner from station A, carries it up and over to station B, and returns it in a continuous animation. Four turquoise bellows suction cups stay attached to the panel throughout the motion.

![Animated headliner transfer](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/headliner-transfer.gif)

**[Download the Blender scene](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/scene/headliner_transfer.blend)** · **[Animation GIF](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/headliner-transfer.gif)** · **[Visual references](references/README.md)**

![Factory cell overview](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/hero.png)

## Geometry

The robot is assembled from rigid meshes: pedestal, turntable, shoulder casting, upper arm, elbow housing, forearm, wrist, and aluminum vacuum frame. A parented joint hierarchy animates rotations; the rigid links never scale or deform. Four individually modeled open bellows cups, plungers, and vacuum hoses make the attachments visible.

The headliner is the **midsurface of a thin shell**, with 1,104 control quads and a live, level-2 Catmull–Clark subdivision modifier. It has a molded crown, tapered outline, one sunroof opening, two console/lamp openings, and two small mounting openings. The outer perimeter and all five holes are **open mesh boundaries**. There is no Solidify modifier, back surface, rim wall, or cap. A nominal 3 mm physical thickness is recorded as metadata only.

### Headliner control mesh

Orange marks the six open boundary loops; turquoise circles mark the four suction footprints. The screenshot shows the unsubdivided cage over its own surface, so every displayed edge is a real control edge.

![Headliner quad cage and open hole boundaries](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/headliner-wireframe.png)

### Subdivision mesh

The finer wireframe shows the actual evaluated level-1 subdivision mesh over the smooth level-2 surface. All openings remain open.

![Subdivided headliner mesh](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/subdivision-wireframe.png)

### Robot mesh

The inspection view overlays the original rigid-part mesh edges on the shaded assembly. These inspection views are also saved as separate scenes inside the `.blend`.

![Robot and gripper mesh wireframes](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/robot-wireframe.png)

## Animation and inspection

Open `scene/headliner_transfer.blend` in **Blender 4.5 LTS** and press Space in the main scene. No add-ons, external textures, or running Python scripts are required for playback. Use the Scene selector to switch between:

1. **01 • HEADLINER TRANSFER** — finished factory cell and animation.
2. **02 • Headliner control topology** — quad cage, open boundaries, and contact footprints.
3. **03 • Subdivision surface topology** — refined surface and evaluated wireframe.
4. **04 • Robot rigid mesh topology** — arm, gripper, and panel wireframes.

The animation has 192 frames at 24 fps (8 seconds). The GIF samples it at 12 fps. Station centers are 4.70 m apart; the panel reference height rises from 1.15 m to 2.70 m. Timeline markers identify the station holds, lift, overhead crossing, and return. Analytical two-link inverse kinematics drives the arm, with base azimuth and wrist compensation preserving the horizontal headliner orientation. Joint rotations are baked at every frame.

This is a **kinematic transfer demonstration**: suction stays engaged, including at station holds. It does not simulate release, vacuum pressure, shell deformation, dynamics, or manufacturer joint limits.

## Validation

The builder checks the shell topology, exact subdivided-surface contact positions, tool path, and fixed link lengths across the full animation. Results are saved in [geometry.json](validation/geometry.json), with the requested motion in [motion.json](validation/motion.json). A separate [saved-asset audit](validation/asset-audit.json) checks connectedness and suction contact at every frame, and checks the headliner against the principal arm meshes for surface intersections. This is not a comprehensive cell or self-collision certification.

| Check | Result |
|---|---:|
| Control vertices / quads | 1,235 / 1,104 |
| Open boundary loops | 6: outer perimeter + 5 holes |
| Edges incident to more than two faces | 0 |
| Euler characteristic | −4 |
| Suction attachments | 4 |
| Maximum tool path error | < 0.001 mm |
| Maximum rigid-link length error | < 0.001 mm |
| Maximum suction-center distance from subdivision surface | < 0.001 mm |

## Reproduce

Git LFS stores the Blender scene, GIF, and PNG presentation images. Install Git LFS before cloning:

```sh
git lfs install
git clone https://github.com/alecjacobson/astra-factory-robot.git
cd astra-factory-robot
git lfs pull
```

Rebuild and render with Blender 4.5, Python 3, Pillow, DejaVu Sans fonts, and [gifski](https://gif.ski/):

```sh
mkdir -p build/frames scene media validation
blender -b --python scripts/build_scene.py
blender -b scene/headliner_transfer.blend --python scripts/render.py -- animation
blender -b scene/headliner_transfer.blend --python scripts/render.py -- stills
blender -b scene/headliner_transfer.blend --python scripts/validate_scene.py
python scripts/package_media.py --gifski /path/to/gifski
```

Rendering uses Cycles with OptiX when available, with a CPU fallback in the render scripts. For a nonstandard font installation, pass `--font-dir /path/to/dejavu` to the packaging script. The animation renderer reuses the symmetric return trajectory when packaging the GIF; the Blender file contains the complete baked return motion. `build/` contains disposable intermediate renders and is ignored by Git. Reference sources and modeling decisions are recorded in [references/README.md](references/README.md).
