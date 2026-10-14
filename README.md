# HL / 04 — Robotic headliner transfer

A factory robot lifts an automobile headliner from station A, carries it up and over to station B, and returns it in a continuous animation. Four turquoise bellows suction cups stay attached to the panel throughout the motion.

![Animated headliner transfer](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/headliner-transfer.gif)

**[Download the Blender scene](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/scene/headliner_transfer.blend)** · **[Animation GIF](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/headliner-transfer.gif)** · **[Visual and dimensional references](references/README.md)**

![Factory cell overview](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/hero.png)

## Physical dimensions

**One Blender unit = one meter**, in every scene. The second pass resizes the cell geometry to a representative passenger-car headliner and long-reach industrial arm. The resize is baked into mesh data and local translations; all object scales remain `(1, 1, 1)`. Dimensions below are measured from the saved geometry, rather than inferred from labels. The complete measurement report is [dimensions.json](validation/dimensions.json).

| Feature | Final size |
|---|---:|
| Headliner envelope, width × length | approximately **1,421 × 1,870 mm** |
| Headliner form depth, highest to lowest surface point | approximately **165 mm** |
| Nominal material thickness | **3 mm**, metadata only |
| Suction lip diameter | **100 mm** |
| Bellows height | **61 mm** |
| Vacuum hose outside diameter | **10 mm** |
| Adjustable plunger rod diameter | **16 mm** |
| Receiving pad diameter | **70 mm** |
| Main arm links | **1,657.5 + 1,530 mm** |
| Station center spacing | **3,995 mm** |
| Headliner origin, station / overhead height | **977.5 / 2,295 mm** |

The 165 mm form depth describes the curved panel's envelope, not its material thickness. The two arm lengths sum to 3,187.5 mm; that is a geometric link-length sum, not a manufacturer-rated robot reach. The panel and arm remain original generic designs, not measured OEM CAD.

Compared with the first pass, the cell is 15% smaller, the oversized 190 mm suction lips are now 100 mm, and hose, fastener, plunger, and fixture-pad dimensions have been revised individually. All four cup lips conform to the evaluated shell surface; the station pads and their support posts are refitted to that surface.

## Refined headliner

The headliner is the **midsurface of a thin shell**, with **8,184 control quads** and a live, level-2 Catmull–Clark subdivision modifier. The forming geometry includes a rounded, turned-down perimeter, a raised sunroof shoulder and recessed reveal, two visor trays, overhead-console and rear-lamp seats, four grab-handle mounting pockets, and a shallow transverse formed contour. A fine procedural textile shader adds surface finish without texture dependencies.

There are **eleven open apertures**: one sunroof, one overhead console, one rear reading lamp, and eight clip-access slots paired in the four handle pockets. The outer perimeter and every opening remain open mesh boundaries. All formed details belong to the same connected quad surface. There is no Solidify modifier, back surface, rim wall, or cap.

![Refined headliner forming details and metric dimensions](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/headliner-detail.png)

### Headliner control mesh

Orange marks the twelve open boundary loops; turquoise circles mark the four 100 mm suction footprints. The screenshot shows the unsubdivided cage over its own surface. Feature-aligned rows and smoothly rounded corner patches preserve the aperture shoulders under subdivision.

![Headliner quad cage and open hole boundaries](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/headliner-wireframe.png)

### Subdivision mesh

The finer wireframe shows the actual evaluated level-1 subdivision mesh over the smooth level-2 surface. Every opening remains open.

![Subdivided headliner mesh](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/subdivision-wireframe.png)

### Robot mesh

The arm consists of rigid meshes: pedestal, turntable, shoulder casting, upper arm, elbow housing, forearm, wrist, and aluminum vacuum frame. A parented joint hierarchy animates rotations without scaling or deformation. The inspection view overlays the original rigid-part mesh edges on the shaded assembly.

![Robot and gripper mesh wireframes](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/robot-wireframe.png)

## Animation and inspection

Open `scene/headliner_transfer.blend` in **Blender 4.5 LTS** and press Space in the main scene. No add-ons, external textures, or running Python scripts are required for playback. Use the Scene selector to switch between:

1. **01 • HEADLINER TRANSFER** — finished factory cell and animation.
2. **02 • Headliner control topology** — quad cage, open boundaries, and contact footprints.
3. **03 • Subdivision surface topology** — refined surface and evaluated wireframe.
4. **04 • Robot rigid mesh topology** — arm, gripper, and panel wireframes.
5. **05 • Headliner formed details** — shaded close-up with lighting that reveals the molded contours.

The animation has 192 frames at 24 fps (8 seconds). The GIF samples it at 12 fps. Timeline markers identify the station holds, lift, overhead crossing, and return. Analytical two-link inverse kinematics drives the arm, with base azimuth and wrist compensation preserving the horizontal headliner orientation. Joint rotations are baked at every frame.

This is a **kinematic transfer demonstration**: suction stays engaged, including at station holds. It does not simulate release, vacuum pressure, shell deformation, dynamics, or manufacturer joint limits.

## Validation

The builder checks shell topology, subdivided-surface contact positions, the tool path, and fixed link lengths. Results are saved in [geometry.json](validation/geometry.json), with the requested motion in [motion.json](validation/motion.json). The [saved-asset audit](validation/asset-audit.json) additionally checks the connected surface, non-inverted projected faces, suction lips, and receiving-pad contacts. It checks the headliner against the principal arm meshes at every integer frame; it is not comprehensive cell or self-collision certification.

| Check | Result |
|---|---:|
| Control vertices / quads | 8,588 / 8,184 |
| Connected shell surfaces | 1 |
| Open boundary loops | 12: outer perimeter + 11 apertures |
| Edges incident to more than two faces | 0 |
| Euler characteristic | −10 |
| Suction attachments | 4 |
| Tool path and rigid-link errors | < 0.001 mm |
| Suction-center, lip, and receiving-pad contact errors | < 0.001 mm |
| Unit settings | Metric, 1 meter per Blender unit, all object scales applied |

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
blender -b scene/headliner_transfer.blend --python scripts/audit_dimensions.py --python scripts/validate_scene.py
python scripts/package_media.py --gifski /path/to/gifski
```

Rendering uses Cycles with OptiX when available, with a CPU fallback in the render scripts. For a nonstandard font installation, pass `--font-dir /path/to/dejavu` to the packaging script. The animation renderer reuses the symmetric return trajectory when packaging the GIF; the Blender file contains the complete baked return motion. The renderer skips existing animation frames, so clear `build/frames/*.png` after editing geometry before rerendering. `build/` contains disposable intermediate renders and is ignored by Git.

The forming layout is in [headliner_design.py](scripts/headliner_design.py). `DESIGN_SCALE = 0.85` is a baked geometric resize of the first-pass layout, not a change of physical units. Hardware dimensions in that module are specified in final meters. Reference sources and dimensional rationale are recorded in [references/README.md](references/README.md).
