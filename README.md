# HL / 04 — Robotic headliner transfer

A factory robot transfers an automobile headliner between stations using four suction contacts or two opposing diagonal contacts. Both layouts share the headliner, cup size, tooling frame and prescribed robot motion.

## Settled dynamic shell comparison

The receiver stands are removed from the simulation scenes. The robot stays fixed while the shell settles under gravity, and only then begins the transfer. **The GIFs start after settling**, so the initial fall no longer dominates the motion. Labels and plots measure deformation relative to each model's own settled shape, at **1× physical scale**.

Each suction attachment rigidly constrains the **entire 100 mm diameter circular region**. The full-order and enriched skinning-eigenmode models use a curved-rest Bending-Active Cosserat shell implemented in Warp, with Warp/cuDSS precomputation. Both settling and transfer run inside **one device-resident CUDA graph**. The gate requires every vertex's speed below 0.01 mm/s and acceleration below 1 mm/s² for 0.25 s after the gravity ramp; the robot cannot move if settling fails. [Equations, implementation and reproduction](simulation/README.md).

**Four rigid suction regions — full simulation / enriched skinning ROM**

![Four-contact settled shell comparison](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/shell-comparison-4-contacts.gif)

**Two rigid suction regions — opposing diagonal cups 01 and 04**

![Two-contact settled shell comparison](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/shell-comparison-2-contacts.gif)

| Patches | Settling time, full / ROM | Peak motion from equilibrium, full / ROM | Maximum RMS motion error | Relative motion error |
|---|---:|---:|---:|---:|
| 4 | 2.32 / 2.31 s | 1.92 / 1.62 mm | 0.072 mm | 15.3% |
| 2 | 5.92 / 2.79 s | 352.98 / 40.07 mm | 109.332 mm | 88.1% |

Errors compare `x(t)-x(0)` between models over 49 saved transfer samples in the gripper-local frame. The full and ROM models use the same timestep within each layout: 1/120 s for four patches, 1/960 s for two. Every settling and transfer timestep converged; patch drift is zero. Nested graph audits found no host nodes or host/device transfers during simulation. [Four-contact results and curves](results/dynamics-4/comparison.json) · [Two-contact results and curves](results/dynamics-2/comparison.json).

**The two-contact ROM remains too stiff.** Its initial peak gravitational sag is 73.0 mm, compared with 345.1 mm in the full model. Separating settling from transfer makes the motion-induced discrepancy measurable, but does not remove this reduction error. The ROM has 804 coordinates; the full models have 11,867 and 12,245 free DOFs.

Material properties remain illustrative: E = 300 MPa, Poisson ratio 0.3, density 250 kg/m³, thickness 3 mm; shell mass approximately 1.675 kg. Settling uses damping of 20 s⁻¹; transfer uses 1 s⁻¹. The model excludes suction failure, release, self-collision and tooling contact, and the shared simulation mesh has not undergone a spatial convergence study.

**EigenFit assessment.** A new Warp/cuDSS spectral diagnostic compares twelve full and reduced rest-state modes. The two-contact model's lowest three frequencies already agree within 0.55%; higher modes show stiffening and weaker mode-shape correspondence. The [assessment and measurements](simulation/EIGENFIT.md) explain why this does not yet justify a nonlinear correction. These animations retain the original shell energy; no spectral correction or Young's-modulus adjustment has been applied.

[Four-contact Blender scene](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/scene/headliner_dynamics_4_contacts.blend) · [Two-contact Blender scene](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/scene/headliner_dynamics_2_contacts.blend)

[Four-contact motion plot](results/dynamics-4/comparison.png) · [Two-contact motion plot](results/dynamics-2/comparison.png)

The simulation mesh has 2,172 vertices, 3,925 triangles and eleven open holes. Conforming 48-sided polygons cover the suction disks with at most 0.108 mm radial overreach. Green faces mark rigid contact regions. The original subdivision surface and robot wireframe scenes remain available in the Blender files.

![BAC shell wireframe and complete suction regions](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/bac-shell-wireframe.png)

## Original kinematic scene

**Four-contact baseline**

![Animated headliner transfer](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/headliner-transfer.gif)

**Two-contact comparison — cups 01 and 04**

![Two-contact headliner transfer](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/headliner-transfer-2-contacts.gif)

The two-contact GIF is a geometric, kinematic comparison prepared for later stability analysis. It does not predict wobble, slipping, loss of suction, or shell deformation.

**[Download the Blender scene](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/scene/headliner_transfer.blend)** · **[Animation GIF](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/headliner-transfer.gif)** · **[Two-contact GIF](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/headliner-transfer-2-contacts.gif)** · **[Visual and dimensional references](references/README.md)**

![Factory cell overview](https://media.githubusercontent.com/media/alecjacobson/astra-factory-robot/main/media/hero.png)

## Physical dimensions

**One Blender unit = one meter**, in every scene. The second pass resizes the cell geometry to a representative passenger-car headliner and long-reach industrial arm. The resize is baked into mesh data and local translations; all object scales remain `(1, 1, 1)`. Dimensions below are measured from the saved geometry, rather than inferred from labels. The complete measurement report is [dimensions.json](validation/dimensions.json).

| Feature | Final size |
|---|---:|
| Headliner envelope, width × length | approximately **1,421 × 1,870 mm** |
| Headliner form depth, highest to lowest surface point | approximately **165 mm** |
| Nominal material thickness | **3 mm**, constitutive thickness in the simulation |
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

## Selecting two or four suction contacts

Use Blender's **Scene dropdown** in the top bar:

| Scene | Active suction contacts |
|---|---|
| **01 • HEADLINER TRANSFER** | Four corners: 01, 02, 03, 04 |
| **06 • TWO-CONTACT TRANSFER** | Opposing diagonal pair: 01 and 04 |

Press Space to play, or F12 to render the selected configuration. Both options are in the same `.blend`; switching requires no scripts, add-ons, drivers, or automatic Python execution. The in-file text **START HERE • contact configurations** repeats these instructions.

The two-contact scene excludes the complete optional 02/03 assembly collection: bellows cups, stems, collars, fittings, contact markers, and hoses. The rectangular frame is retained, with two empty attachment locations. The four passive pads on each stationary receiving fixture are unchanged. Existing topology inspection scenes show the four-contact reference layout.

Both layouts keep the headliner rigidly attached to the tool. The reduced-contact arrangement is provided for examination; the prescribed motion does not establish mechanical stability. Active contact IDs, local contact coordinates, cup diameter, and verification results are in [contact-configurations.json](validation/contact-configurations.json).

## Animation and inspection

Open `scene/headliner_transfer.blend` in **Blender 4.5 LTS** and press Space in the main scene. No add-ons, external textures, or running Python scripts are required for playback. Use the Scene selector to switch between:

1. **01 • HEADLINER TRANSFER** — finished factory cell and animation.
2. **02 • Headliner control topology** — quad cage, open boundaries, and contact footprints.
3. **03 • Subdivision surface topology** — refined surface and evaluated wireframe.
4. **04 • Robot rigid mesh topology** — arm, gripper, and panel wireframes.
5. **05 • Headliner formed details** — shaded close-up with lighting that reveals the molded contours.
6. **06 • TWO-CONTACT TRANSFER** — the same transfer with only diagonal cups 01 and 04.

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
| Suction attachments | Selectable 4 or 2 |
| Tool path and rigid-link errors | < 0.001 mm |
| Suction-center, lip, and receiving-pad contact errors | < 0.001 mm |
| Unit settings | Metric, 1 meter per Blender unit, all object scales applied |

The [configuration audit](validation/contact-configurations.json) separately verifies both layouts for all 192 frames: exactly four or two visible suction assemblies and hoses, correct center and lip contact, and identical evaluated headliner transforms.

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
blender -b scene/headliner_transfer.blend --python scripts/render.py -- animation-2
blender -b scene/headliner_transfer.blend --python scripts/render.py -- two-contact-hero
blender -b scene/headliner_transfer.blend --python scripts/render.py -- stills
blender -b scene/headliner_transfer.blend --python scripts/audit_dimensions.py --python scripts/validate_scene.py
blender -b scene/headliner_transfer.blend --python scripts/validate_contact_configurations.py
python scripts/package_media.py --gifski /path/to/gifski
python scripts/package_media.py --contacts 2 --gifski /path/to/gifski
```

Rendering uses Cycles with OptiX when available, with a CPU fallback in the render scripts. For a nonstandard font installation, pass `--font-dir /path/to/dejavu` to the packaging script. The animation renderer reuses the symmetric return trajectory when packaging the GIF; the Blender file contains the complete baked return motion. The renderer skips existing animation frames, so clear `build/frames/*.png` and `build/frames-2/*.png` after editing geometry before rerendering. `build/` contains disposable intermediate renders and is ignored by Git.

The forming layout is in [headliner_design.py](scripts/headliner_design.py). `DESIGN_SCALE = 0.85` is a baked geometric resize of the first-pass layout, not a change of physical units. Hardware dimensions in that module are specified in final meters. Reference sources and dimensional rationale are recorded in [references/README.md](references/README.md).

Contact layouts and native scene exclusions are defined in [contact_configurations.py](scripts/contact_configurations.py). The four-contact scene is the default when the file opens.
