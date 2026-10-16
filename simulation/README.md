# Warp BAC shell and skinning ROM

The full-order and reduced simulations use the same triangulated, curved-rest headliner, the same StVK membrane and Bending-Active Cosserat (BAC) bending energy, the same mass, loading, damping, timestep and rigid contact regions within each layout. Blender is used for geometry export and visualization, not for dynamics.

## Mechanical model

We reimplement the `MidedgeAngleTanFormulation` described in [Better Bending: Analysis, Construction and Verification of Discrete Bending Models for Kirchhoff–Love Shells](https://doi.org/10.1145/3811373), following the equations in [alecjacobson/bending-active-cosserat](https://github.com/alecjacobson/bending-active-cosserat/tree/9299fd987821039047fbf4de204293222cf78751). There is one independent, **massless director angle per mesh edge**, as well as three position coordinates per vertex. Directors participate in the nonlinear minimization; this is not a dihedral-spring surrogate.

For each face, the first fundamental form is the edge Gram matrix. The three curvature measures are `II_i = 2 altitude_i tan(half_dihedral_i + orientation_i theta_edge_i)`. They give `b = [[II_0+II_1, II_0], [II_0, II_0+II_2]]`. Both first and second rest forms are computed from the curved midsurface, with zero rest director angles.

The plane-stress Lamé coefficients are `lambda = E nu/(1-nu²)` and `mu = E/[2(1+nu)]`. For `Q(S) = lambda/2 tr(S)² + mu tr(S²)`, face energies are:

- Membrane: `t A/4 Q(a_rest^-1 (a-a_rest))`.
- Bending: `t³ A/12 Q(a_rest^-1 (b-b_rest))`.

The default illustrative material is **E = 300 MPa, nu = 0.3, density = 250 kg/m³, thickness = 3 mm**. It is not calibrated to an OEM headliner or a layered composite. Mass is area-lumped at vertices. No thickness mesh, cap, or solidification is introduced. All eleven apertures remain open.

`bac.py` evaluates eight energy residuals per face with exact first-order dual arithmetic in Warp. Energy is `0.5 rᵀr`, gradient is `Jᵀr`, and the rest Hessian is `JᵀJ`. `bac_hessian.py` uses mixed second-order dual arithmetic for the consistent nonlinear tangent `JᵀJ + sum r_i Hessian(r_i)`. Both first and second derivatives are analytic Warp computations, not finite differences. The nonlinear solver evaluates the actual energy and gradient, not a linearized bending energy. An Armijo line search rejects collapsed triangles and crossings of the BAC tangent barrier.

## Suction regions and prescribed handles

Each suction cup is a **rigid 100 mm diameter region**, not a point pin. The disk is measured in the headliner’s local XY footprint. A conforming 48-sided circumscribed polygon covers the complete disk; its maximum radial overreach is 0.108 mm. All vertices of every triangle inside the polygon are fixed relative to its cup. All directors strictly inside the rigid region are fixed at rest; rim directors remain variational. The polygon construction avoids locking coarse triangles well beyond the cup boundary.

Every cup has the same prescribed rigid transform in this transfer because all cups belong to one rigid end effector. The surface points within each region inherit that transform, preserving the curved rest surface under the cup. Outside the patches, the shell deforms dynamically. Four-contact mode uses cups 1–4; two-contact mode uses the opposing diagonal cups 1 and 4. Each layout has its own constrained weight basis. Removing two cups does not leave their former regions constrained.

The existing transfer keeps the headliner orientation constant. `carrier(t)` evaluates its original quintic A→B→A translation on the device. The local-frame inertial predictor includes the exact discrete second difference of that translation. This implements prescribed moving supports; it is **not** a two-way robot multibody solve. Rotating carriers would require the corresponding rotating-frame inertial treatment or world-coordinate reconstruction, which is not implemented in this fixed-orientation example.

The robot is held fixed while gravity ramps to 9.81 m/s² over two seconds. Settling uses a 1/120 s timestep and mass-proportional damping of 20 s⁻¹. A device-side gate requires **every vertex's speed below 0.01 mm/s and acceleration below 1 mm/s² for 0.25 s**, after the gravity ramp. If a Newton solve fails or the 30 s settling limit expires, the robot does not move and the run is rejected.

Each model then starts the same four-second transfer from its own settled shape, retaining its tiny residual velocity. Transfer damping is 1 s⁻¹ and gravity remains fully applied. Time zero in the GIFs and Blender comparison scenes is the end of settling; the initial fall is omitted from playback. This deliberately separates preparation from motion, and does not assume the two models have identical gravitational equilibria. Settling trajectories and convergence measurements are saved separately. `--no-settle` restores the old stress-free start with its 0.4 s gravity ramp.

Receiver stands are removed from the simulation scenes. Contact with tooling, suction loss, friction and self-collision remain outside the mechanical model. The four-second A→B trajectory is not assumed to be a periodic dynamic steady state.

## Skinning reduction

`reduction.py` uses [warpack](https://github.com/alecjacobson/warpack/blob/b587ce49951f2944122ce5615dacf537f130799c/benchmarks/applications/SKINNING.md) for constrained scalar weight eigenmodes:

```
H_w W + Cᵀ Lambda = M_w W Omega
C W = 0
Wᵀ M_w W = I
```

`H_w` is the coordinate-block trace of the **position block** of the BAC rest Hessian, with rest directors held fixed during weight design. `M_w` is scalar vertex mass. `C` selects every patch vertex. cuDSS factors the indefinite KKT shift-invert system during setup. Original equilibrium and constraint residuals must pass before proceeding.

The default also adds **18 compliance weights** (two inverse-mass response cycles for three coordinate body loads, taking all three displacement components) and **9 nonlinear-correction weights**. The latter come from the even part of the nonlinear elastic force at opposite unit-acceleration response samples, solved through the shifted rest tangent. This is offline, load-aware enrichment, not additional eigenpairs and not training on the comparison trajectory. It remains entirely Warp/cuDSS. `--enrichment 0` selects the unaugmented benchmark construction; `--no-nonlinear-enrichment` omits the nonlinear correction. The default has 40 eigenweights plus 27 enrichment weights, giving 804 affine coordinates. The optional `--free-directors` flag adds independent director coordinates spanning the lifted director rows; it does not restore every full-model director.

Each weight generates twelve affine displacement columns, `w_j(i) [X_i,Y_i,Z_i,1]` for each output axis. These are mass-orthonormalized in Warp. The attachment entries are explicitly zero, so all reduced coordinates are compatible with the prescribed handle values. This enforces positional attachment, rather than relying on rig-complementarity orthogonality alone. Here the prescribed skinning-handle contribution is the common rigid carrier transform: `x_world(t) = p(t) + x_rest + B q(t)`. Because every active cup shares that transform and every patch row of B is zero, each entire patch follows its prescribed handle exactly. Independent cup motions would require additional compatible particular handle fields.

For each position column, director coordinates are lifted by solving `H_tt D = -H_tx B` at rest with cuDSS. Consequently the ROM has a **rest-linear director lift**. It evaluates the full nonlinear BAC energy on this subspace, but cannot represent every independent nonlinear director change. This is an explicit reduction approximation, and increasing the number of skinning weights does not by itself prove completeness in the combined position/director space. The full simulation retains every free director. The comparison reports the resulting trajectory discrepancy.

The [basis study](../validation/rom-basis-study.json) compares several constructions over the same initial 0.5-second loading trajectory with a softer **30 MPa** material. That exploratory transfer became difficult to solve during motion; the final comparisons use **300 MPa**. Neither value is calibrated. The unaugmented 40-weight basis is much too stiff. Enrichment reduces the discrepancy, but the resulting ROM still does not closely reproduce this formed shell. Increasing to 80 eigenweights improves it only modestly; adding independent director coordinates has little effect in this test. These are measured approximation limits, not solver convergence failures. The final 300 MPa comparison reports its own discrepancy separately.

## Device-only integration

After setup, one CUDA graph contains both settling and the requested transfer:

```
GPU stationary settling loop (including sustained speed/acceleration gate)
If settled: copy state and velocity on device, reset transfer time
GPU transfer timestep loop
  prescribed carrier / inertial predictor
  GPU nonlinear convergence loop
    BAC residuals and exact Jacobian
    GPU preconditioned-CG convergence loop
    GPU Armijo line-search loop
  velocity update, diagnostics, device-resident frame storage
```

The full model uses elementwise consistent-Hessian products and a Warp FSAI preconditioner. Both layouts now use FSAI alone. An optional coarse skinning correction remains available (`--full-preconditioner two-level`), but becomes ineffective during large bends. FSAI numeric factors are refitted to the current tangent on frozen sparse supports inside the captured loop. A GPU guard commits finite positive candidates; rejected candidates leave the previous factors intact. This bypasses the public FSAI update's host-side validation using prevalidated, immutable topology and device-side numeric checks. The exact mechanical operator is independent of this approximate inverse. This accelerates the full solve without restricting its solution to the skinning basis. The truncated Newton-CG option stops at nonpositive curvature and keeps the preceding descent iterate. A device-side descent-direction check falls back to the positive preconditioner if the Newton solve produces a non-descent direction. The ROM projects the same Hessian products and uses a precomputed reduced rest-tangent inverse. Linear iterations, nonlinear convergence, line-search decisions, timestep advancement and output-buffer writes all run on the GPU.

**cuDSS is used only in setup**, avoiding its vendor scalar host-to-device metadata transfers in the simulation graph. There are no NumPy/SciPy/CuPy mechanical solves. Host work consists of mesh topology and file I/O, setup orchestration, graph launch, and post-run validation/rendering. Shapely is an offline mesh-construction dependency. NumPy handles data files and independent validation, not the mechanical runtime.

Both models and layouts use truncated Newton-CG, a maximum of 200 Newton iterations, and relative gradient norm 1e-6 or absolute tolerance 1e-5. The earlier [tolerance study](../validation/rom-tolerance-study.json) found a 2.2 nanometre trajectory change when tightening the four-contact ROM tolerance in the stress-free-start comparison; it is not a new tolerance study of settling. Every settling and transfer timestep must pass its declared stopping criterion. A device-side failure flag stops the captured loop if a solve fails. The driver writes a diagnostic checkpoint and raises instead of accepting the trajectory. The two-contact layout uses a finer transfer timestep to resolve its large bending transitions. The Armijo comparison permits the roundoff margin `1e-12 * max(1 J, abs(E))`; the gradient stopping criterion is unaffected. Recovered line-search failures are reported separately.

The nonlinear energy and tangent evaluations still visit the full mesh: this is a Galerkin ROM, without cubature or hyper-reduction.

The graph audit uses CUDA's verbose DOT dump, including the bodies of nested conditional nodes. It rejects host nodes, host-to-device or device-to-host copies, and unclassified memory copies. Device-to-device copies and memsets are allowed. Startup/JIT, cuDSS factorization, FSAI construction, graph capture and final output downloads are outside simulation timing. The reported graph runtime now includes settling and transfer. Timings are single synchronized runs on a shared NVIDIA L40, not isolated benchmark measurements or a hardware-independent real-time guarantee.

## Reproduction

Use a fresh Python environment, CUDA 12.4+ capable driver, and a supported NVIDIA GPU. The setup script pins Warp 1.18.0, warpack at `6f7cfb61f0e434765fb1a6fd76be241d6ff34d7b` and warp-cuDSS, and installs the unmodified FSAI overlay used by warpack's benchmark.

```bash
bash scripts/setup_shell_env.sh
python -m tests.check_bac
python -m tests.check_hessian
python -m tests.check_device_refit
python -m tests.check_device_stop
python -m tests.check_carrier
python -m tests.check_truncated_cg
python -m tests.check_contacts
python -m tests.check_settling
python -m simulation.run --contacts 4 --weights 40 --steps 480 \
  --dt 0.008333333333333333 --stride 10 --output build/comparison-4
python -m simulation.audit build/comparison-4
python -m tests.check_basis build/comparison-4
python -m simulation.run --contacts 2 --weights 40 --steps 3840 \
  --dt 0.001041666666666667 --stride 80 --output build/comparison-2
python -m simulation.audit build/comparison-2
python -m tests.check_basis build/comparison-2
```

The committed `headliner_mesh.json` is the exported midsurface before contact remeshing. To regenerate it, run Blender 4.5 on `scene/headliner_transfer.blend` with `scripts/export_shell.py`. Both FOM and ROM use the same simulation tessellation, not the much denser display subdivision mesh. Full-order means all mechanical DOFs of that tessellation; it is not a claim of spatial mesh convergence.

Bake and render the trajectories with Blender 4.5, then compose the GIFs:

```bash
blender -b scene/headliner_transfer.blend --python-exit-code 1 \
  --python scripts/render_simulation.py -- --contacts 4 --render
python scripts/package_simulation.py --contacts 4 --gifski /path/to/gifski
# Repeat with --contacts 2.
```

The saved comparison Blender files contain chronological shape-key samples at 12 fps, interpolated linearly during playback. Green faces identify rigid suction regions. Each simulation scene has a following shell camera and a complete robot-cell camera. Deformations are displayed at **1× physical scale**. The original kinematic and mesh-inspection scenes remain available.

## Verification and provenance

`tests/libshell_bac_golden.txt` is the independent LibShell-generated numeric fixture from the reference project's `reference/golden.txt`; it contains the input geometry, deformation, material, energy and full gradient. The fixture is validation data only. No TinyAD, Eigen or LibShell code runs in the solver.

`tests/check_bac.py` checks energy and gradient against that fixture, rigid-motion invariance, curved-rest stress freedom, finite-difference gradients, and the exact rest Hessian. `tests/check_hessian.py` checks the consistent tangent at a significantly deformed configuration. `tests/check_settling.py` verifies that the GPU gate prevents all robot timesteps when settling has not completed, and that successful settling transfers the exact state into the first animation sample. `tests/check_device_stop.py` verifies that an unconverged solve stops the graph after its first attempted timestep. `tests/check_carrier.py` compares prescribed supports with all 193 exported Blender poses. `tests/check_truncated_cg.py` checks positive-definite convergence and negative-curvature truncation. `tests/check_device_refit.py` verifies device-only factor acceptance and rejection. `tests/check_contacts.py` verifies complete disk coverage and activation of every patch vertex in both layouts. `simulation/audit.py` checks topology, patch drift, nonlinear stopping criteria, the full/ROM trajectory discrepancy and captured graph contents. Simulation histories and graph dumps are retained with the published results so solver limits can be inspected.

## Result files

Displacement and ROM-error metrics use the 49 saved samples at 12 fps; convergence histories cover every integration timestep. Relative trajectory error is the mass-weighted norm of `x_ROM - x_full`, divided by the mass-weighted norm of `x_full - x_rest`, over those samples. Both are measured in the carrier-local frame, so the prescribed four-metre rigid translation does not dilute the deformation error. Reported displacement peaks are sampled peaks, not continuous-time maxima.

The new motion-specific metrics instead use `d_full(t)=x_full(t)-x_full(0)` and `d_ROM(t)=x_ROM(t)-x_ROM(0)`, with each initial state gravity-settled. Motion error is `d_ROM-d_full`; its relative mass L2 norm is normalized by `d_full`. These remove the static sag discrepancy from the movement comparison. Absolute discrepancies are still archived, including the difference between initial equilibria. Plots and GIF labels show motion relative to equilibrium.

The [EigenFit assessment](EIGENFIT.md) compares the rest-state full and reduced spectra and explains why a frequency correction has not been applied blindly to this nonlinear ROM.

Each `results/dynamics-N` folder contains the shared mesh, eigenmode diagnostics, full and reduced trajectory arrays, convergence histories, CUDA graph dumps, independent audits and comparison curves. `full-settling.npz` and `rom-settling.npz` retain the settling state, velocity, decimated frames and every-step speed/acceleration measurements. `rom.npz` also stores the basis, with one generalized-coordinate column stored per **row** of the array. The first `3*nv` entries of each row are vertex displacements; the remaining entries are director angles. Samples store local positions and director angles. Recover default ROM coordinates as `q_row = (x-x_rest)_row M B_position^T`; the basis audit verifies both position-subspace membership and the director lift along the saved trajectory. World translation is supplied by `carrier(t)`.

Binary scenes, images, GIFs and NPZ arrays are in Git LFS. Run `git lfs pull` after cloning to obtain the actual files rather than pointer text.
