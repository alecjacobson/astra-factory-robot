# EigenFit assessment for this headliner

[EigenFit, Chen et al., SCA 2019](https://dannykaufman.io/projects/eigen-fit.pdf) is a relevant reference for spectral stiffening. Its linear construction replaces selected coarse eigenvalues using fine-mesh values. The nonlinear method retains rest-state eigenvalue ratios but updates the low-mode subspace at each timestep, correcting both elastic force and tangent stiffness (equations 15a–c). It also treats indefinite element tangents. Applying a stiffness-only correction inside our existing energy line search would not implement that method consistently.

Here the reference and ROM share the **same mesh, mass, material and constraints**. The difference is the admissible deformation space, including the ROM's rest-linear director lift. EigenFit is therefore a possible adaptation to reduction error, rather than evidence of mesh-resolution error in this comparison.

## What we measured

`simulation/spectral_study.py` computes twelve small-vibration modes of the full shell and the existing 804-coordinate ROM at the stress-free rest shape. All eigensolves run in Warp with cuDSS shifted inverses. The full problem eliminates the massless directors implicitly, using the mixed inverse, and operates in mass-normalized free vertex coordinates. No artificial director mass is introduced. Original-problem residuals are checked independently after the inverse eigensolve.

For two patches, the first three full/ROM frequencies are:

| Mode | Full (Hz) | ROM (Hz) | Suggested stiffness ratio λ_full / λ_ROM | Mass-weighted shape correlation squared |
|---|---:|---:|---:|---:|
| 1 | 2.73323 | 2.73337 | 0.999903 | 0.999995 |
| 2 | 3.41475 | 3.42494 | 0.994058 | 0.999809 |
| 3 | 8.65599 | 8.70286 | 0.989257 | 0.994703 |
| 4 | 10.95011 | 12.94993 | 0.714993 | 0.541416 |

The lowest three frequencies already agree within 0.55%, despite the [previous two-patch transfer's 79% trajectory discrepancy](https://github.com/alecjacobson/astra-factory-robot/blob/921ad1d/results/dynamics-2/comparison.json). Higher modes have substantial spectral error, but sorted full/ROM eigenvectors also cease to correspond reliably. For example, the fifth ordered pair has squared correlation only 0.0061. Pairing those modes solely by sorted eigenvalue would require further justification.

For four patches, the lowest two frequencies agree within 0.39%; mode-shape correspondence deteriorates from the third ordered pair onward. Full diagnostics, including projection losses and residuals, are in [two-patch results](../validation/spectral-study-2.json) and [four-patch results](../validation/spectral-study-4.json).

These measurements establish that there is higher-mode Ritz stiffening. They **do not establish that a rest-state spectral correction fixes the nonlinear two-patch error**. In particular, matching eigenvalues cannot add missing deformation directions or relax the imposed linear relationship between positions and directors. The new animations therefore retain the original material and shell energy and explicitly label the ROM discrepancy. No EigenFit correction or scalar Young's-modulus adjustment has been silently applied.

A position-only fit is not an energy fit. As an additional diagnostic, `simulation/settled_projection.py` projects the settled full state into the ROM's mass-orthonormal position space, then evaluates the actual Warp shell energy both with the full directors and with the ROM director lift. This tests the nonlinear admissibility of that particular projected state; it does not prove that no other ROM coordinates could fit the reference.

For the two-patch settled shape, that position fit has only **0.445 mm mass-weighted RMS error**, but retaining the full directors raises elastic energy from **0.930 J to 4,147 J**. Imposing the ROM director lift on the same projected positions crosses the BAC admissibility barrier. Thus a small positional fit error does not imply a low-strain or admissible shell state. This is evidence of a nonlinear approximation problem beyond the already accurate lowest three rest frequencies, not proof that another reduced state cannot fit better. [Two-patch projection diagnostic](../validation/settled-projection-2.json) · [Four-patch diagnostic](../validation/settled-projection-4.json).

A nonlinear correction would need mode correspondence under deformation, force/tangent-consistent integration, and comparison against the settled full simulation. The present rest-state diagnostic is a reproducible first check, not an implementation or validation of nonlinear EigenFit.

## Reproduce

Use the Warp/warpack environment described in [README.md](README.md):

```bash
python -m simulation.spectral_study --contacts 2
python -m simulation.spectral_study --contacts 4
python -m simulation.settled_projection --contacts 2 --folder results/dynamics-2
python -m simulation.settled_projection --contacts 4 --folder results/dynamics-4
```

By default the script reads the saved ROM basis from `results/dynamics-N/rom.npz`; use `--folder` to select another result directory. NumPy is used for file I/O and independent post-solve comparisons, not the eigensolves.
