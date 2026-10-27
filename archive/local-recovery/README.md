# Recovered local work

This archive preserves local material found during the repository deletion audit
on 2026-10-27. `manifest.json` records original paths or object IDs, byte counts,
and SHA-256 checksums. File contents are preserved without modification. Names
assigned to unreachable objects describe their contents; their original paths
are not known with certainty.

- `experiments/`: three handwritten preconditioner experiments from ignored `build/`.
- `inputs/`: the two saved failure states read by those experiments.
- `drafts/`: the ignored README draft and unreachable source/documentation revisions.
- `results/`: three small result records recovered from unreachable Git blobs.
- `scenes/`: the unique Blender backup and three unreferenced local LFS scenes.

The Blender scenes and NumPy inputs use the repository's existing Git LFS rules.
Run `git lfs pull` after cloning to retrieve them. The README-with-two draft also
preserves one unreachable blob; one recovered scene preserves the payload of an
unreachable LFS pointer. The manifest identifies these without duplicating content.

The archived Python files are historical experiments, not maintained entry points.
Their original working-directory paths are recorded in the manifest. They may
require an earlier simulation implementation; no runtime compatibility is claimed.

Caches, intermediate renders, simulation outputs, and logs remain excluded.
The initial-pass Blender scene and the two dynamics `.blend1` backups already
have identical copies in committed history. The downloaded EigenFit paper is
identified by its source link in `simulation/EIGENFIT.md`; its extracted text is
derived from that paper. Neither is included here.
