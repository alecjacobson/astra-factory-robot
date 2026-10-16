"""Archive audited trajectories and graph evidence for Git LFS publication."""

import argparse, shutil
from pathlib import Path
from simulation.audit import audit

p = argparse.ArgumentParser()
p.add_argument("--contacts", type=int, required=True)
p.add_argument("--folder")
args = p.parse_args()
source = Path(args.folder or f"build/comparison-{args.contacts}")
target = Path(f"results/dynamics-{args.contacts}")
report = audit(source)
assert all(not report[m]["debug_host_checkpoints"] for m in ["full", "rom"])
target.mkdir(parents=True, exist_ok=True)
for name in [
    "mesh.json",
    "modes.json",
    "full.json",
    "rom.json",
    "full.npz",
    "rom.npz",
    "full-graph.dot",
    "rom-graph.dot",
    "comparison.json",
    "comparison_curves.npz",
    "comparison.png",
    "basis-audit.json",
]:
    shutil.copy2(source / name, target / name)
print(target)

for model in ["full", "rom"]:
    if report[model].get("settling"):
        for extension in ["json", "npz"]:
            name = f"{model}-settling.{extension}"
            shutil.copy2(source / name, target / name)
