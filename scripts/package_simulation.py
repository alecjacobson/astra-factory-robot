"""Side-by-side GIFs from true chronological full/ROM simulation frames."""

import argparse, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--contacts", type=int, required=True)
p.add_argument("--gifski", default="gifski")
p.add_argument("--folder")
args = p.parse_args()
folder = Path(args.folder or ROOT / f"build/comparison-{args.contacts}")
report = json.loads((folder / "comparison.json").read_text())
curves = np.load(folder / "comparison_curves.npz")
out = folder / "gif-frames"
out.mkdir(exist_ok=True)
font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def f(size, b=False):
    return ImageFont.truetype(bold if b else font, size)


for j in range(len(curves["time"])):
    canvas = Image.new("RGB", (1480, 904), "#101b27")
    d = ImageDraw.Draw(canvas)
    d.text(
        (22, 15),
        f"{args.contacts} RIGID SUCTION PATCHES  /  NONLINEAR BAC SHELL",
        font=f(27, True),
        fill="#edf3f5",
    )
    material = report["full"]["material"]
    d.text((1115, 17), "ILLUSTRATIVE MATERIAL", font=f(15, True), fill="#b9c7d5")
    d.text(
        (1115, 40),
        f"{material['young_Pa'] / 1e6:g} MPa · {material['thickness_m'] * 1000:g} mm · {material['density_kg_m3']:g} kg/m³",
        font=f(14),
        fill="#b9c7d5",
    )
    d.text((22, 54), "FULL SIMULATION", font=f(22, True), fill="#56c9bf")
    d.text((762, 54), "ENRICHED SKINNING ROM", font=f(22, True), fill="#f2bd5b")
    for model, x in [("full", 10), ("rom", 750)]:
        canvas.paste(
            Image.open(folder / (model + "-frames") / f"{j:04d}.png")
            .convert("RGB")
            .resize((720, 720)),
            (x, 92),
        )
    d = ImageDraw.Draw(canvas)
    phase_label = (
        "gravity settled" if report["full"].get("settling") else "stress-free start"
    )
    d.text(
        (22, 821),
        f"Transfer t = {curves['time'][j]:.2f} s    |    {phase_label}    |    physical scale 1×",
        font=f(20),
        fill="#e1e9ef",
    )
    reference_label = (
        "equilibrium" if report["full"].get("settling") else "initial state"
    )
    d.text(
        (22, 858),
        f"Motion from {reference_label}: full {curves['motion_full_peak'][j] * 1000:.2f} / ROM {curves['motion_rom_peak'][j] * 1000:.2f} mm",
        font=f(19),
        fill="#b9c7d5",
    )
    d.text(
        (1030, 858),
        f"{report['rom']['basis_dofs']} vs {report['full']['free_dofs']:,} free DOFs",
        font=f(19),
        fill="#b9c7d5",
    )
    canvas.save(out / f"{j:04d}.png")
    if j == len(curves["time"]) // 2:
        canvas.save(ROOT / f"media/shell-comparison-{args.contacts}-contacts.png")
paths = sorted(out.glob("*.png"))
subprocess.run(
    [
        args.gifski,
        "--fps",
        "12",
        "--quality",
        "85",
        "--repeat",
        "0",
        "--width",
        "1200",
        "--output",
        str(ROOT / f"media/shell-comparison-{args.contacts}-contacts.gif"),
        *[str(x) for x in paths],
    ],
    check=True,
)
