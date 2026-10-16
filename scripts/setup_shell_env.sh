#!/usr/bin/env bash
set -euo pipefail
# Install into an activated Python environment. Runtime is pinned for graph support.
python -m pip install 'warp-lang @ https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl' \
 'warpack[cudss] @ git+https://github.com/alecjacobson/warpack.git@6f7cfb61f0e434765fb1a6fd76be241d6ff34d7b' \
 'shapely==2.1.2' pillow matplotlib
# Unmodified upstream FSAI implementation; numerical setup and application are Warp.
warp_site=$(python -c 'import pathlib,warp; print(pathlib.Path(warp.__file__).parent)')
for relative in _src/optim/linear.py _src/optim/fsai.py optim/linear.py; do
 curl --fail --location "https://raw.githubusercontent.com/alecjacobson/warp/04427b997fe7ff07bc6ed1dde5ca9813496c6b55/warp/$relative" --output "$warp_site/$relative"
done
