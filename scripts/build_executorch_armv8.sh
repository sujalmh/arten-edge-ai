#!/usr/bin/env bash
# Build ExecuTorch 1.1.0 for ARMv8.0 (Raspberry Pi 3 / Pi 4 / CM4).
#
# The upstream aarch64 wheels are compiled for ARMv8.2 and use `sdot`, which
# traps on Cortex-A53/A72. This builds the same runtime with `-march=armv8-a`,
# KleidiAI off and the XNNPACK dot-product/i8mm kernels disabled.
#
# Requirements: an arm64 Linux host (an Apple Silicon Mac with Docker works;
# the container runs natively, not under emulation).
#
# Usage:
#   bash scripts/build_executorch_armv8.sh
#   cp wheels/executorch-*.whl vendor/executorch-armv8/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:-$HOME/.cache/arten-executorch-armv8}"
OUT="${OUT:-$ROOT/wheels}"
SRC="$WORK/executorch"

mkdir -p "$WORK" "$OUT"

if [[ ! -d "$SRC/.git" ]]; then
  echo "== cloning ExecuTorch v1.1.0 =="
  git clone --depth 1 --branch v1.1.0 \
    --recurse-submodules --shallow-submodules \
    https://github.com/pytorch/executorch.git "$SRC"
fi

echo "== patching the export-stack import =="
python3 - "$SRC/extension/pybindings/portable_lib.py" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
if "_EXPERIMENTAL_WARNING" not in text:
    old = "import executorch.exir._warnings as _exir_warnings\n"
    new = (
        "try:\n"
        "    import executorch.exir._warnings as _exir_warnings\n"
        "\n"
        "    _EXPERIMENTAL_WARNING = _exir_warnings.ExperimentalWarning\n"
        "except Exception:  # exir needs torch>=2.10; the runtime does not\n"
        "    _EXPERIMENTAL_WARNING = UserWarning\n"
    )
    text = text.replace(old, new)
    text = text.replace("_exir_warnings.ExperimentalWarning", "_EXPERIMENTAL_WARNING")
    text = text.replace("del _exir_warnings", "del _EXPERIMENTAL_WARNING")
    path.write_text(text)
print("patched", path)
PY

echo "== reordering torch headers before the portable c10 copy =="
python3 - "$SRC/CMakeLists.txt" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()

# 1. Torch headers must win over ExecuTorch's header-only c10 copy.
old = "util PUBLIC ${_common_include_directories} ${TORCH_INCLUDE_DIRS}"
new = "util PUBLIC ${TORCH_INCLUDE_DIRS} ${_common_include_directories}"
if old in text:
    text = text.replace(old, new)

# 2. Disabling the optimized kernels also skips find_package_torch_headers();
#    the pybind target still needs TORCH_INCLUDE_DIRS.
old_block = (
    "if(EXECUTORCH_BUILD_KERNELS_OPTIMIZED)\n"
    "  # find pytorch lib here to make it available to all sub-directories. Find it\n"
    "  # before including portable so that optimized_portable_kernels can use it.\n"
    "  find_package_torch_headers()\n"
    "endif()"
)
new_block = (
    "if(EXECUTORCH_BUILD_KERNELS_OPTIMIZED OR EXECUTORCH_BUILD_PYBIND)\n"
    "  # find pytorch lib here to make it available to all sub-directories. Find it\n"
    "  # before including portable so that optimized_portable_kernels can use it.\n"
    "  find_package_torch_headers()\n"
    "endif()"
)
if old_block in text:
    text = text.replace(old_block, new_block)

path.write_text(text)
print("patched", path)
PY

echo "== writing the container build script =="
cat > "$WORK/build_in_container.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq build-essential cmake ninja-build git \
  python3 python3-venv python3-dev python3-pip pkg-config

python3 -m venv /work/venv
. /work/venv/bin/activate
python -m pip install -q --upgrade pip setuptools wheel "cmake>=3.29" ninja
python -c "import shutil; print('cmake:', shutil.which('cmake'))"
python -m pip install -q --index-url https://pypi.org/simple torch==2.9.0
python -m pip install -q "numpy<2" pyyaml packaging typing-extensions

cd /work/executorch
rm -rf cmake-out pip-out build executorch.egg-info
export CMAKE_ARGS="-DEXECUTORCH_BUILD_XNNPACK=ON \
-DEXECUTORCH_BUILD_PYBIND=ON \
-DEXECUTORCH_BUILD_KERNELS_LLM=OFF \
-DEXECUTORCH_BUILD_KERNELS_LLM_AOT=OFF \
-DEXECUTORCH_XNNPACK_ENABLE_KLEIDI=OFF \
-DXNNPACK_ENABLE_ARM_DOTPROD=OFF \
-DXNNPACK_ENABLE_ARM_I8MM=OFF \
-DCMAKE_NO_SYSTEM_FROM_IMPORTED=ON \
-DCMAKE_C_FLAGS=-march=armv8-a \
-DCMAKE_CXX_FLAGS=-march=armv8-a \
-DCMAKE_BUILD_TYPE=Release"
python -m pip wheel . --no-deps --no-build-isolation -w /out -v
EOF

echo "== building in an arm64 Debian container =="
docker run --rm -v "$WORK:/work" -v "$OUT:/out" -w /work \
  debian:bookworm bash /work/build_in_container.sh

echo "== done =="
ls -lh "$OUT"
echo
echo "Install on the Pi with:"
echo "  mkdir -p $ROOT/vendor/executorch-armv8"
echo "  cp $OUT/executorch-*.whl $ROOT/vendor/executorch-armv8/"
echo "  bash scripts/setup_pi.sh"
