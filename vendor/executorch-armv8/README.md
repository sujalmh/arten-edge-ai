# ExecuTorch ARMv8.0 runtime

This directory holds the ExecuTorch 1.1.0 wheel used on Raspberry Pi 3 and
Pi 4. Those boards are ARMv8.0 (Cortex-A53/A72). The upstream aarch64 wheels
are compiled for ARMv8.2 and contain `sdot` (dot-product) instructions, which
trap with `Illegal instruction` on ARMv8.0.

The wheel here is built from the `v1.1.0` tag with:

- `-march=armv8-a`
- `EXECUTORCH_XNNPACK_ENABLE_KLEIDI=OFF`
- `XNNPACK_ENABLE_ARM_DOTPROD=OFF`
- `XNNPACK_ENABLE_ARM_I8MM=OFF`
- `EXECUTORCH_BUILD_KERNELS_LLM=OFF` and
  `EXECUTORCH_BUILD_KERNELS_LLM_AOT=OFF` (not needed for detection)
- PyTorch **2.9.0**, the last release that runs on ARMv8.0
  (2.10 uses ARMv8.2 `sdot` and crashes with SIGILL)
- torch headers ordered ahead of ExecuTorch's header-only `c10` copy
- `extension/pybindings/portable_lib.py` patched so the runtime does not import
  `executorch.exir` (the export stack needs torch 2.10; inference does not)

Rebuild with:

```bash
bash scripts/build_executorch_armv8.sh
cp wheels/executorch-*.whl vendor/executorch-armv8/
```

Pi 5 does not use this wheel: `scripts/setup_pi.sh` installs the prebuilt
ARMv8.2 runtime there instead.

## Wheel

```text
executorch-1.1.0a0+17adba1-cp311-cp311-linux_aarch64.whl
sha256 055a31f34fbbff4747493885401a478857c37008b2d9517cb73a48de32f2b6f7
```

Verified on a Raspberry Pi 3B+ (Cortex-A53, 64-bit Bookworm): no `sdot` in the
compiled binding, model loads, and a one-frame run produces detections.
