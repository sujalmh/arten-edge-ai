"""ExecuTorch model loading and inference.

The runtime is imported lazily so `--help`, unit tests and report generation do
not need a 900 MB virtualenv. Inference timing here is model-only: the caller
measures preprocessing and postprocessing separately.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .constants import (
    INPUT_SIZE,
    KNOWN_MODELS,
    PI5_REFERENCE,
)
from .metrics import environment_info, file_size_mb, sha256_file


class ModelLoadError(RuntimeError):
    """Raised when the .pte cannot be loaded or executed."""


class ExecuTorchDetector:
    """A thin wrapper around an ExecuTorch program's `forward` method."""

    def __init__(
        self,
        model_path: str | Path,
        threads: int | None = None,
        warmup: int = 3,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise ModelLoadError(
                f"model not found: {self.model_path}\n"
                "Run `bash scripts/download_model.sh` or point --model at a .pte file."
            )

        try:
            import torch
            from executorch.runtime import Runtime
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelLoadError(
                "ExecuTorch/PyTorch is not installed. On the Pi run:\n"
                "  bash scripts/setup_pi.sh\n"
                f"Original error: {exc}"
            ) from exc

        self._torch = torch
        self.threads = threads
        if threads is not None and threads > 0:
            torch.set_num_threads(int(threads))

        started = time.perf_counter()
        try:
            self._runtime = Runtime.get()
            self._program = self._runtime.load_program(str(self.model_path))
            self._method = self._program.load_method("forward")
        except Exception as exc:  # pragma: no cover - runtime dependent
            raise ModelLoadError(
                f"could not load {self.model_path.name} with ExecuTorch: {exc}\n"
                "The prebuilt model targets ExecuTorch 1.1.0; a newer runtime "
                "usually works, but pin 1.1.0 if you see operator/schema errors."
            ) from exc
        self.load_ms = round((time.perf_counter() - started) * 1000, 2)

        self.model_size_mb = file_size_mb(self.model_path)
        self.sha256 = sha256_file(self.model_path)
        self.model_entry = KNOWN_MODELS.get(self.sha256)
        if self.model_entry:
            self.model_name = self.model_entry["name"]
            self.model_description = self.model_entry["description"]
            self.model_source = self.model_entry["source"]
            self.model_reference = self.model_entry["reference"]
            self.expected_sha256 = self.sha256
        else:
            self.model_name = f"{self.model_path.stem} (custom .pte)"
            self.model_description = (
                "Custom ExecuTorch .pte; no matching Arm model card."
            )
            self.model_source = None
            self.model_reference = PI5_REFERENCE
            self.expected_sha256 = None
        try:
            from importlib.metadata import version

            self.executorch_version = version("executorch")
        except Exception:  # pragma: no cover
            self.executorch_version = "unknown"
        self.torch_version = torch.__version__

        if warmup > 0:
            self.warmup(warmup)

    def warmup(self, runs: int = 3) -> None:
        dummy = np.zeros((3, INPUT_SIZE[0], INPUT_SIZE[1]), dtype=np.float32)
        for _ in range(max(1, runs)):
            self.infer(dummy)

    def infer(self, chw: np.ndarray) -> tuple[np.ndarray, float]:
        """Run one frame. Returns (raw output, model latency in ms)."""

        if chw.shape != (3, INPUT_SIZE[0], INPUT_SIZE[1]):
            raise ValueError(
                f"expected input {(3, INPUT_SIZE[0], INPUT_SIZE[1])}, got {chw.shape}"
            )
        tensor = self._torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
        started = time.perf_counter()
        outputs = self._method.execute([tensor])
        latency_ms = (time.perf_counter() - started) * 1000.0

        if not outputs:
            raise ModelLoadError("model returned no outputs")
        out = outputs[0]
        if hasattr(out, "detach"):
            out = out.detach().cpu().numpy()
        else:
            out = np.asarray(out)
        return out, latency_ms

    def info(self) -> dict:
        env = environment_info(self.threads)
        env.update(
            {
                "model": {
                    "name": self.model_name,
                    "description": self.model_description,
                    "path": str(self.model_path),
                    "size_mb": self.model_size_mb,
                    "sha256": self.sha256,
                    "expected_sha256": self.expected_sha256,
                    "sha256_match": (
                        self.sha256 == self.expected_sha256
                        if self.expected_sha256
                        else None
                    ),
                    "format": "ExecuTorch .pte",
                    "backend": "XNNPACK (prebuilt with Arm KleidiAI kernels)",
                    "quantization": (
                        "INT8 static post-training quantization; stem conv and "
                        "detection head remain FP32 (per Arm model card)"
                    ),
                    "input_shape": [1, 3, INPUT_SIZE[0], INPUT_SIZE[1]],
                    "input_range": [0.0, 1.0],
                    "classes": 80,
                    "dataset": "COCO 2017",
                    "source": self.model_source,
                    "load_ms": self.load_ms,
                    "torch": self.torch_version,
                    "executorch": self.executorch_version,
                },
                "reference": self.model_reference,
            }
        )
        return env
