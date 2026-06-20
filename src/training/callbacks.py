"""Training callbacks for Stage 2 runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.utils.io import ensure_parent_dir

try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - only for very minimal environments
    class TrainerCallback:  # type: ignore[no-redef]
        pass


class GpuMemoryCallback(TrainerCallback):
    """Log CUDA memory to Trainer logs and a JSONL sidecar file.

    The class intentionally avoids importing Transformers at module import time
    so lightweight dry runs remain cheap.
    """

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        ensure_parent_dir(self.output_path)

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any):
        if logs is None:
            logs = {}

        try:
            import torch
        except Exception:
            return control

        if not torch.cuda.is_available():
            return control

        device = torch.cuda.current_device()
        allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
        reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated_gb = torch.cuda.max_memory_allocated(device) / 1024**3
        max_reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3

        logs["gpu/allocated_gb"] = allocated_gb
        logs["gpu/reserved_gb"] = reserved_gb
        logs["gpu/max_allocated_gb"] = max_allocated_gb
        logs["gpu/max_reserved_gb"] = max_reserved_gb

        record = {
            "time": time.time(),
            "step": int(getattr(state, "global_step", 0)),
            "allocated_gb": allocated_gb,
            "reserved_gb": reserved_gb,
            "max_allocated_gb": max_allocated_gb,
            "max_reserved_gb": max_reserved_gb,
        }
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
        return control
