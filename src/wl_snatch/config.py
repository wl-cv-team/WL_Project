"""Detector settings, weight names and device choice. Imports no model library."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# yolo26n.pt was confirmed in the official Ultralytics docs. The other two are
# standard Ultralytics names that were not confirmed in an official table (see
# docs/decisions/0001-environment.md). All three are COCO-pretrained starting
# points for training, not plate detectors: they do not detect plates.
WEIGHT_NAMES = {"yolo26": "yolo26n.pt", "yolo11": "yolo11n.pt", "yolov8": "yolov8n.pt"}

_DEVICES = ("cpu", "mps", "cuda")


@dataclass(frozen=True)
class DetectorConfig:
    model_name: str
    imgsz: int = 640
    conf: float = 0.25
    device: str = "auto"
    seed: int = 0
    epochs: int = 50
    # Trained plate-detector weights to run. `model_name` stays the architecture
    # name (for records and for `weight_name`, which training will use).
    weights_path: Path | str | None = None

    def __post_init__(self) -> None:
        if self.weights_path is not None and (
            not isinstance(self.weights_path, (Path, str)) or not str(self.weights_path)
        ):
            raise ValueError(
                f"weights_path must be None or a non-empty path, got {self.weights_path!r}"
            )
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError(f"model_name must be a non-empty string, got {self.model_name!r}")
        for field in ("imgsz", "epochs"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field} must be an int, got {value!r}")
            if value <= 0:
                raise ValueError(f"{field} must be positive, got {value}")
        if isinstance(self.conf, bool) or not isinstance(self.conf, (int, float)):
            raise ValueError(f"conf must be a number, got {self.conf!r}")
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError(f"conf must be between 0 and 1, got {self.conf}")


def weight_name(model_name: str) -> str:
    """Weight file name for a model name; ValueError listing the known names otherwise."""
    try:
        return WEIGHT_NAMES[model_name]
    except KeyError:
        known = ", ".join(sorted(WEIGHT_NAMES))
        raise ValueError(f"unknown model name {model_name!r}; known names: {known}") from None


def resolve_device(
    requested: str, mps_available: bool, cuda_available: bool
) -> tuple[str, str | None]:
    """Device to use and, when it differs from the request, the reason.

    "auto" picks mps, then cuda, then cpu, and gives no reason. A named device
    that is not available falls back to cpu with a reason.
    """
    if requested == "auto":
        if mps_available:
            return "mps", None
        if cuda_available:
            return "cuda", None
        return "cpu", None
    kind = "cuda" if re.fullmatch(r"cuda:\d+", requested) else requested
    if kind not in _DEVICES:
        raise ValueError(
            f"unknown device {requested!r}; use auto, cuda:<index> or one of "
            f"{', '.join(_DEVICES)}"
        )
    available = {"cpu": True, "mps": mps_available, "cuda": cuda_available}
    if available[kind]:
        return requested, None
    return "cpu", f"{requested} requested but not available; using cpu"
