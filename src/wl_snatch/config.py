"""Detector settings, weight names and device choice. Imports no model library."""

from __future__ import annotations

import re
from dataclasses import dataclass

# yolo26n.pt is listed in the Ultralytics YOLO26 docs. The other two are the
# standard Ultralytics names; see docs/decisions/0001-environment.md.
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

    def __post_init__(self) -> None:
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
