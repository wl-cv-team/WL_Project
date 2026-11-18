import pytest

from wl_snatch.config import DetectorConfig, resolve_device, weight_name


def test_config_rejects_nonpositive_imgsz_and_epochs():
    assert DetectorConfig(model_name="yolo26").imgsz == 640
    for field in ("imgsz", "epochs"):
        for value in (0, -1):
            with pytest.raises(ValueError, match=field):
                DetectorConfig(model_name="yolo26", **{field: value})


def test_config_rejects_conf_outside_zero_one():
    for value in (-0.01, 1.01):
        with pytest.raises(ValueError, match="conf"):
            DetectorConfig(model_name="yolo26", conf=value)
    assert DetectorConfig(model_name="yolo26", conf=0.0).conf == 0.0
    assert DetectorConfig(model_name="yolo26", conf=1.0).conf == 1.0


def test_weight_name_maps_known_models_and_rejects_unknown():
    assert weight_name("yolo26") == "yolo26n.pt"
    assert weight_name("yolo11") == "yolo11n.pt"
    assert weight_name("yolov8") == "yolov8n.pt"
    with pytest.raises(ValueError) as excinfo:
        weight_name("nope")
    for known in ("yolo26", "yolo11", "yolov8"):
        assert known in str(excinfo.value)
    assert "nope" in str(excinfo.value)


def test_resolve_device_returns_requested_when_available():
    assert resolve_device("cpu", False, False) == ("cpu", None)
    assert resolve_device("mps", True, False) == ("mps", None)
    assert resolve_device("cuda", False, True) == ("cuda", None)


def test_resolve_device_unavailable_returns_cpu_with_reason():
    device, reason = resolve_device("mps", False, True)
    assert device == "cpu"
    assert reason and "mps" in reason
    device, reason = resolve_device("cuda", True, False)
    assert device == "cpu"
    assert reason and "cuda" in reason


def test_resolve_device_rejects_unknown_name():
    with pytest.raises(ValueError, match="tpu"):
        resolve_device("tpu", True, True)


def test_auto_prefers_mps_then_cuda_then_cpu():
    assert resolve_device("auto", True, True) == ("mps", None)
    assert resolve_device("auto", False, True) == ("cuda", None)
    assert resolve_device("auto", False, False) == ("cpu", None)
