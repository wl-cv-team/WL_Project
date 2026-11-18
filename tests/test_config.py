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


@pytest.mark.parametrize("name", ["", None, 26])
def test_config_rejects_empty_or_non_string_model_name(name):
    with pytest.raises(ValueError, match="model_name"):
        DetectorConfig(model_name=name)


@pytest.mark.parametrize("field", ["imgsz", "epochs"])
@pytest.mark.parametrize("value", [True, 640.0, "640"])
def test_config_rejects_non_int_imgsz_and_epochs(field, value):
    with pytest.raises(ValueError, match=field):
        DetectorConfig(model_name="yolo26", **{field: value})


@pytest.mark.parametrize("value", [True, "0.5", None])
def test_config_rejects_non_real_conf(value):
    with pytest.raises(ValueError, match="conf"):
        DetectorConfig(model_name="yolo26", conf=value)


def test_config_accepts_int_conf_in_bounds():
    assert DetectorConfig(model_name="yolo26", conf=1).conf == 1


def test_resolve_device_accepts_indexed_cuda():
    assert resolve_device("cuda:0", False, True) == ("cuda:0", None)
    device, reason = resolve_device("cuda:1", True, False)
    assert device == "cpu"
    assert reason and "cuda:1" in reason
    with pytest.raises(ValueError, match="cuda:x"):
        resolve_device("cuda:x", True, True)


def test_config_stores_weights_path_as_given():
    from pathlib import Path

    assert DetectorConfig(model_name="yolo26").weights_path is None
    path = Path("artifacts/plate.pt")
    assert DetectorConfig(model_name="yolo26", weights_path=path).weights_path == path
    assert DetectorConfig(model_name="yolo26", weights_path="plate.pt").weights_path == "plate.pt"


def test_config_rejects_empty_or_wrong_type_weights_path():
    for value in ("", 5, b"x"):
        with pytest.raises(ValueError, match="weights_path"):
            DetectorConfig(model_name="yolo26", weights_path=value)
