"""Tests for FrameSourceFactory.from_config() using the hardware-free MockCapture."""

import builtins
import json

import pytest

from framesource import FrameSourceFactory
from framesource.errors import MissingDependencyError, UnknownSourceTypeError

from conftest import MockCapture

MOCK_TYPE = "mock"


@pytest.fixture(autouse=True)
def register_mock():
    FrameSourceFactory.register_capture_type(MOCK_TYPE, MockCapture)
    yield
    FrameSourceFactory.unregister_capture_type(MOCK_TYPE)


def test_from_config_dict_creates_instance():
    cap = FrameSourceFactory.from_config(
        {"source_type": MOCK_TYPE, "source_id": "mock", "connect": False}
    )
    assert isinstance(cap, MockCapture)
    assert cap.source == "mock"


def test_from_config_does_not_mutate_caller_dict():
    cfg = {"source_type": MOCK_TYPE, "source_id": "mock", "connect": False}
    original = dict(cfg)
    FrameSourceFactory.from_config(cfg)
    assert cfg == original


def test_from_config_connect_false_honoured():
    cap = FrameSourceFactory.from_config(
        {"source_type": MOCK_TYPE, "source_id": "mock", "connect": False}
    )
    assert not cap.is_connected
    # 'connect' is a control kwarg and must not leak into the instance config.
    assert "connect" not in cap.config


def test_from_config_extra_kwargs_forwarded():
    cap = FrameSourceFactory.from_config(
        {"source_type": MOCK_TYPE, "source_id": "mock", "connect": False, "width": 32}
    )
    assert cap.width == 32


def test_from_config_json_file_str_path(tmp_path):
    cfg = {"source_type": MOCK_TYPE, "source_id": "mock", "connect": False}
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    cap = FrameSourceFactory.from_config(str(path))
    assert isinstance(cap, MockCapture)
    assert cap.source == "mock"


def test_from_config_json_file_pathlib(tmp_path):
    cfg = {"source_type": MOCK_TYPE, "source_id": "mock", "connect": False}
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    cap = FrameSourceFactory.from_config(path)  # pass a Path object
    assert isinstance(cap, MockCapture)


def test_from_config_yaml_file(tmp_path):
    yaml = pytest.importorskip("yaml")
    cfg = {"source_type": MOCK_TYPE, "source_id": "mock", "connect": False}
    path = tmp_path / "camera.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    cap = FrameSourceFactory.from_config(path)
    assert isinstance(cap, MockCapture)
    assert cap.source == "mock"


def test_from_config_yaml_missing_pyyaml_raises(tmp_path, monkeypatch):
    # Simulate PyYAML being absent regardless of whether it is installed.
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    path = tmp_path / "camera.yaml"
    path.write_text("source_type: mock\nsource_id: mock\n", encoding="utf-8")

    with pytest.raises(MissingDependencyError):
        FrameSourceFactory.from_config(path)


def test_from_config_unknown_extension_raises(tmp_path):
    path = tmp_path / "camera.txt"
    path.write_text("not a real config", encoding="utf-8")

    with pytest.raises(ValueError):
        FrameSourceFactory.from_config(path)


def test_from_config_unknown_extension_raises_before_reading_missing_file(tmp_path):
    # Unknown extension is rejected even if the file does not exist.
    path = tmp_path / "camera.ini"
    with pytest.raises(ValueError):
        FrameSourceFactory.from_config(path)


def test_from_config_missing_file_raises_file_not_found(tmp_path):
    path = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        FrameSourceFactory.from_config(path)


def test_from_config_unknown_source_type_raises():
    with pytest.raises(UnknownSourceTypeError):
        FrameSourceFactory.from_config(
            {"source_type": "definitely_not_registered", "source_id": 0}
        )
