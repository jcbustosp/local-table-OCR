from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import pytest

from ..engine import OCREngine


@pytest.fixture
def settings():
    return SimpleNamespace(language="en")


@pytest.fixture
def engine(settings, monkeypatch):
    # Mock PPStructureV3 so we don't load real model
    mock_pp = MagicMock()
    mock_instance = MagicMock()
    mock_instance.predict.return_value = [{"text": "dummy"}]
    mock_pp.return_value = mock_instance

    monkeypatch.setattr("src.engine.PPStructureV3", mock_pp)

    return OCREngine(settings)


def test_is_valid_image_true(tmp_path):
    file = tmp_path / "img.png"
    file.write_bytes(b"fake")

    assert OCREngine.is_valid_image(file) is True


def test_is_valid_image_false_extension(tmp_path):
    file = tmp_path / "img.txt"
    file.write_bytes(b"fake")

    assert OCREngine.is_valid_image(file) is False


def test_is_valid_image_false_not_file(tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()

    assert OCREngine.is_valid_image(folder) is False


def test_process_image_success(engine, monkeypatch, tmp_path):
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"fake")

    # Mock cv2.imread to return fake image array
    monkeypatch.setattr(cv2, "imread", lambda _: "fake_image")

    result = engine.process_image(image_path)

    assert result == [{"text": "dummy"}]
    engine.engine.predict.assert_called_once()


def test_process_image_file_missing(engine, tmp_path):
    missing = tmp_path / "missing.jpg"

    with pytest.raises(FileNotFoundError):
        engine.process_image(missing)


def test_process_image_invalid_image(engine, monkeypatch, tmp_path):
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"fake")

    # Simulate unreadable image
    monkeypatch.setattr(cv2, "imread", lambda _: None)

    with pytest.raises(ValueError):
        engine.process_image(image_path)
