from pathlib import Path
from unittest.mock import MagicMock, Mock

import cv2
import pytest

from ..engine import OCREngine, OcrSettings


@pytest.fixture
def settings():
    return OcrSettings(language="en")


@pytest.fixture
def engine(settings: OcrSettings, monkeypatch: pytest.MonkeyPatch):
    # Mock PPStructureV3 so we don't load real model
    mock_pp = MagicMock()
    mock_instance = MagicMock()
    mock_instance.predict.return_value = [{"text": "dummy"}]
    mock_pp.return_value = mock_instance

    monkeypatch.setattr("src.engine.PPStructureV3", mock_pp)

    return OCREngine(settings)


def test_is_valid_image_true(tmp_path: Path):
    file = tmp_path / "img.png"
    file.write_bytes(b"fake")

    assert OCREngine.is_valid_image(file) is True


def test_is_valid_image_false_extension(tmp_path: Path):
    file = tmp_path / "img.txt"
    file.write_bytes(b"fake")

    assert OCREngine.is_valid_image(file) is False


def test_is_valid_image_false_not_file(tmp_path: Path):
    folder = tmp_path / "folder"
    folder.mkdir()

    assert OCREngine.is_valid_image(folder) is False


def test_process_image_success(engine: OCREngine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"fake")

    # Mock cv2.imread to return fake image array
    def fake_imread(_: str) -> str:
        return "fake_image"

    monkeypatch.setattr(cv2, "imread", fake_imread)

    result = engine.process_image(image_path)

    assert result == [{"text": "dummy"}]
    engine.engine.predict.assert_called_once()


def test_process_image_file_missing(engine: OCREngine, tmp_path: Path):
    missing = tmp_path / "missing.jpg"

    with pytest.raises(FileNotFoundError):
        engine.process_image(missing)


def test_process_image_invalid_image(
    engine: OCREngine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"fake")

    # Simulate unreadable image
    monkeypatch.setattr(cv2, "imread", Mock(return_value=None))

    with pytest.raises(ValueError):
        engine.process_image(image_path)
