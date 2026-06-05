import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.main import main, run_pipeline


class FakeOCR:
    def __init__(self, valid=True):
        self._valid = valid
        self.process_image = MagicMock(return_value=["data"])

    def is_valid_image(self, path: Path) -> bool:
        return self._valid


class FakeExporter:
    def __init__(self):
        self.save_results = MagicMock()


def test_run_pipeline_single_file(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dummy")

    input_file = tmp_path / "img.png"
    input_file.write_bytes(b"fake")

    # --- mock config ---
    fake_config = SimpleNamespace(
        ocr=SimpleNamespace(export_format="csv"),
        paths=SimpleNamespace(input_dir=input_file, output_dir=tmp_path / "out"),
    )

    monkeypatch.setattr("src.main.load_config_file", lambda _: fake_config)

    # --- mock OCR ---
    fake_ocr = FakeOCR(valid=True)
    monkeypatch.setattr("src.main.OCREngine", lambda settings: fake_ocr)

    # --- mock exporter ---
    fake_exporter = FakeExporter()
    monkeypatch.setattr("src.main.DataExporter", lambda **_: fake_exporter)

    # run
    run_pipeline(str(config_file))

    assert fake_ocr.process_image.called
    assert fake_exporter.save_results.called


def test_run_pipeline_invalid_file_extension(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dummy")

    input_file = tmp_path / "img.txt"
    input_file.write_bytes(b"fake")

    fake_config = SimpleNamespace(
        ocr=SimpleNamespace(export_format="csv"),
        paths=SimpleNamespace(input_dir=input_file, output_dir=tmp_path / "out"),
    )

    monkeypatch.setattr("src.main.load_config_file", lambda _: fake_config)
    monkeypatch.setattr("src.main.OCREngine", lambda settings: FakeOCR(valid=False))
    monkeypatch.setattr("src.main.DataExporter", lambda **_: FakeExporter())

    # sys.exit is triggered → catch it
    with pytest.raises(SystemExit):
        run_pipeline(str(config_file))


def test_run_pipeline_directory_with_images(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dummy")

    dir_path = tmp_path / "images"
    dir_path.mkdir()

    img1 = dir_path / "a.png"
    img1.write_bytes(b"fake")

    img2 = dir_path / "b.jpg"
    img2.write_bytes(b"fake")

    fake_config = SimpleNamespace(
        ocr=SimpleNamespace(export_format="csv"),
        paths=SimpleNamespace(input_dir=dir_path, output_dir=tmp_path / "out"),
    )

    monkeypatch.setattr("src.main.load_config_file", lambda _: fake_config)

    fake_ocr = FakeOCR(valid=True)
    monkeypatch.setattr("src.main.OCREngine", lambda settings: fake_ocr)

    fake_exporter = FakeExporter()
    monkeypatch.setattr("src.main.DataExporter", lambda **_: fake_exporter)

    run_pipeline(str(config_file))

    assert fake_ocr.process_image.call_count == 2
    assert fake_exporter.save_results.call_count == 2


def test_run_pipeline_directory_empty(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dummy")

    dir_path = tmp_path / "empty"
    dir_path.mkdir()

    fake_config = SimpleNamespace(
        ocr=SimpleNamespace(export_format="csv"),
        paths=SimpleNamespace(input_dir=dir_path, output_dir=tmp_path / "out"),
    )

    monkeypatch.setattr("src.main.load_config_file", lambda _: fake_config)
    monkeypatch.setattr("src.main.OCREngine", lambda settings: FakeOCR(valid=False))
    monkeypatch.setattr("src.main.DataExporter", lambda **_: FakeExporter())

    # should NOT crash, just return
    run_pipeline(str(config_file))


def test_main_missing_args(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py"])

    with pytest.raises(SystemExit) as e:
        main()

    captured = capsys.readouterr()

    assert e.value.code == 1
    assert "Missing required arguments" in captured.err


def test_run_pipeline_directory_skips_invalid_files(tmp_path, monkeypatch):

    config_file = tmp_path / "config.yaml"
    config_file.write_text("dummy")

    dir_path = tmp_path / "images"
    dir_path.mkdir()

    valid = dir_path / "a.png"
    valid.write_bytes(b"fake")

    invalid = dir_path / "b.txt"
    invalid.write_bytes(b"fake")

    fake_config = SimpleNamespace(
        ocr=SimpleNamespace(export_format="csv"),
        paths=SimpleNamespace(input_dir=dir_path, output_dir=tmp_path / "out"),
    )

    monkeypatch.setattr("src.main.load_config_file", lambda _: fake_config)

    class FakeOCR:
        def is_valid_image(self, path):
            return path.suffix == ".png"  # ❗ creates both branches

        def process_image(self, path):
            return ["data"]

    monkeypatch.setattr("src.main.OCREngine", lambda settings: FakeOCR())

    fake_exporter = MagicMock()
    monkeypatch.setattr("src.main.DataExporter", lambda **_: fake_exporter)

    run_pipeline(str(config_file))

    # only valid file processed
    assert fake_exporter.save_results.call_count == 1


def test_pipeline_handles_single_file_error(monkeypatch, tmp_path):

    config_file = tmp_path / "config.yaml"
    config_file.write_text("dummy")

    input_file = tmp_path / "img.png"
    input_file.write_bytes(b"fake")

    # --- fake config ---
    fake_config = MagicMock()
    fake_config.ocr.export_format = "csv"
    fake_config.paths.output_dir = tmp_path / "out"

    monkeypatch.setattr("src.main.load_config_file", lambda _: fake_config)

    # --- OCR engine ---
    class FakeOCR:
        def is_valid_image(self, path):
            return True

        def process_image(self, path):
            raise Exception("OCR failed")  # triggers inner except

    monkeypatch.setattr("src.main.OCREngine", lambda settings: FakeOCR())

    # --- exporter ---
    fake_exporter = MagicMock()
    monkeypatch.setattr("src.main.DataExporter", lambda **_: fake_exporter)

    # should NOT raise
    run_pipeline(str(config_file))

    # exporter never called due to failure
    assert not fake_exporter.save_results.called


def test_main_runs_pipeline(monkeypatch):
    called = {}

    def fake_run_pipeline(config_file_path):
        called["config"] = config_file_path

    monkeypatch.setattr("src.main.run_pipeline", fake_run_pipeline)

    monkeypatch.setattr(sys, "argv", ["main.py", "config.yaml"])

    main()

    assert called["config"] == "config.yaml"


def test_pipeline_global_crash(monkeypatch, tmp_path):

    config_file = tmp_path / "config.yaml"
    config_file.write_text("dummy")

    input_file = tmp_path / "img.png"
    input_file.write_bytes(b"fake")

    # force config loader crash → outer exception
    def crash(_):
        raise Exception("boom")

    monkeypatch.setattr("src.main.load_config_file", crash)

    with pytest.raises(SystemExit) as e:
        run_pipeline(str(config_file))

    assert e.value.code == 1
