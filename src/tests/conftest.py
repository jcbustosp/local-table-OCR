from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.engine import VALID_EXTENSIONS


@dataclass
class PipelineContext:
    config_file: Path
    fake_ocr: Mock
    fake_exporter: Mock


PipelineSetup = Callable[[Path], PipelineContext]


@pytest.fixture
def pipeline_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PipelineSetup:
    def _setup(input_path: Path) -> PipelineContext:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy")

        fake_config = SimpleNamespace(
            ocr=SimpleNamespace(export_format="csv"),
            paths=SimpleNamespace(
                input_dir=input_path,
                output_dir=tmp_path / "out",
            ),
        )

        fake_ocr = Mock()

        def is_valid_format(path: Path) -> bool:
            return path.suffix in VALID_EXTENSIONS

        fake_ocr.is_valid_image.side_effect = is_valid_format
        fake_ocr.process_image.return_value = ["data"]

        fake_exporter = Mock()

        monkeypatch.setattr(
            "src.main.load_config_file",
            Mock(return_value=fake_config),
        )
        monkeypatch.setattr(
            "src.main.OCREngine",
            Mock(return_value=fake_ocr),
        )
        monkeypatch.setattr(
            "src.main.DataExporter",
            Mock(return_value=fake_exporter),
        )

        return PipelineContext(
            config_file=config_file,
            fake_ocr=fake_ocr,
            fake_exporter=fake_exporter,
        )

    return _setup
