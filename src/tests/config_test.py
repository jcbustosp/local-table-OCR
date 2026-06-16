import logging
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import (
    OcrSettings,
    load_config_file,
    setup_logger,
)


def test_ocr_settings_defaults():
    cfg = OcrSettings()
    assert cfg.language == "en"
    assert cfg.export_format == "csv"


def test_ocr_settings_custom():
    cfg = OcrSettings(language="fr", export_format="excel")
    assert cfg.language == "fr"
    assert cfg.export_format == "excel"


def test_setup_logger():
    setup_logger()
    root = logging.getLogger()

    assert root.level <= logging.DEBUG
    assert len(root.handlers) > 0


def test_load_config_file_with_yaml(tmp_path: Path):
    yaml_file = tmp_path / "config.yaml"

    yaml_file.write_text(
        yaml.dump(
            {
                "ocr": {"language": "de", "export_format": "csv"},
                "paths": {"input_dir": str(tmp_path / "in"), "output_dir": str(tmp_path / "out")},
            }
        ),
        encoding="utf-8",
    )

    config = load_config_file(yaml_file)

    assert config.ocr.language == "de"
    assert config.paths.output_dir.exists()


def test_load_config_file_missing(tmp_path: Path):
    missing = tmp_path / "missing.yaml"

    with pytest.raises(ValueError, match="Configuration file not found at"):
        load_config_file(missing)


def test_load_config_file_empty(tmp_path: Path):
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text("", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config_file(yaml_file)
