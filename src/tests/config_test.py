import logging

import yaml

from src.config import (
    AppConfig,
    OcrSettings,
    PathSettings,
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


def test_app_config_defaults():
    cfg = AppConfig()
    assert isinstance(cfg.ocr, OcrSettings)
    assert isinstance(cfg.paths, PathSettings)


def test_setup_logger():
    setup_logger()
    root = logging.getLogger()

    assert root.level <= logging.DEBUG
    assert len(root.handlers) > 0


def test_load_config_file_with_yaml(tmp_path):
    yaml_file = tmp_path / "config.yaml"

    yaml_file.write_text(
        yaml.dump(
            {
                "ocr": {"language": "de", "export_format": "csv"},
                "paths": {"output_dir": str(tmp_path / "out")},
            }
        ),
        encoding="utf-8",
    )

    config = load_config_file(yaml_file)

    assert config.ocr.language == "de"
    assert config.paths.output_dir.exists()


def test_load_config_file_missing(tmp_path, capsys):
    missing = tmp_path / "missing.yaml"

    config = load_config_file(missing)
    captured = capsys.readouterr()

    assert "Configuration file not found" in captured.out
    assert config.ocr.language == "en"
    assert config.paths.output_dir.exists()


def test_load_config_file_empty(tmp_path):
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text("", encoding="utf-8")

    config = load_config_file(yaml_file)

    assert config.ocr.language == "en"
    assert config.paths.output_dir.exists()
