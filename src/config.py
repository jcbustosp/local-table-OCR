import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

EXPORT_FORMATS = Literal["csv", "excel", "both"]


class OcrSettings(BaseModel):
    """Validates deep learning structural engine settings."""

    language: str = Field(default="en", description="Target OCR language dictionary")
    export_format: EXPORT_FORMATS = Field(
        default="csv",
        description="Target table output format: 'excel', 'csv', or 'both'",
    )


class PathSettings(BaseModel):
    """Validates structural filesystem mappings."""

    input_dir: Path = Field(description="Path of file or folder with input files to process.")
    output_dir: Path = Field(default=Path("./dist"), description="Folder to save results")


class AppConfig(BaseModel):
    """The root configuration object filled purely by the input YAML file."""

    ocr: OcrSettings
    paths: PathSettings


def setup_logger() -> None:
    """Configures global logging behaviors uniformly to always show full logs."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
        force=True,
    )


def load_config_file(yaml_path: Path) -> AppConfig:
    """Loads a specific configuration yaml path, validates it, and starts logging."""
    if not yaml_path.exists():
        raise ValueError(f"Configuration file not found at '{yaml_path}'.")

    with yaml_path.open(encoding="utf-8") as f:
        raw_data: object = yaml.safe_load(f)

    config = AppConfig.model_validate(raw_data)

    setup_logger()

    # Automatically ensure output directory exists
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)

    return config
