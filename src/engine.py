import logging
from pathlib import Path
from typing import ClassVar

import cv2
from paddleocr import PPStructureV3

from src.config import OcrSettings

logger = logging.getLogger(__name__)


class OCREngine:
    # Supported image formats for PaddleOCR processing
    VALID_EXTENSIONS: ClassVar[frozenset[str]] = frozenset(
        {
            ".png",
            ".jpg",
            ".jpeg",
        }
    )

    def __init__(self, settings: OcrSettings):
        """Cleanly injects only the configuration slices this module requires."""
        logger.info(
            "Initializing Native PP-StructureV3 Pipeline [Lang: %s]...",
            settings.language,
        )
        self.engine = PPStructureV3(lang=settings.language)

    @classmethod
    def is_valid_image(cls, file_path: Path) -> bool:
        """Validates if a given file path is a supported image extension."""
        return file_path.is_file() and file_path.suffix.lower() in cls.VALID_EXTENSIONS

    def process_image(self, image_path: Path):
        """Reads a validated target image and executes OCR structural tracking."""
        if not image_path.exists():
            raise FileNotFoundError(f"Target image missing at: {image_path}")

        logger.info("📸 Processing Image: %s", image_path.name)
        img = cv2.imread(str(image_path))

        if img is None:
            raise ValueError(f"Could not open or parse image format: {image_path}")

        logger.debug("Running layout analysis and cell segmentation...")
        return self.engine.predict(img)
