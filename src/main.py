import logging
import sys
from pathlib import Path

from src.config import load_config_file
from src.engine import OCREngine
from src.exporter import DataExporter

logger = logging.getLogger(__name__)


def run_pipeline(config_file_path: str) -> None:
    """Orchestrates single file validation or multi-file directory batch processing."""
    try:
        # 1. Load configuration and spin up processors via Dependency Injection
        config = load_config_file(Path(config_file_path))
        ocr_engine = OCREngine(settings=config.ocr)

        # Inject output directory and export format
        exporter = DataExporter(
            output_dir=config.paths.output_dir, export_format=config.ocr.export_format
        )

        target = config.paths.input_dir
        images_to_process = []

        # 2. Resolve input targets dynamically
        if target.is_file():
            if ocr_engine.is_valid_image(target):
                images_to_process.append(target)
            else:
                logger.error("Unsupported file extension format: %s", target.suffix)
                sys.exit(1)

        elif target.is_dir():
            logger.info("Directory target detected. Scanning for valid image files...")
            for file_path in sorted(target.iterdir()):
                if ocr_engine.is_valid_image(file_path):
                    images_to_process.append(file_path)

            if not images_to_process:
                logger.warning("No valid images found inside: %s", target)
                return
        else:
            logger.error("Input target path does not exist: %s", target)
            sys.exit(1)

        # 3. Process discovered images sequentially
        logger.info("Found %d file(s) queue ready for processing.", len(images_to_process))

        for idx, img_path in enumerate(images_to_process, start=1):
            logger.info("--- [%d / %d] ---", idx, len(images_to_process))
            try:
                raw_data = ocr_engine.process_image(img_path)
                exporter.save_results(raw_data, source_image_path=img_path)
            except Exception as e:
                logger.error("Failed to process file %s: %s", img_path.name, e)

    except Exception as e:
        logger.exception("An unexpected orchestration crash occurred: %s", e)
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        sys.stderr.write(
            "\n❌ Error: Missing required arguments.\n"
            "💡 Usage:\n"
            "   uv run python -m src.main <CONFIG_PATH> <INPUT_FILE_OR_DIR>\n\n"
        )
        sys.exit(1)

    CONFIG_INPUT = sys.argv[1]

    run_pipeline(CONFIG_INPUT)


if __name__ == "__main__":  # pragma: nocover
    main()
