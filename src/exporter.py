import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import EXPORT_FORMATS

logger = logging.getLogger(__name__)


class DataExporter:
    def __init__(self, output_dir: Path, export_format: EXPORT_FORMATS) -> None:
        """Initializes the exporter with a strongly-typed target destination path."""
        self.output_dir: Path = output_dir
        self.export_format: str = export_format

    def save_results(self, result: list[Any], source_image_path: Path) -> Path:
        base_name = source_image_path.stem
        target_folder = self.output_dir / base_name
        target_folder.mkdir(parents=True, exist_ok=True)
        
        for idx, res in enumerate(result):
            suffix = f"_{idx}" if len(result) > 1 else ""
            excel_path = target_folder / f"{base_name}{suffix}_table.xlsx"
            
            # 1. Always generate the Excel file first since we know it works
            res.save_to_xlsx(save_path=str(excel_path))
            logger.info("Generated Excel workbook: %s", excel_path.name)

            # 2. If CSV is requested, convert that verified Excel file
            if self.export_format in ("csv", "both"):
                try:
                    # Read the Excel file we just created
                    xl = pd.ExcelFile(excel_path)
                    for sheet in xl.sheet_names:
                        df = xl.parse(sheet)
                        # Save to CSV using the sheet name for clarity
                        csv_path = target_folder / f"{base_name}_{sheet.lower()}.csv"
                        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                        logger.info("Successfully converted %s to CSV.", sheet)
                    
                    # Cleanup if the user only wanted CSV
                    if self.export_format == "csv":
                        excel_path.unlink()
                except Exception as e:
                    logger.error("Error during Excel-to-CSV conversion: %s", e)

        logger.info("✅ Structured maps successfully generated at: %s/", target_folder)
        return target_folder
