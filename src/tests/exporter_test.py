from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.exporter import DataExporter

# -------------------------
# Helpers
# -------------------------


class FakeResult:
    def __init__(self):
        self.save_to_xlsx = MagicMock()


def make_image(tmp_path: Path):
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")
    return img


# -------------------------
# Tests
# -------------------------


def test_save_results_creates_folder_and_excel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = DataExporter(tmp_path / "out", "csv")

    result = [FakeResult()]
    source_image = make_image(tmp_path)

    fake_df = pd.DataFrame({"a": [1, 2]})
    fake_df.to_csv = MagicMock()

    # Mock ExcelFile
    mock_excel_cls = MagicMock()
    mock_excel = mock_excel_cls.return_value
    mock_excel.sheet_names = ["Sheet1"]

    monkeypatch.setattr(pd, "ExcelFile", mock_excel_cls)

    # Mock read_excel
    monkeypatch.setattr(pd, "read_excel", MagicMock(return_value=fake_df))

    target = exporter.save_results(result, source_image)

    assert target.exists()
    result[0].save_to_xlsx.assert_called_once()
    fake_df.to_csv.assert_called_once()


def test_save_results_excel_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    exporter = DataExporter(tmp_path / "out", "excel")

    result = [FakeResult()]
    source_image = make_image(tmp_path)

    monkeypatch.setattr(pd, "ExcelFile", MagicMock())

    target = exporter.save_results(result, source_image)

    assert target.exists()
    assert result[0].save_to_xlsx.called


def test_save_results_multiple_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    exporter = DataExporter(tmp_path / "out", "csv")

    result = [FakeResult(), FakeResult()]
    source_image = make_image(tmp_path)

    fake_df = pd.DataFrame({"x": [1]})

    mock_excel = MagicMock()
    mock_excel.sheet_names = ["Sheet1"]
    mock_excel.parse.return_value = fake_df

    monkeypatch.setattr(pd, "ExcelFile", mock_excel)

    fake_df.to_csv = MagicMock()

    exporter.save_results(result, source_image)

    assert result[0].save_to_xlsx.called
    assert result[1].save_to_xlsx.called


def test_csv_only_deletes_excel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    exporter = DataExporter(tmp_path / "out", "csv")

    result = [FakeResult()]
    source_image = make_image(tmp_path)

    path_holder = {}

    def fake_save(save_path: str):
        p = Path(save_path)
        p.write_text("fake")
        path_holder["p"] = p

    result[0].save_to_xlsx = fake_save

    mock_excel = MagicMock()
    mock_excel.sheet_names = ["Sheet1"]
    mock_excel.parse.return_value = pd.DataFrame({"a": [1]})

    monkeypatch.setattr(pd, "ExcelFile", mock_excel)

    exporter.save_results(result, source_image)

    assert not path_holder["p"].exists()


def test_exception_handling_does_not_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = DataExporter(tmp_path / "out", "csv")

    result = [FakeResult()]
    source_image = make_image(tmp_path)

    def raise_excel_error(*args: object, **kwargs: object) -> None:
        raise Exception("boom")

    monkeypatch.setattr(pd, "ExcelFile", raise_excel_error)

    exporter.save_results(result, source_image)

    result[0].save_to_xlsx.assert_called_once()


def test_excel_only_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    exporter = DataExporter(tmp_path / "out", "excel")

    result = [FakeResult()]
    source_image = make_image(tmp_path)

    mock_excel = MagicMock()

    monkeypatch.setattr(pd, "ExcelFile", mock_excel)

    exporter.save_results(result, source_image)

    mock_excel.assert_not_called()


def test_csv_conversion_failure_is_handled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    exporter = DataExporter(tmp_path / "out", "csv")

    result = [FakeResult()]
    source_image = make_image(tmp_path)

    class FakeExcel:
        sheet_names = ["Sheet1"]  # noqa: RUF012

        def parse(self):
            raise Exception("boom")

    monkeypatch.setattr(pd, "ExcelFile", FakeExcel())

    exporter.save_results(result, source_image)


def test_suffix_multiple_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    exporter = DataExporter(tmp_path / "out", "csv")

    result = [FakeResult(), FakeResult()]
    source_image = make_image(tmp_path)

    mock_excel = MagicMock()
    mock_excel.sheet_names = ["Sheet1"]
    mock_excel.parse.return_value = pd.DataFrame({"x": [1]})

    monkeypatch.setattr(pd, "ExcelFile", mock_excel)

    exporter.save_results(result, source_image)


def test_export_format_both_does_not_delete_excel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    exporter = DataExporter(tmp_path / "out", "both")

    result = [FakeResult()]
    source_image = make_image(tmp_path)

    excel_holder = {}

    def fake_save(save_path: str):
        p = Path(save_path)
        p.write_text("fake")
        excel_holder["path"] = p

    result[0].save_to_xlsx = fake_save

    mock_excel = MagicMock()
    mock_excel.sheet_names = ["Sheet1"]
    mock_excel.parse.return_value = pd.DataFrame({"a": [1]})

    monkeypatch.setattr(pd, "ExcelFile", mock_excel)

    exporter.save_results(result, source_image)

    assert excel_holder["path"].exists()
