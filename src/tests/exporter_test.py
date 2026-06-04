from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from src.exporter import DataExporter

# -------------------------
# Helpers
# -------------------------


class FakeResult:
    def __init__(self):
        self.save_to_xlsx = MagicMock()


def make_image(tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")
    return img


# -------------------------
# Tests
# -------------------------


def test_save_results_creates_folder_and_excel(tmp_path, monkeypatch):
    exporter = DataExporter(tmp_path / "out", "csv")

    result = [FakeResult()]
    source_image = make_image(tmp_path)

    fake_df = pd.DataFrame({"a": [1, 2]})

    mock_excel = MagicMock()
    mock_excel.sheet_names = ["Sheet1"]
    mock_excel.parse.return_value = fake_df

    monkeypatch.setattr(pd, "ExcelFile", lambda _: mock_excel)

    fake_df.to_csv = MagicMock()

    target = exporter.save_results(result, source_image)

    assert target.exists()
    assert result[0].save_to_xlsx.called
    fake_df.to_csv.assert_called()


def test_save_results_excel_only(tmp_path, monkeypatch):
    exporter = DataExporter(tmp_path / "out", "excel")

    result = [FakeResult()]
    source_image = make_image(tmp_path)

    monkeypatch.setattr(pd, "ExcelFile", MagicMock())

    target = exporter.save_results(result, source_image)

    assert target.exists()
    assert result[0].save_to_xlsx.called


def test_save_results_multiple_results(tmp_path, monkeypatch):
    exporter = DataExporter(tmp_path / "out", "csv")

    result = [FakeResult(), FakeResult()]
    source_image = make_image(tmp_path)

    fake_df = pd.DataFrame({"x": [1]})

    mock_excel = MagicMock()
    mock_excel.sheet_names = ["Sheet1"]
    mock_excel.parse.return_value = fake_df

    monkeypatch.setattr(pd, "ExcelFile", lambda _: mock_excel)

    fake_df.to_csv = MagicMock()

    exporter.save_results(result, source_image)

    assert result[0].save_to_xlsx.called
    assert result[1].save_to_xlsx.called


def test_csv_only_deletes_excel(tmp_path, monkeypatch):
    exporter = DataExporter(tmp_path / "out", "csv")

    result = [FakeResult()]
    source_image = make_image(tmp_path)

    path_holder = {}

    def fake_save(save_path):
        p = Path(save_path)
        p.write_text("fake")
        path_holder["p"] = p

    result[0].save_to_xlsx = fake_save

    mock_excel = MagicMock()
    mock_excel.sheet_names = ["Sheet1"]
    mock_excel.parse.return_value = pd.DataFrame({"a": [1]})

    monkeypatch.setattr(pd, "ExcelFile", lambda _: mock_excel)

    exporter.save_results(result, source_image)

    assert not path_holder["p"].exists()


def test_exception_handling_does_not_crash(tmp_path, monkeypatch):
    exporter = DataExporter(tmp_path / "out", "csv")

    result = [FakeResult()]
    source_image = make_image(tmp_path)

    monkeypatch.setattr(pd, "ExcelFile", lambda *_: (_ for _ in ()).throw(Exception("boom")))

    exporter.save_results(result, source_image)

    assert result[0].save_to_xlsx.called


def test_excel_only_branch(tmp_path, monkeypatch):
    exporter = DataExporter(tmp_path / "out", "excel")

    result = [FakeResult()]
    source_image = make_image(tmp_path)

    mock_excel = MagicMock()

    monkeypatch.setattr(pd, "ExcelFile", mock_excel)

    exporter.save_results(result, source_image)

    mock_excel.assert_not_called()


def test_csv_conversion_failure_is_handled(tmp_path, monkeypatch):
    exporter = DataExporter(tmp_path / "out", "csv")

    result = [FakeResult()]
    source_image = make_image(tmp_path)

    class FakeExcel:
        sheet_names = ["Sheet1"]  # noqa: RUF012

        def parse(self, sheet):
            raise Exception("boom")

    monkeypatch.setattr(pd, "ExcelFile", lambda _: FakeExcel())

    exporter.save_results(result, source_image)


def test_suffix_multiple_results(tmp_path, monkeypatch):
    exporter = DataExporter(tmp_path / "out", "csv")

    result = [FakeResult(), FakeResult()]
    source_image = make_image(tmp_path)

    mock_excel = MagicMock()
    mock_excel.sheet_names = ["Sheet1"]
    mock_excel.parse.return_value = pd.DataFrame({"x": [1]})

    monkeypatch.setattr(pd, "ExcelFile", lambda _: mock_excel)

    exporter.save_results(result, source_image)


def test_export_format_both_does_not_delete_excel(tmp_path, monkeypatch):
    exporter = DataExporter(tmp_path / "out", "both")

    result = [FakeResult()]
    source_image = make_image(tmp_path)

    excel_holder = {}

    def fake_save(save_path):
        p = Path(save_path)
        p.write_text("fake")
        excel_holder["path"] = p

    result[0].save_to_xlsx = fake_save

    mock_excel = MagicMock()
    mock_excel.sheet_names = ["Sheet1"]
    mock_excel.parse.return_value = pd.DataFrame({"a": [1]})

    monkeypatch.setattr(pd, "ExcelFile", lambda _: mock_excel)

    exporter.save_results(result, source_image)

    assert excel_holder["path"].exists()
