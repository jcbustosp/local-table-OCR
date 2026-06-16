import sys
from pathlib import Path

import pytest

from src.main import main, run_pipeline

from .conftest import PipelineSetup


@pytest.mark.parametrize(
    "file_name, expected_output, calls", [("foo.jpg", 0, 1), ("foo.txt", 1, 0)]
)
def test_run_pipeline_single_file(
    tmp_path: Path,
    pipeline_setup: PipelineSetup,
    file_name: str,
    expected_output: int,
    calls: int,
):
    input_file = tmp_path / file_name
    input_file.write_bytes(b"fake")

    ctx = pipeline_setup(input_file)

    assert run_pipeline(str(ctx.config_file)) == expected_output

    assert ctx.fake_ocr.process_image.call_count == calls
    assert ctx.fake_exporter.save_results.call_count == calls


def test_run_pipeline_invalid_file_extension(tmp_path: Path, pipeline_setup: PipelineSetup):
    input_file = tmp_path / "img.png"
    input_file.write_bytes(b"fake")

    ctx = pipeline_setup(
        Path("img.txt"),
    )

    assert run_pipeline(str(ctx.config_file)) == 1


def test_run_pipeline_directory_with_images(
    tmp_path: Path,
    pipeline_setup: PipelineSetup,
):
    dir_path = tmp_path / "images"
    dir_path.mkdir()

    (dir_path / "a.png").write_bytes(b"fake")
    (dir_path / "b.jpg").write_bytes(b"fake")

    ctx = pipeline_setup(dir_path)

    assert run_pipeline(str(ctx.config_file)) == 0

    ctx.fake_ocr.process_image.assert_called()
    assert ctx.fake_ocr.process_image.call_count == 2

    ctx.fake_exporter.save_results.assert_called()
    assert ctx.fake_exporter.save_results.call_count == 2


def test_run_pipeline_directory_empty(
    tmp_path: Path,
    pipeline_setup: PipelineSetup,
):
    dir_path = tmp_path / "images"
    dir_path.mkdir()

    ctx = pipeline_setup(dir_path)

    assert run_pipeline(str(ctx.config_file)) == 0


def test_run_pipeline_directory_skips_invalid_files(
    tmp_path: Path,
    pipeline_setup: PipelineSetup,
):
    dir_path = tmp_path / "images"
    dir_path.mkdir()

    (dir_path / "a.png").write_bytes(b"fake")
    (dir_path / "b.txt").write_bytes(b"fake")

    ctx = pipeline_setup(dir_path)

    assert run_pipeline(str(ctx.config_file)) == 0
    assert ctx.fake_exporter.save_results.call_count == 1


def test_pipeline_continues_after_single_file_failure(
    tmp_path: Path,
    pipeline_setup: PipelineSetup,
):
    dir_path = tmp_path / "images"
    dir_path.mkdir()

    (dir_path / "a.png").write_bytes(b"fake")
    (dir_path / "b.jpg").write_bytes(b"fake")

    ctx = pipeline_setup(dir_path)

    # make ONLY second file fail
    def fake_process(path: Path):
        if path.name == "b.jpg":
            raise Exception("boom")
        return ["data"]

    ctx.fake_ocr.process_image.side_effect = fake_process

    result = run_pipeline(str(ctx.config_file))

    assert result == 1  # because at least one failure

    # both files attempted
    assert ctx.fake_ocr.process_image.call_count == 2
    # only successful file written
    assert ctx.fake_exporter.save_results.call_count == 1


def test_pipeline_global_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dummy")

    def crash():
        raise Exception("Pipeline crash")

    monkeypatch.setattr("src.main.load_config_file", lambda: crash)

    assert run_pipeline(str(config_file)) == 1


def test_main_missing_args(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(sys, "argv", ["main.py"])

    with pytest.raises(SystemExit) as e:
        main()

    captured = capsys.readouterr()

    assert e.value.code == 1
    assert "Missing required arguments" in captured.err


def test_main_runs_pipeline(monkeypatch: pytest.MonkeyPatch):
    called = {}

    def fake_run_pipeline(config_file_path: Path):
        called["config"] = config_file_path
        return 0

    monkeypatch.setattr("src.main.run_pipeline", fake_run_pipeline)

    monkeypatch.setattr(sys, "argv", ["main.py", "config.yaml"])

    with pytest.raises(SystemExit) as e:
        main()

    assert e.value.code == 0
    assert called["config"] == "config.yaml"
