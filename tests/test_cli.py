import csv
import os
import subprocess
from pathlib import Path

import pytest

from interviews_processing import run_diarize


def test_help_exits_cleanly(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["interviews-transcribe", "--help"])

    with pytest.raises(SystemExit) as exc:
        run_diarize.main()

    assert exc.value.code == 0
    assert "--whisper-model" in capsys.readouterr().out


def test_existing_csv_is_skipped_without_running_whisper(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audio = tmp_path / "audio" / "OBE" / "P1.wav"
    audio.parent.mkdir(parents=True)
    audio.touch()
    out = tmp_path / "results" / "OBE"
    out.mkdir(parents=True)
    (out / "P1.csv").touch()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("whisper should not run"))

    status, _, str_file = run_diarize.process_audio_file(
        str(audio), str(audio.parent), "tiny", "en", task="transcribe"
    )

    assert status == "SKIPPED"
    assert str_file is None


def test_translate_writes_to_language_to_eng_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audio = tmp_path / "OBE" / "P1.wav"
    audio.parent.mkdir()
    audio.touch()
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: calls.append(cmd))
    monkeypatch.setattr(run_diarize, "convert_str_to_csv", lambda *a: None)

    status, _, str_file = run_diarize.process_audio_file(
        str(audio), str(audio.parent), "tiny", "fr", task="translate"
    )

    assert status == "SUCCESS"
    assert Path(str_file) == Path("results", "OBE", "fr_to_eng", "P1.str")
    cmd = calls[0]
    assert cmd[cmd.index("--task") + 1] == "translate"
    assert cmd[cmd.index("-d") + 1] == os.path.join("results", "OBE", "fr_to_eng", "")


def test_failed_whisper_run_is_logged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audio = tmp_path / "OBE" / "P1.wav"
    audio.parent.mkdir()
    audio.touch()

    def fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", fail)
    run_diarize.init_log()

    status, msg, _ = run_diarize.process_audio_file(
        str(audio), str(audio.parent), "tiny", "en", task="transcribe"
    )

    assert status == "FAILED"
    assert "non-zero exit status 1" in msg
    with open(run_diarize.LOG_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["status"] == "FAILED"
    assert rows[-1]["file"].endswith("P1.wav")


@pytest.mark.slow
def test_end_to_end_transcription(tmp_path, monkeypatch, capfd):
    """Runs the real pipeline. Set AIP_TEST_AUDIO to a short speech recording (a few seconds)."""
    source = os.environ.get("AIP_TEST_AUDIO")
    if not source:
        pytest.skip("set AIP_TEST_AUDIO to a short speech recording to run this test")
    audio_dir = tmp_path / "e2e"
    audio_dir.mkdir()
    audio = audio_dir / Path(source).name
    audio.write_bytes(Path(source).read_bytes())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["interviews-transcribe", "-d", str(audio_dir), "--whisper-model", "tiny",
         "-e", audio.suffix],
    )

    run_diarize.main()

    output = capfd.readouterr()
    assert "Source splitting failed" not in output.out + output.err  # demucs vocal separation ran
    stem = audio.stem
    assert (tmp_path / "results" / "e2e" / f"{stem}.csv").exists()
    assert (tmp_path / "results" / "processed" / "e2e" / f"{stem}.txt").exists()
