from datetime import timedelta

import pandas as pd

from interviews_processing.utils.format_helpers import (
    convert_str_to_csv,
    extract_id,
    format_timedelta,
    get_files,
)
from interviews_processing.whisper_diarization.helpers import write_srt


def test_get_files_walks_subfolders_and_filters_extensions(tmp_path):
    (tmp_path / "exp" / "sub").mkdir(parents=True)
    for name in ["exp/a.wav", "exp/sub/b.m4a", "exp/notes.txt"]:
        (tmp_path / name).touch()

    found = get_files(str(tmp_path / "exp"), [".wav", ".m4a"])

    assert sorted(p.replace("\\", "/").split("exp/")[1] for p in found) == ["a.wav", "sub/b.m4a"]


def test_extract_id():
    assert extract_id("participant_12_part2") == 12
    assert extract_id("no_digits") is None


def test_format_timedelta():
    assert format_timedelta(timedelta(hours=1, minutes=2, seconds=3.9)) == "01:02:03"


def test_srt_round_trips_to_csv(tmp_path, segments):
    str_file = tmp_path / "P07_interview.str"
    with open(str_file, "w", encoding="utf-8-sig") as f:
        write_srt(segments, f)

    convert_str_to_csv(str(str_file), experiment="OBE")

    df = pd.read_csv(tmp_path / "P07_interview.csv", encoding="utf-8-sig")
    assert list(df.columns) == [
        "Experiment", "File Name", "Id", "Content Type", "Start Time", "End Time", "Speaker", "Content",
    ]
    assert len(df) == 3
    assert df["Experiment"].unique().tolist() == ["OBE"]
    assert df["Id"].unique().tolist() == [7]
    assert df["Speaker"].tolist() == [0, 1, 1]
    assert df.loc[1, "Start Time"] == "00:00:02,000"
    assert df.loc[1, "Content"] == "I went to my office and I worked."
