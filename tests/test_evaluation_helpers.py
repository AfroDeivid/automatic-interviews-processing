import pandas as pd
import pytest

from interviews_processing.utils.evaluation_helpers import (
    calculate_wer_and_generate_html,
    compute_der,
    extract_text_WER,
    format_time,
    load_data_time,
    process_folder_csv,
    time_to_seconds,
)


def turns(*rows):
    """Build a diarization DataFrame from (start_s, end_s, speaker) tuples."""
    return pd.DataFrame(rows, columns=["Start", "End", "Speaker"])


def transcript(rows):
    return pd.DataFrame(rows, columns=["Start Time", "End Time", "Speaker", "Content"])


REFERENCE = transcript([
    ("00:00:00,000", "00:00:05,000", "Interviewer", "Can you tell me about your day?"),
    ("00:00:05,000", "00:00:10,000", "Participant", "I went to my office and worked."),
])


@pytest.mark.parametrize(
    "value, expected",
    [("00:01:02,500", 62.5), ("01:00:00.250", 3600.25), (" 00:00:07 ", 7.0)],
)
def test_time_to_seconds(value, expected):
    assert time_to_seconds(value) == expected


def test_time_to_seconds_rejects_bad_format():
    with pytest.raises(ValueError):
        time_to_seconds("01:02")


def test_format_time():
    assert format_time(3723.5) == "01:02:03.500"


def test_der_is_zero_for_identical_diarization():
    ref = turns((0, 10, "A"), (10, 20, "B"))

    _, errors = compute_der(ref, ref.copy())

    assert errors["DER"] == 0
    assert errors["Reference Speech Duration"] == 20


def test_der_counts_speaker_confusion():
    ref = turns((0, 10, "A"), (10, 20, "B"))
    pred = turns((0, 10, "A"), (10, 20, "A"))

    segments, errors = compute_der(ref, pred)

    assert errors["Confusion Duration"] == 10
    assert errors["DER"] == 0.5
    assert segments["Error Type"].tolist() == ["", "Confusion"]


def test_der_counts_missed_speech_without_tolerance():
    ref = turns((0, 10, "A"), (10, 20, "B"))
    pred = turns((0, 10, "A"))

    _, errors = compute_der(ref, pred, tolerance=0)

    assert errors["Missed Duration"] == 10
    assert errors["DER"] == 0.5


def test_der_tolerance_ignores_small_boundary_shifts():
    ref = turns((0, 10, "A"))
    pred = turns((0, 9.5, "A"))

    _, errors = compute_der(ref, pred, tolerance=1.0)

    assert errors["Missed Duration"] == 0


def test_der_uses_given_total_duration():
    ref = turns((0, 10, "A"), (10, 20, "B"))
    pred = turns((0, 10, "A"), (10, 20, "A"))

    _, errors = compute_der(ref, pred, total_ref_duration=40)

    assert errors["DER"] == 0.25


def test_load_data_time_adds_seconds_columns(tmp_path):
    REFERENCE.to_csv(tmp_path / "ref.csv", index=False)
    REFERENCE.to_csv(tmp_path / "pred.csv", index=False)

    df_ref, df_pred = load_data_time(tmp_path / "ref.csv", tmp_path / "pred.csv")

    assert df_ref["Start"].tolist() == [0.0, 5.0]
    assert df_pred["End"].tolist() == [5.0, 10.0]


def test_extract_text_wer_strips_timestamps_and_speakers(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text(
        "00:00:01,000 --> 00:00:02,000\n[Interviewer]: Hello there.\n\n[Participant]: Hi.",
        encoding="utf-8",
    )

    assert extract_text_WER(path).split() == ["Hello", "there.", "Hi."]


def test_wer_is_zero_for_identical_transcripts(tmp_path, nltk_punkt):
    REFERENCE.to_csv(tmp_path / "ref.csv", index=False)

    metrics = calculate_wer_and_generate_html(
        tmp_path / "ref.csv", tmp_path / "ref.csv", tmp_path / "out.html"
    )

    assert metrics["WER"] == 0
    assert (tmp_path / "out.html").exists()


def test_wer_detects_a_wrong_word(tmp_path, nltk_punkt):
    REFERENCE.to_csv(tmp_path / "ref.csv", index=False)
    pred = REFERENCE.copy()
    pred.loc[1, "Content"] = "I went to my house and worked."
    pred.to_csv(tmp_path / "pred.csv", index=False)

    metrics = calculate_wer_and_generate_html(
        tmp_path / "pred.csv", tmp_path / "ref.csv", tmp_path / "out.html"
    )

    assert metrics["Substitutions"] == 1
    assert metrics["Deletions"] == metrics["Insertions"] == 0
    assert metrics["WER"] == round(1 / metrics["Total Words"], 4)


def test_process_folder_csv_combines_wer_and_der(tmp_path, nltk_punkt):
    (tmp_path / "ref").mkdir()
    (tmp_path / "pred").mkdir()
    REFERENCE.to_csv(tmp_path / "ref" / "P1.csv", index=False)
    REFERENCE.to_csv(tmp_path / "pred" / "P1.csv", index=False)

    metrics = process_folder_csv(str(tmp_path / "pred"), str(tmp_path / "ref"))

    assert metrics["Filename"].tolist() == ["P1"]
    assert metrics.loc[0, "WER"] == 0
    assert metrics.loc[0, "DER"] == 0
    assert (tmp_path / "pred" / "visual_comparison" / "P1_WER.html").exists()
    assert (tmp_path / "pred" / "visual_comparison" / "P1_Diarization.html").exists()
