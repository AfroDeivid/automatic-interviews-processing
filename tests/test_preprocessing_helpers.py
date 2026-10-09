import math

import pandas as pd
import pytest

from interviews_processing.utils.format_helpers import convert_str_to_csv
from interviews_processing.utils.preprocessing_helpers import (
    add_turn_index,
    assign_roles,
    extract_part_number,
    merge_csv_parts_and_copy_others,
    preprocessing_csv,
    simpler_clean,
    visual_clean,
)
from interviews_processing.whisper_diarization.helpers import write_srt


@pytest.mark.parametrize(
    "text, expected",
    [
        ("i think  so", "I think so."),
        ("what is it?", "What is it ?"),
        ("done. next one", "Done. Next one."),
    ],
)
def test_visual_clean(text, expected):
    assert visual_clean(text) == expected


@pytest.mark.parametrize("text", ["", "   ", None])
def test_visual_clean_returns_none_for_empty_text(text):
    assert visual_clean(text) is None


def test_simpler_clean_removes_fillers_and_repeated_words():
    assert simpler_clean("um I uh think think so", ["um", "uh"]) == "I think so."


def test_assign_roles_picks_the_first_person_speaker_as_participant():
    df = pd.DataFrame({
        "Speaker": [0, 1, 0, 1],
        "Content": [
            "Can you tell me about your day?",
            "I went to my office and I worked.",
            "How did it feel?",
            "My day was fine, I think.",
        ],
    })

    df_roles, _ = assign_roles(df)

    assert df_roles["Role"].tolist() == ["Interviewer", "Participant", "Interviewer", "Participant"]


def test_assign_roles_numbers_multiple_interviewers():
    df = pd.DataFrame({
        "Speaker": [0, 1, 2],
        "Content": ["Could you describe it?", "I saw myself from above.", "Do you remember how?"],
    })

    df_roles, _ = assign_roles(df)

    assert df_roles["Role"].tolist() == ["Interviewer 1", "Participant", "Interviewer 2"]


def test_add_turn_index():
    df = pd.DataFrame({"Speaker": ["A", "A", "B", "A"]})

    assert add_turn_index(df)["turn_index"].tolist() == [0, 0, 1, 2]


def test_extract_part_number():
    assert extract_part_number("P1 Part 3.csv") == 3
    assert extract_part_number("P1_part12.csv") == 12
    assert math.isinf(extract_part_number("P1 follow-up.csv"))


def test_merge_csv_parts_in_order_and_copy_others(tmp_path):
    src = tmp_path / "src" / "P1"
    src.mkdir(parents=True)
    pd.DataFrame({"Content": ["second"]}).to_csv(src / "P1_part2.csv", index=False)
    pd.DataFrame({"Content": ["first"]}).to_csv(src / "P1_part1.csv", index=False)
    pd.DataFrame({"Content": ["other"]}).to_csv(src / "P1_followup.csv", index=False)

    merge_csv_parts_and_copy_others(str(tmp_path / "src"), str(tmp_path / "out"))

    merged = pd.read_csv(tmp_path / "out" / "P1" / "P1_interview_merged.csv")
    assert merged["Content"].tolist() == ["first", "second"]
    assert (tmp_path / "out" / "P1" / "P1_followup.csv").exists()


def test_preprocessing_csv_writes_processed_csv_and_dialogue(tmp_path, monkeypatch, segments):
    monkeypatch.chdir(tmp_path)
    exp = tmp_path / "results" / "OBE"
    exp.mkdir(parents=True)
    filler_segment = {"speaker": "Speaker 1", "start_time": 6100, "end_time": 6200, "text": "um"}
    with open(exp / "P1.str", "w", encoding="utf-8-sig") as f:
        write_srt(segments + [filler_segment], f)
    convert_str_to_csv(str(exp / "P1.str"), "OBE")

    preprocessing_csv("results/OBE/P1.str")

    processed = tmp_path / "results" / "processed" / "OBE"
    df = pd.read_csv(processed / "P1.csv")
    assert df["Speaker"].tolist() == ["Interviewer", "Participant", "Participant"]  # filler-only row dropped
    dialogue = (processed / "P1.txt").read_text(encoding="utf-8")
    assert dialogue.startswith("[Interviewer]: Can you tell me about your day ?")
    assert "[Participant]: I went to my office and I worked. It was fine." in dialogue
