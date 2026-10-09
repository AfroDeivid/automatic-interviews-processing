import io

import pytest

from interviews_processing.whisper_diarization.helpers import (
    filter_missing_timestamps,
    format_timestamp,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    process_language_arg,
    write_srt,
)


def test_words_are_mapped_to_the_speaker_turn_they_start_in():
    words = [
        {"start": 0.0, "end": 0.5, "text": "Hello"},
        {"start": 0.6, "end": 1.0, "text": "there."},
        {"start": 2.1, "end": 2.5, "text": "Hi."},
    ]
    turns = [[0, 1500, 0], [1500, 3000, 1]]

    mapping = get_words_speaker_mapping(words, turns)

    assert [m["speaker"] for m in mapping] == [0, 0, 1]
    assert mapping[0] == {"word": "Hello", "start_time": 0, "end_time": 500, "speaker": 0}


def test_words_after_the_last_turn_go_to_the_last_speaker():
    words = [{"start": 5.0, "end": 5.5, "text": "late"}]
    turns = [[0, 1000, 0], [1000, 2000, 1]]

    assert get_words_speaker_mapping(words, turns)[0]["speaker"] == 1


def test_sentences_split_on_speaker_change():
    mapping = [
        {"word": "Hello", "start_time": 0, "end_time": 500, "speaker": 0},
        {"word": "there.", "start_time": 600, "end_time": 1000, "speaker": 0},
        {"word": "Hi.", "start_time": 2100, "end_time": 2500, "speaker": 1},
    ]

    sentences = get_sentences_speaker_mapping(mapping, [[0, 1500, 0], [1500, 3000, 1]])

    assert [s["speaker"] for s in sentences] == ["Speaker 0", "Speaker 1"]
    assert sentences[0]["text"].strip() == "Hello there."
    assert (sentences[0]["start_time"], sentences[0]["end_time"]) == (0, 1000)
    assert sentences[1]["text"].strip() == "Hi."


def test_realignment_gives_a_sentence_its_majority_speaker():
    mapping = [
        {"word": "I", "start_time": 0, "end_time": 100, "speaker": 0},
        {"word": "like", "start_time": 100, "end_time": 200, "speaker": 0},
        {"word": "it.", "start_time": 200, "end_time": 300, "speaker": 1},
    ]

    realigned = get_realigned_ws_mapping_with_punctuation(mapping)

    assert [m["speaker"] for m in realigned] == [0, 0, 0]
    assert mapping[2]["speaker"] == 1  # input is not mutated


def test_speaker_aware_transcript_starts_a_paragraph_per_speaker(segments):
    out = io.StringIO()

    get_speaker_aware_transcript(segments, out)

    paragraphs = out.getvalue().split("\n\n")
    assert len(paragraphs) == 2
    assert paragraphs[0].startswith("Speaker 0: Can you tell me")
    assert paragraphs[1].startswith("Speaker 1: I went")
    assert "It was fine." in paragraphs[1]


@pytest.mark.parametrize(
    "ms, kwargs, expected",
    [
        (0, {}, "00:00:00,000"),
        (3_723_004, {}, "01:02:03,004"),
        (61_005, {"always_include_hours": False}, "01:01,005"),
        (61_005, {"decimal_marker": "."}, "00:01:01.005"),
    ],
)
def test_format_timestamp(ms, kwargs, expected):
    assert format_timestamp(ms, **kwargs) == expected


def test_format_timestamp_rejects_negative_values():
    with pytest.raises(AssertionError):
        format_timestamp(-1)


def test_write_srt(segments):
    out = io.StringIO()

    write_srt(segments[:1] + [{**segments[1], "text": "a --> b "}], out)

    assert out.getvalue() == (
        "1\n00:00:00,000 --> 00:00:01,500\nSpeaker 0: Can you tell me about your day?\n\n"
        "2\n00:00:02,000 --> 00:00:05,250\nSpeaker 1: a -> b\n\n"
    )


def test_missing_timestamps_are_filled_from_neighbours():
    words = [
        {"word": "a", "start": 0.0, "end": 0.5},
        {"word": "b"},
        {"word": "c", "start": 1.0, "end": 1.5},
    ]

    result = filter_missing_timestamps(words)

    assert [w["word"] for w in result] == ["a", "b", "c"]
    assert (result[1]["start"], result[1]["end"]) == (0.5, 1.0)


def test_missing_first_timestamp_uses_initial_timestamp():
    words = [{"word": "a"}, {"word": "b", "start": 1.0, "end": 1.5}]

    result = filter_missing_timestamps(words, initial_timestamp=0.2)

    assert (result[0]["start"], result[0]["end"]) == (0.2, 1.0)


@pytest.mark.parametrize(
    "language, model, expected",
    [
        ("en", "large-v3", "en"),
        ("FR", "large-v3", "fr"),
        ("english", "large-v3", "en"),
        ("en", "medium.en", "en"),
        (None, "large-v3", None),
    ],
)
def test_process_language_arg(language, model, expected):
    assert process_language_arg(language, model) == expected


@pytest.mark.parametrize(
    "language, model",
    [("klingon", "large-v3"), ("fr", "medium.en")],
)
def test_process_language_arg_rejects_invalid_combinations(language, model):
    with pytest.raises(ValueError):
        process_language_arg(language, model)
