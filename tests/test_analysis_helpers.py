import pandas as pd
import pytest

# Needs the `analysis` extra: uv sync --extra analysis
pytest.importorskip("spacy")
pytest.importorskip("wordcloud")
pytest.importorskip("networkx")

from interviews_processing.utils.analysis_helpers import (  # noqa: E402
    aggregate_counts,
    build_network_from_interviews,
    count_unique_words,
    count_word_frequencies,
    preprocess_text,
    standardize_speaker_labels,
    tag_topic,
)


def test_standardize_speaker_labels_merges_interviewers():
    df = pd.DataFrame({"Speaker": ["Interviewer 1", "Participant", "Interviewer 2", "Interviewer"]})

    out = standardize_speaker_labels(df)

    assert out["Speaker"].tolist() == ["Interviewer", "Participant", "Interviewer", "Interviewer"]
    assert out["Speaker_original"].tolist()[2] == "Interviewer 2"


def test_aggregate_counts():
    df = pd.DataFrame({"Id": [1, 1, 2], "Speaker": ["P", "P", "P"], "Word Count": [3, 4, 5]})

    out = aggregate_counts(df, ["Id"])

    assert out.set_index("Id")["Word Count"].to_dict() == {1: 7, 2: 5}


def test_preprocess_text_lemmatizes_and_drops_stopwords():
    tokens = preprocess_text("The cats were running quickly!").split()

    assert "cat" in tokens
    assert "run" in tokens
    assert "the" not in tokens
    assert "!" not in tokens


def test_preprocess_text_can_retain_stopwords():
    assert "not" in preprocess_text("I do not know", retain_stopwords={"not"}).split()
    assert preprocess_text("") == ""


def test_count_word_frequencies_grouped_and_normalized():
    df = pd.DataFrame({"Id": [1, 1, 2], "preprocessed_content": ["body float", "body", "light"]})

    out = count_word_frequencies(df, groupby_columns=["Id"], normalize=True)

    freqs = {(row.Id, row.Word): row.Frequency for row in out.itertuples()}
    assert freqs[(1, "body")] == pytest.approx(2 / 3)
    assert freqs[(2, "light")] == 1.0


def test_count_unique_words_counts_participants_not_mentions():
    df = pd.DataFrame({
        "Experiment": ["OBE"] * 3,
        "Id": [1, 1, 2],
        "preprocessed_content": ["body body", "body", "body light"],
    })

    out = count_unique_words(df, ["Experiment"])

    assert out.set_index("Word")["Participant_Count"].to_dict() == {"body": 2, "light": 1}


@pytest.fixture
def interviews():
    return pd.DataFrame({
        "File Name": ["P1", "P1", "P1", "P2", "P2"],
        "turn_index": [0, 1, 2, 0, 1],
        "one_topic_name": ["body", "light", "light", "body", "light"],
    })


def test_topic_network_aggregates_transitions(interviews):
    G = build_network_from_interviews(interviews)

    assert G["body"]["light"]["weight"] == 2
    assert G["light"]["light"]["weight"] == 1
    assert G.nodes["light"] == {"occurrence": 3, "appearance": 2}


def test_topic_network_without_self_loops(interviews):
    G = build_network_from_interviews(interviews, include_self_loops=False)

    assert not G.has_edge("light", "light")


def test_tag_topic_respects_exclusions_and_multiple_topics():
    df = pd.DataFrame({
        "one_topic": [3, 3, 5],
        "multiple_topics": [[3], [3], [5, 3]],
    })

    tag_topic(df, 3, exclude_indices=[1], name_tag="vision")
    assert df["tag"].tolist()[:2] == ["vision", None] or pd.isna(df.loc[1, "tag"])
    assert df.loc[0, "tag"] == "vision"

    tag_topic(df, 3, exclude_indices=[], name_tag="vision-any", multiple=True)
    assert df.loc[2, "tag"] == "vision-any"
