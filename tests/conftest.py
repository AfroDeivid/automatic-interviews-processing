import os

import pytest

# Analysis helpers import matplotlib; never open a window during tests.
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(scope="session")
def nltk_punkt():
    """WER helpers use nltk.word_tokenize, which needs the punkt_tab data."""
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        if not nltk.download("punkt_tab", quiet=True):
            pytest.skip("NLTK punkt_tab data not available (offline?)")


@pytest.fixture
def segments():
    """Sentence-level segments as produced by get_sentences_speaker_mapping (times in ms)."""
    return [
        {"speaker": "Speaker 0", "start_time": 0, "end_time": 1500, "text": "Can you tell me about your day? "},
        {"speaker": "Speaker 1", "start_time": 2000, "end_time": 5250, "text": "I went to my office and I worked. "},
        {"speaker": "Speaker 1", "start_time": 5300, "end_time": 6000, "text": "It was fine. "},
    ]
