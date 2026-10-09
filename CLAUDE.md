# automatic-interviews-processing

Transcription + speaker diarization CLI for audio interviews (Whisper + NeMo MSDD), plus the evaluation, text-analysis and topic-modeling notebooks built on its outputs. The former standalone `lnco-transcribe` repo was merged here; the original conda-based semester project is tagged `v1.0-semester-project`.

## Layout
- `interviews_processing/` — flat top-level package.
  - `run_diarize.py` holds `main()`, exposed as the `interviews-transcribe` CLI. It calls `whisper_diarization/diarize.py` in a subprocess per audio file (by file path, so `diarize.py` uses script-relative imports: `from helpers import ...`, `from diarization import ...`).
  - `whisper_diarization/` — adapted Whisper-Diarization code; the MSDD YAML lives in `diarization/msdd/`.
  - `utils/` — `format_helpers`, `preprocessing_helpers` (used by the CLI) and `evaluation_helpers`, `analysis_helpers`, `text_html` (used by the notebooks; need the `analysis` extra).
- `notebooks/` — research notebooks + Streamlit app; they import `interviews_processing.utils.*` and use paths relative to `notebooks/`.
- `scripts/` — dataset-specific file utilities with hard-coded paths.
- `tests/` — pytest suite.
- `requirements.txt` / `constraints.txt` — pip-based fallback (kept independent of the uv workflow).

## Dependency management (uv)
- `uv sync` — transcription dependencies; `uv sync --extra analysis` adds the notebook dependencies (incl. the spaCy `en_core_web_sm` wheel).
- `uv run interviews-transcribe ...` — run the CLI inside the managed environment.
- `uv lock` after changing dependencies in `pyproject.toml`; `uv add` / `uv remove` update both.

Four dependencies are pinned to git repos (`demucs`, `deepmultilingualpunctuation`, `ctc-forced-aligner`, `indic-numtowords`), declared in `[tool.uv.sources]`; the `dependencies` list only carries the plain names. Build backend is hatchling with `packages = ["interviews_processing"]`.

## Tests
- `uv run pytest` — fast unit tests, no models needed. Analysis tests are skipped if the `analysis` extra is missing; WER tests download NLTK `punkt_tab` on first run.
- `uv run pytest -m slow` with `AIP_TEST_AUDIO=<short speech clip>` — real end-to-end transcription.

## Conventions
- The CLI writes to `results/` and `processing_log.csv` relative to the current working directory.