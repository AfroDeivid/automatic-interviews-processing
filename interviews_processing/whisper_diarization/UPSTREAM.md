# Vendored code: whisper-diarization

The code in this folder is adapted from [MahmoudAshraf97/whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization) by Mahmoud Ashraf, distributed under the BSD 2-Clause License (see [LICENSE](LICENSE); its original README is kept as [README.md](README.md)).

It is not a pip-installable package, so it is copied here rather than installed as a dependency.

## History

- **2024-09** — first copied into this project (commit `25b12b4`, "whisper-diarization").
- **2025** — re-synced with the restructured upstream (`diarization/msdd/` package) in the former `lnco-transcribe` repository.
- **2026-10** — compared with upstream `main` at [`c6614d7`](https://github.com/MahmoudAshraf97/whisper-diarization/commit/c6614d717764592c4261f2590c1eb29e87948c3e) (2026-08-15), and added its Sortformer diarizer (`diarization/sortformer/`, unchanged). Apart from formatting, the differences are listed below.

## Local changes

- `diarize.py`
  - `-d/--directory`: write the outputs to a given folder (used by `run_diarize.py` for batch processing).
  - `--task transcribe|translate`: passed to Whisper, to translate non-English interviews to English.
  - Source separation runs through [`demucs_separate.py`](demucs_separate.py) with the current interpreter (`sys.executable`) instead of a bare `python -m demucs.separate`; the wrapper writes the stems with `soundfile`, because recent `torchaudio.save` needs a shared FFmpeg build on Windows.
  - Prints the language, model, device and task at start-up.
- `diarization/msdd/msdd.py`: writes the temporary mono WAV with `soundfile` (32-bit float). Upstream solved the same `torchaudio.save` problem with the standard `wave` module.
- `diarization/__init__.py`, `diarization/sortformer/sortformer.py`: identical to upstream.
- `helpers.py`: unchanged apart from formatting.

## Comparing with upstream later

```bash
git clone https://github.com/MahmoudAshraf97/whisper-diarization /tmp/whisper-diarization
git diff --no-index --ignore-all-space /tmp/whisper-diarization/diarize.py interviews_processing/whisper_diarization/diarize.py
```

When pulling in upstream changes, keep the local changes above and run the end-to-end test (`uv run pytest -m slow`, see the main README).
