"""
How to use:
1) Preview only (safe, no changes):
    python placeholder_audio_from_transcripts.py

2) Apply changes (replace completed audio with empty placeholder files):
    python placeholder_audio_from_transcripts.py --apply

Rules:
- data/grief_eng audio: requires one transcript CSV in results/grief_eng
- data/grief_fr audio: requires two transcript CSVs:
  1) results/grief_fr
  2) results/grief_fr/fr_to_eng
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class CheckResult:
    audio_path: Path
    rel_no_ext: Path
    language: str
    transcript_exists: bool
    translation_exists: bool
    completed: bool


def find_audio_files(root: Path, extensions: Iterable[str]) -> list[Path]:
    normalized_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in normalized_exts
    ]


def build_result_for_eng(audio_path: Path, audio_root: Path, results_root: Path) -> CheckResult:
    rel_no_ext = audio_path.relative_to(audio_root).with_suffix("")
    transcript_csv = results_root / rel_no_ext.parent / f"{rel_no_ext.name}.csv"
    transcript_exists = transcript_csv.exists()
    return CheckResult(
        audio_path=audio_path,
        rel_no_ext=rel_no_ext,
        language="eng",
        transcript_exists=transcript_exists,
        translation_exists=False,
        completed=transcript_exists,
    )


def build_result_for_fr(audio_path: Path, audio_root: Path, results_root: Path) -> CheckResult:
    rel_no_ext = audio_path.relative_to(audio_root).with_suffix("")
    transcript_csv = results_root / rel_no_ext.parent / f"{rel_no_ext.name}.csv"
    translated_csv = results_root / "fr_to_eng" / rel_no_ext.parent / f"{rel_no_ext.name}.csv"
    transcript_exists = transcript_csv.exists()
    translation_exists = translated_csv.exists()
    return CheckResult(
        audio_path=audio_path,
        rel_no_ext=rel_no_ext,
        language="fr",
        transcript_exists=transcript_exists,
        translation_exists=translation_exists,
        completed=transcript_exists and translation_exists,
    )


def make_placeholder(audio_path: Path) -> None:
    # Truncate in place to keep the same filename/path while freeing disk space.
    audio_path.write_bytes(b"")


def process_group(
    label: str,
    audio_root: Path,
    results_root: Path,
    extensions: list[str],
    dry_run: bool,
) -> tuple[int, int, int]:
    if not audio_root.exists():
        print(f"[{label}] Audio folder not found, skipping: {audio_root}")
        return 0, 0, 0

    audio_files = find_audio_files(audio_root, extensions)
    scanned = 0
    completed = 0
    replaced = 0

    for audio_file in audio_files:
        scanned += 1

        if label == "eng":
            check = build_result_for_eng(audio_file, audio_root, results_root)
        else:
            check = build_result_for_fr(audio_file, audio_root, results_root)

        if not check.completed:
            continue

        completed += 1

        current_size = audio_file.stat().st_size
        if current_size == 0:
            print(f"[{label}] Already placeholder: {audio_file}")
            continue

        if dry_run:
            print(f"[{label}] DRY-RUN would replace: {audio_file} ({current_size} bytes)")
            replaced += 1
            continue

        make_placeholder(audio_file)
        replaced += 1
        print(f"[{label}] Replaced with empty placeholder: {audio_file}")

    return scanned, completed, replaced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace audio files with empty placeholders when expected transcripts already exist. "
            "Rule: English audio requires one CSV; French audio requires both FR and FR->ENG CSVs."
        )
    )
    parser.add_argument(
        "--eng-audio-dir",
        type=Path,
        default=Path("data/grief_eng"),
        help="Path to English audio root.",
    )
    parser.add_argument(
        "--fr-audio-dir",
        type=Path,
        default=Path("data/grief_fr"),
        help="Path to French audio root.",
    )
    parser.add_argument(
        "--eng-results-dir",
        type=Path,
        default=Path("results/grief_eng"),
        help="Path to English transcript results root.",
    )
    parser.add_argument(
        "--fr-results-dir",
        type=Path,
        default=Path("results/grief_fr"),
        help="Path to French transcript results root (must contain fr_to_eng for translated CSVs).",
    )
    parser.add_argument(
        "-e",
        "--extensions",
        nargs="+",
        default=[".wav", ".m4a", ".mp4"],
        help="Audio extensions to scan.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually modify files. Without this flag, the script runs in dry-run mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = not args.apply

    print("Mode:", "DRY-RUN" if dry_run else "APPLY")

    eng_scanned, eng_completed, eng_replaced = process_group(
        label="eng",
        audio_root=args.eng_audio_dir,
        results_root=args.eng_results_dir,
        extensions=args.extensions,
        dry_run=dry_run,
    )

    fr_scanned, fr_completed, fr_replaced = process_group(
        label="fr",
        audio_root=args.fr_audio_dir,
        results_root=args.fr_results_dir,
        extensions=args.extensions,
        dry_run=dry_run,
    )

    print("\nSummary")
    print(f"ENG scanned={eng_scanned} completed={eng_completed} placeholders_created={eng_replaced}")
    print(f"FR  scanned={fr_scanned} completed={fr_completed} placeholders_created={fr_replaced}")


if __name__ == "__main__":
    main()
