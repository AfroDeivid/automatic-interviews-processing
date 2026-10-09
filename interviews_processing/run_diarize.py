import os
import subprocess
import time
import argparse
import sys
import psutil
import gc
import csv
from pathlib import Path
from datetime import datetime

from interviews_processing.utils.format_helpers import get_files, convert_str_to_csv
from interviews_processing.utils.preprocessing_helpers import preprocessing_csv

# ======================================================
# Logging utilities (CSV structured log)
# ======================================================

LOG_FILE = "processing_log.csv"


def init_log(log_path=LOG_FILE):
    """Create CSV log file with header if it does not exist."""
    log_path = Path(log_path)
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["started_at", "file", "status", "message"])


def write_csv_log(audio_file, status, message="", started_at="", log_path=LOG_FILE):
    """Append a row to the CSV log."""
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([started_at, str(audio_file), status, message])


def log_memory_usage(info=""):
    process = psutil.Process()
    rss = process.memory_info().rss / 1024**2  # MB
    print(f"[MEM] {info}: RSS={rss:.1f} MB")


# ======================================================
# Core processing
# ======================================================

def process_audio_file(
    audio_file,
    directory,
    whisper_model,
    language,
    task=None,
    overwrite=False,
    diarizer="msdd",
):
    """
    Process a single audio file.
    Returns: (status, message, str_file or None)
    """

    started_at = datetime.now().isoformat(timespec="seconds")

    experiment_name = os.path.basename(os.path.normpath(directory)) # Extract only the last folder name
    relative_path = os.path.relpath(audio_file, directory)

    if task == "translate":
        output_dir = os.path.join(
            "results",
            experiment_name,
            f"{language}_to_eng",
            os.path.dirname(relative_path),
        )
    else:
        output_dir = os.path.join(
            "results",
            experiment_name,
            os.path.dirname(relative_path),
        )

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_file))[0] # Get the file name without extension
    str_file = os.path.join(output_dir, f"{base_name}.str")
    csv_file = str_file.replace(".str", ".csv")

    # ---------------- SKIP ----------------
    if not overwrite and os.path.exists(csv_file):
        msg = "CSV already exists"
        print(f"Skipping {audio_file}: {msg}")
        write_csv_log(audio_file, "SKIPPED", msg, started_at=started_at)
        return "SKIPPED", msg, None

    print(f"Processing {audio_file}...")

    command = [
        sys.executable,
        os.path.join(
            os.path.dirname(__file__),
            "whisper_diarization",
            "diarize.py",
        ),
        "-a", audio_file,
        "-d", output_dir,
        "--whisper-model", whisper_model,
        "--language", language,
        "--task", task,
        "--diarizer", diarizer,
    ]

    start_time = time.time()

    try:
        subprocess.run(command, check=True)

        elapsed = time.time() - start_time
        # show in hours, minutes, seconds
        msg = f"Completed in {int(elapsed // 3600)} hr {int((elapsed % 3600) // 60)} min {elapsed % 60:.0f} sec"

        convert_str_to_csv(str_file, experiment_name)

        write_csv_log(audio_file, "SUCCESS", msg, started_at=started_at)
        log_memory_usage(f"{audio_file}")
        return "SUCCESS", msg, str_file

    # ---------------- WHISPER FAILURE ----------------
    except subprocess.CalledProcessError as e:
        msg = (
            f"Error: Command '{e.cmd}' "
            f"returned non-zero exit status {e.returncode}."
        )
        print(msg)
        write_csv_log(audio_file, "FAILED", msg, started_at=started_at)
        log_memory_usage(f"{audio_file}")
        return "FAILED", msg, None

    # ---------------- UNEXPECTED FAILURE ----------------
    except Exception as e:
        msg = f"Unexpected error: {repr(e)}"
        print(msg)
        write_csv_log(audio_file, "FAILED", msg, started_at=started_at)
        log_memory_usage(f"{audio_file}")
        return "FAILED", msg, None


# ======================================================
# Main
# ======================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process audio files for diarization."
    )
    parser.add_argument(
        "-d", "--directory",
        type=str,
        required=True,
        help="Directory containing audio files.",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3",
        help="Whisper model to use.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language spoken in the audio. If the detected language is different "
            "from the specified language and 'task'=None or 'transcribe', "
            "the model will translate the audio to the specified language, otherwise if task='translate' it will translate the audio to English.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task to execute (transcribe or translate). Specify None to follow the 'language' argument; "
            "it will translate when the audio does not match the specified language, useful for multilingual audios. "
            "If the audio is entirely in another language and you want to translate to English (Whisper's best performance), "
            "you can use the 'translate' task.",
    )
    parser.add_argument(
        "--also-transcribe",
        action="store_true",
        help="When task is 'translate', also transcribe in original language.",
    )
    parser.add_argument(
        "-e", "--extensions",
        type=str,
        nargs="+", # can give multiples arguments separate by a space
        default=[".m4a", ".mp4", ".wav"],
        help="Allowed audio file extensions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing transcriptions.",
    )
    parser.add_argument(
        "--diarizer",
        type=str,
        default="msdd",
        choices=["msdd", "sortformer"],
        help="Speaker diarization model: 'msdd' (NeMo MSDD, any number of speakers) or "
            "'sortformer' (NeMo Streaming Sortformer, newer, at most 4 speakers).",
    )

    args = parser.parse_args()

    # Init CSV log
    init_log()

    audio_files = get_files(args.directory, args.extensions)

    print("Parse dir:", args.directory)
    print("Parse audio:", audio_files)
    print("Parse extensions:", args.extensions)
    print("Parse language:", args.language)
    print("Parse task:", args.task)
    print("Parse diarizer:", args.diarizer)

    for audio_file in audio_files:

        # ---------- OPTIONAL ORIGINAL TRANSCRIPTION ----------
        if args.task == "translate" and args.also_transcribe:
            status, msg, str_file = process_audio_file(
                audio_file,
                args.directory,
                args.whisper_model,
                args.language,
                task="transcribe",
                overwrite=args.overwrite,
                diarizer=args.diarizer,
            )

            if status == "SUCCESS":
                preprocessing_csv(str_file)

        time.sleep(2)
        gc.collect()

        # ---------- MAIN TASK ----------
        status, msg, str_file = process_audio_file(
            audio_file,
            args.directory,
            args.whisper_model,
            args.language,
            task="translate" if args.task == "translate" else args.task,
            overwrite=args.overwrite,
            diarizer=args.diarizer,
        )

        if status == "SUCCESS":
            preprocessing_csv(str_file)


if __name__ == "__main__":
    main()