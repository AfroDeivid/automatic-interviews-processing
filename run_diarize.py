import os
import subprocess
import time
import argparse

def process_audio_file(audio_file, whisper_model, language):
    """Process a single audio file with the diarization script."""
    print(f"Processing {audio_file}...")

    # Start the timer
    start_time = time.time()

    # Run the Python script
    subprocess.run([
        "python", "Whisper_Diarization\diarize.py",
        "-a", audio_file,
        "--whisper-model", whisper_model,
        "--language", language
    ])

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished processing {audio_file} in {elapsed_time:.2f} seconds")

def get_audio_files(directory, extensions):
    """Get a list of audio files in the specified directory with given extensions."""
    audio_files = []
    for ext in extensions:
        audio_files.extend(os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(ext))
    return audio_files

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process audio files for diarization.")
    parser.add_argument(
        "-d", "--directory",
        type=str,
        required=True,
        help="Directory containing audio files."
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3",
        help="Whisper model to use."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language for processing."
    )
    parser.add_argument(
        "-e", "--extensions",
        type=str,
        nargs='+', # can give multiples argumens separate by an space
        default=[".m4a",".mp4",".wav"],
        help="List of allowed audio file extensions."
    )

    args = parser.parse_args()

    # Get audio files
    audio_files = get_audio_files(args.directory, args.extensions)
    print("Parse dir: ", args.directory)
    print("Parse audio: ", audio_files)
    print("Parse extensions: ", args.extensions)
    print("Parse Language: ", args.language)

    # Process each audio file
    for audio_file in audio_files:
        process_audio_file(audio_file, args.whisper_model, args.language)

if __name__ == "__main__":
    main()
