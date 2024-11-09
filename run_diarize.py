import os
import subprocess
import time
import argparse

from utils.format_helpers import get_files, convert_str_to_csv

def process_audio_file(audio_file, whisper_model, language):
    """Process a single audio file with the diarization script."""
    print(f"Processing {audio_file}...")

    # Start the timer
    start_time = time.time()

    # Run the Python script
    subprocess.run([
        "python", "src\whisper_diarization\diarize.py",
        "-a", audio_file,
        "--whisper-model", whisper_model,
        "--language", language
    ])

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n\n Finished processing {audio_file} in {int(elapsed_time // 60)} min and {elapsed_time % 60:.0f} sec")

    # Convert .str file to .csv format
    base_input_directory = 'data'
    relative_path = os.path.relpath(audio_file, base_input_directory)  
    str_dir = os.path.join("results", os.path.dirname(relative_path)) 
    #print(".str Dir:", str_dir)
    base_name = os.path.splitext(os.path.basename(audio_file))[0]  # Get the file name without extension
    str_file = os.path.join(str_dir,f"{base_name}.str")

    convert_str_to_csv(str_file, str_dir)


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
        help="The language spoken in the audio. If the detected language is different "
            "from the specified language and 'task'=None or 'transcribe', "
            "the model will translate the audio to the specified language, otherwise if task='translate' it will translate the audio to English."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["transcribe", "translate"],
        help="Task to execute (transcribe or translate). Specify None to follow the 'language' argument; "
            "it will translate when the audio does not match the specified language, useful for multilingual audios. "
            "If the audio is entirely in another language and you want to translate to English (Whisper's best performance), "
            "you can use the 'translate' task."
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
    audio_files = get_files(args.directory, args.extensions)
    print("Parse dir: ", args.directory)
    print("Parse audio: ", audio_files)
    print("Parse extensions: ", args.extensions)
    print("Parse Language: ", args.language)

    # Process each audio file
    for audio_file in audio_files:
        process_audio_file(audio_file, args.whisper_model, args.language)

if __name__ == "__main__":
    main()