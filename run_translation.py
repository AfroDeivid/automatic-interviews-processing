import os
import argparse
import time
import torch
from transformers import SeamlessM4Tv2ForTextToText, AutoProcessor

import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "Translation"))
sys.path.append(src_path)

from translation_helpers import extract_dialogue_from_docx, save_to_csv, translate_csv

def convert_word_to_csv(word_file):
    """Convert a single Word  to a CSV file."""
    print(f"Converting {word_file} to CSV...")

    # Extract dialogues from the Word file
    dialogues = extract_dialogue_from_docx(word_file)

    # Save dialogues to a CSV file
    csv_file = f"csv/{os.path.splitext(os.path.basename(word_file))[0]}.csv"
    save_to_csv(dialogues, csv_file)

    print(f"Successfully converted {word_file} to {csv_file}")
    return csv_file

def translate_csv_file(csv_file, source_lang, target_lang, model, processor, use_cuda):
    """Translate a single CSV file."""
    print(f"Translating {csv_file}...")

    # Start the timer
    start_time = time.time()

    # Translate the CSV file
    translate_csv(csv_file, source_lang, target_lang, model, processor, use_cuda)

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished translating {csv_file} in {int(elapsed_time // 60)} min and {elapsed_time % 60:.0f} sec")

def get_files(directory, extensions):
    """Get a list of files in the specified directory with given extensions."""
    files = []

    for root, dirs, files_in_dir in os.walk(directory):
        for file in files_in_dir:
            if any(file.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, file))
                
    return files

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process files for translation.")
    parser.add_argument(
        "-d", "--directory",
        type=str,
        required=True,
        help="Directory containing files."
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["word", "csv"],
        required=True,
        help="Type of source files: 'word' to convert Word files (.docx) to CSV and optionally translate, 'csv' to translate existing CSV files."
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default="fra",
        help="Source language for translation."
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="eng",
        help="Target language for translation."
    )
    parser.add_argument(
        "--use-cpu",
        action='store_true',
        help="Use CPU instead of CUDA if available."
    )
    parser.add_argument(
        "--no-translate",
        action='store_true',
        help="Only convert files to CSV format without translating."
    )

    args = parser.parse_args()

    # Determine whether to use CUDA or CPU
    use_cuda = not args.use_cpu and torch.cuda.is_available()

    # Load the model and processor if translation is not skipped
    if not args.no_translate:
        model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")
        processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

        if use_cuda:
            model = model.to("cuda")
    else:
        model = None
        processor = None


    csv_files = []

    # Step 1: Convert Word files to CSV if type is 'word'
    if args.type == "word":
        files = get_files(args.directory, [".docx"])
        for word_file in files:
            csv_file = convert_word_to_csv(word_file)
            if csv_file:
                csv_files.append(csv_file)

    elif args.type == "csv":
        csv_files = get_files(args.directory, [".csv"])

    print("Parsed directory: ", args.directory)
    print("Type: ", args.type)
    print("Parsed files: ", files)
    print("No translation: ", args.no_translate)
    print("Parsed source language: ", args.source_lang)
    print("Parsed target language: ", args.target_lang)
    print("Use CUDA: ", use_cuda)
    
    # Step 2: Translate CSV files if translation is not skipped
    if not args.no_translate:
        for csv_file in csv_files:
            translate_csv_file(csv_file, args.source_lang, args.target_lang, model, processor, use_cuda)

if __name__ == "__main__":
    main()