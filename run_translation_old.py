import os
import argparse
import time
import torch
from transformers import SeamlessM4Tv2ForTextToText, AutoProcessor

import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "Translation"))
sys.path.append(src_path)

from translation_helpers import extract_dialogue_from_docx, save_to_csv, translate_csv

def process_word_file(word_file, source_lang, target_lang, model, processor, use_cuda, translate):
    """Process a single Word file for translation."""
    print(f"Processing {word_file}...")

    # Start the timer
    start_time = time.time()

    # Extract dialogues from the Word file
    dialogues = extract_dialogue_from_docx(word_file)

    # Save dialogues to a CSV file
    csv_file = f"csv/{os.path.splitext(os.path.basename(word_file))[0]}.csv"
    save_to_csv(dialogues, csv_file)

    # Translate the CSV file if translation is enabled
    if translate:
        translate_csv(csv_file, source_lang, target_lang, model, processor, use_cuda)

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished processing {word_file} in {int(elapsed_time // 60)} min and {elapsed_time % 60:.0f} sec")

def process_csv_file(csv_file, source_lang, target_lang, model, processor, use_cuda):
    """Process a single CSV file for translation."""
    print(f"Processing {csv_file}...")

    # Start the timer
    start_time = time.time()

    # Translate the CSV file
    translate_csv(csv_file, source_lang, target_lang, model, processor, use_cuda)

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished processing {csv_file} in {int(elapsed_time // 60)} min and {elapsed_time % 60:.0f} sec")

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
        help="Only convert files to CSV without translating."
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

    # Get files based on the type
    if args.type == "word":
        files = get_files(args.directory, [".docx"])
    elif args.type == "csv":
        files = get_files(args.directory, [".csv"])

    print("Parsed directory: ", args.directory)
    print("Type: ", args.type)
    print("Parsed files: ", files)
    print("No translation: ", args.no_translate)
    print("Parsed source language: ", args.source_lang)
    print("Parsed target language: ", args.target_lang)
    print("Use CUDA: ", use_cuda)

    # Process each file based on the type
    for file_path in files:
        if args.type == "word":
            process_word_file(file_path, args.source_lang, args.target_lang, model, processor, use_cuda, not args.no_translate)
        elif args.type == "csv":
            process_csv_file(file_path, args.source_lang, args.target_lang, model, processor, use_cuda)

if __name__ == "__main__":
    main()