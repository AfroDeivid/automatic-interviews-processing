import argparse
import time
import torch
from transformers import SeamlessM4Tv2ForTextToText, AutoProcessor

from utils.translation_helpers import docx_to_csv, translate_by_row_csv, get_files

def translate_csv_file(csv_file, source_lang, target_lang, model, processor, use_cuda):
    """Translate a single CSV file."""
    print(f"Translating {csv_file}...")

    # Start the timer
    start_time = time.time()

    # Translate the CSV file
    translate_by_row_csv(csv_file, source_lang, target_lang, model, processor, use_cuda)

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished translating {csv_file} in {int(elapsed_time // 60)} min and {elapsed_time % 60:.0f} sec")

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
        choices=["docx", "csv"],
        required=True,
        help="Type of source files: 'docx' to convert Word type files to CSV and optionally translate ; 'csv' to translate existing CSV files."
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

    # Load the model and processor if we perform translation, if not skip this step
    if not args.no_translate:
        model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")
        processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        if use_cuda:
            model = model.to("cuda")
    else:
        model = None
        processor = None


    csv_files = []

    # Step 1: Convert Word files to CSV if type is 'docx', otherwise get CSV files
    if args.type == "docx":
        files = get_files(args.directory, [".docx"])
        print(files, "\n")
        for word_file in files:

            print(f"Converting {word_file} to CSV...")
            csv_file = docx_to_csv(word_file)
            print(f"Successfully converted {word_file} to {csv_file} \n")
            csv_files.append(csv_file)

    elif args.type == "csv":
        csv_files = get_files(args.directory, [".csv"])

    print("Parsed directory: ", args.directory)
    print("Type: ", args.type)
    print("Parsed files: ", files)
    print("No translation: ", args.no_translate)
    if not args.no_translate:
        print("Parsed source language: ", args.source_lang)
        print("Parsed target language: ", args.target_lang)
        print("Use CUDA: ", use_cuda)
    
    # Step 2: Translate CSV files if translation is not skipped
    if not args.no_translate:
        for csv_file in csv_files:
            translate_csv_file(csv_file, args.source_lang, args.target_lang, model, processor, use_cuda)

if __name__ == "__main__":
    main()