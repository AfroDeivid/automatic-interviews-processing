import csv
import docx # pip install python-docx
import unicodedata
import os
import torch
from tqdm import tqdm
import re

def get_files(directory, extensions):
    """Get a list of files in the specified directory and its subdirectories with given extensions."""
    files = []

    for root, dirs, files_in_dir in os.walk(directory):
        for file in files_in_dir:
            if any(file.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, file))
                
    return files

# Function docx_to_csv
def extract_dialogue_from_docx(docx_file):
    separator = ":"
    alternative_separators = [";","."]
    known_speakers = set()

    doc = docx.Document(docx_file)
    dialogues = []
    last_speaker = None

    # Loop through each paragraph in the document
    for paragraph in doc.paragraphs:
        text = unicodedata.normalize('NFKD', paragraph.text.strip())

        # Ignore empty lines
        if not text:
            continue

        # Check if the paragraph contains the separator (indicating dialogue)
        if separator in text:
            # Split speaker and text based on the delimiter
            parts = text.split(separator, 1)

            if len(parts) == 2:
                speaker = parts[0].strip()
                dialogue_text = parts[1].strip()
                
                # Check if the speaker's name consists of more than one word
                if len(speaker.split()) > 1 and speaker not in known_speakers:
                    print(f"Warning: The speaker '{speaker}' has more than one word. This might be an error.")
                    user_input = input(f"Do you recognize this speaker '{speaker}'? (y/n): ").strip().lower()
                    if user_input == "y":
                        known_speakers.add(speaker)
                        print("Thanks! We set this speaker as valid name.")
                    elif user_input == "n" and last_speaker:
                        # Reassign the text to the previous speaker
                        dialogues[-1]["Text"] += " " + text
                        print("Thanks! Therefore we assing this text to the previous speaker.")
                        continue

                # Maintain a list of known speakers
                known_speakers.add(speaker)

                dialogues.append({"Speaker": speaker, "Text": dialogue_text})
                last_speaker = speaker
        else:
            # Check for known speakers and alternative separators
            for speaker in known_speakers:
                if text.startswith(speaker):
                    # Check if any alternative separator is present after the speaker's name
                    remaining_text = text[len(speaker):].strip()
                    found_separator = None

                    for sep in alternative_separators:
                        if remaining_text.startswith(sep):
                            found_separator = sep
                            print(f"Warning: Detected alternative separator '{sep}' on line {len(dialogues)+2}. Assuming '{separator}' instead.")
                            break
                    
                    # Handle the text based on the identified separator
                    if found_separator:
                        dialogue_text = remaining_text[len(found_separator):].strip()
                    else:
                        # If no separator is found, assume the remaining text is dialogue text
                        print(f"Warning: Detected known speaker '{speaker}' without a separator on line {len(dialogues)+2}. Assuming missing separator.")
                        dialogue_text = remaining_text

                    dialogues.append({"Speaker": speaker, "Text": dialogue_text})
                    last_speaker = speaker
                    break
            else:
                # If no known speaker is found, assume text belongs to the last speaker
                if last_speaker:
                    dialogues[-1]["Text"] += " " + text

    return dialogues

def docx_to_csv(docx_file, output_directory="results", data_directory="data"):

    dialogues = extract_dialogue_from_docx(docx_file)

    # Save the extracted dialogues to a CSV file
    base_name = os.path.splitext(os.path.basename(docx_file))[0] # Get the file name without extension
    relative_path = os.path.relpath(docx_file, data_directory)  
    directory = os.path.join(output_directory, os.path.dirname(relative_path))
    csv_file = os.path.join(directory, f"{base_name}.csv")

    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Speaker", "Text"])
        writer.writeheader()
        for row in dialogues:
            # Replace newline characters with spaces
            row["Text"] = row["Text"].replace('\n',' ')
            writer.writerow(row)

    return csv_file

# Function translations
def translation(source_lang, target_lang, text, model, processor, use_cuda = True):
    if use_cuda:
        text_inputs = processor(text, return_tensors="pt", src_lang=source_lang).to("cuda")
    else:
        text_inputs = processor(text, return_tensors="pt", src_lang=source_lang)
        
    output_tokens = model.generate(**text_inputs, tgt_lang=target_lang)
    translated_text = processor.decode(output_tokens[0], skip_special_tokens=True)

    return translated_text

def split_text_into_chunks(text, max_tokens, processor, src_lang):
    """
    Splits a given text into smaller chunks of uterances (sentence) based on a maximum token limit.

    Parameters:
    - text (str): The text to be split into chunks.
    - max_tokens (int): Maximum number of tokens per chunk.
    - processor: Tokenizer/processor to estimate token counts.
    - src_lang (str): Source language code.

    Returns:
    - chunks (list of str): A list of text chunks that respect the token limit.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split text into sentences
    chunks = []
    current_chunk = ''

    for sentence in sentences:
        combined_text = (current_chunk + ' ' + sentence).strip()
        tokens = processor(combined_text, return_tensors="pt", src_lang=src_lang)
        num_tokens = len(tokens['input_ids'][0])

        if num_tokens <= max_tokens:
            current_chunk = combined_text  # Add sentence to current chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)  # Save the current chunk
            current_chunk = sentence.strip()  # Start a new chunk

    # Add the last chunk if any
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def translate_by_row_csv_with_chunking(input_csv, source_lang, target_lang, model, processor, max_tokens, use_cuda=True):
    """
    Translates the 'Text' column of a CSV file row by row, splitting long text into smaller chunks for translation.

    Parameters:
    - input_csv (str): Path to the input CSV file.
    - source_lang (str): Source language code.
    - target_lang (str): Target language code.
    - model: The translation model.
    - processor: Tokenizer/processor associated with the model.
    - max_tokens (int): Maximum number of tokens per chunk.
    - use_cuda (bool): Whether to use GPU acceleration.

    Output:
    - Saves a new CSV file with translated text while maintaining row structure.
    """
    encoding = 'utf-8'

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(input_csv), ".."))
    output_folder = os.path.join(parent_dir, f"translation_{source_lang}_to_{target_lang}")
    os.makedirs(output_folder, exist_ok=True)
    output_csv = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_csv))[0]}.csv")

    # Ensure CUDA is available if requested
    use_cuda = use_cuda and torch.cuda.is_available()

    # Count total rows for the progress bar
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        total_rows = sum(1 for _ in infile) - 1  # Subtract 1 for the header

    # Open the input file for reading
    with open(input_csv, mode='r', encoding=encoding) as infile:
        reader = csv.DictReader(infile)

        # Open the output file for writing
        with open(output_csv, mode='w', newline='', encoding=encoding) as outfile:
            writer = csv.DictWriter(outfile, fieldnames=["Speaker", "Text"])
            writer.writeheader()

            # Process each row individually
            for row in tqdm(reader, desc="Translating", total=total_rows):
                speaker = row["Speaker"]
                text = row["Text"]

                # Split the text into chunks
                #chunks = split_text_into_chunks(text, max_tokens, processor, source_lang)
                # # Translate each chunk
                # translated_chunks = []
                # for chunk in chunks:
                #     translated_chunk = translation(source_lang, target_lang, chunk, model, processor, use_cuda)
                #     translated_chunks.append(translated_chunk)
                # Reassemble the translated chunks
                #translated_text = ' '.join(translated_chunks)

                sentences = re.split(r'(?<=[.!?])\s+', text)  # Split text into sentences
                translated_sentences = []
                for sentence in sentences:
                    translated_sentence = translation(source_lang, target_lang, sentence, model, processor, use_cuda)
                    translated_sentences.append(translated_sentence)

                # Reassemble the translated_sentences
                translated_text = ' '.join(translated_sentences)

                # Write the speaker and translated text to the new CSV file immediately
                writer.writerow({"Speaker": speaker, "Text": translated_text})

    print(f"Translation completed. Output saved to {output_csv}")

def translate_folder(input_folder, source_lang, target_lang, model, processor, max_tokens, override=False, use_cuda=True):
    """
    Translates all CSV files in a folder, splitting long text into smaller chunks for translation.

    Parameters:
    - input_folder (str): Path to the folder containing input CSV files.
    - source_lang (str): Source language code.
    - target_lang (str): Target language code.
    - model: The translation model.
    - processor: Tokenizer/processor associated with the model.
    - max_tokens (int): Maximum number of tokens per chunk.
    - override (bool): Whether to re-translate files that already have an output.
    - use_cuda (bool): Whether to use GPU acceleration.

    Output:
    - Translates all files in the folder, saving outputs in a new folder.
    """
    encoding = 'utf-8'

    # Create the output folder
    parent_dir = os.path.abspath(os.path.join(input_folder, ".."))
    output_folder = os.path.join(parent_dir, f"translation_{source_lang}_to_{target_lang}")
    os.makedirs(output_folder, exist_ok=True)

    # Find all CSV files in the input folder
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    for input_file in tqdm(input_files, desc="Processing Files"):
        input_path = os.path.join(input_folder, input_file)
        output_path = os.path.join(output_folder, input_file)

        # Skip translation if override is False and output file already exists
        if not override and os.path.exists(output_path):
            print(f"Skipping {input_file}, output already exists.")
            continue

        # Count total rows for the progress bar
        with open(input_path, mode='r', encoding=encoding) as infile:
            total_rows = sum(1 for _ in infile) - 1  # Subtract 1 for the header

        # Translate file row by row
        with open(input_path, mode='r', encoding=encoding) as infile:
            reader = csv.DictReader(infile)

            with open(output_path, mode='w', newline='', encoding=encoding) as outfile:
                writer = csv.DictWriter(outfile, fieldnames=["Speaker", "Text"])
                writer.writeheader()

                for row in tqdm(reader, desc=f"Translating {input_file}", total=total_rows):
                    speaker = row["Speaker"]
                    text = row["Text"]

                    # Split the text into chunks
                    #chunks = split_text_into_chunks(text, max_tokens, processor, source_lang)
                    # # Translate each chunk
                    # translated_chunks = []
                    # for chunk in chunks:
                    #     translated_chunk = translation(source_lang, target_lang, chunk, model, processor, use_cuda)
                    #     translated_chunks.append(translated_chunk)
                    # Reassemble the translated chunks
                    #translated_text = ' '.join(translated_chunks)

                    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split text into sentences
                    translated_sentences = []
                    for sentence in sentences:
                        translated_sentence = translation(source_lang, target_lang, sentence, model, processor, use_cuda)
                        translated_sentences.append(translated_sentence)

                    # Reassemble the translated_sentences
                    translated_text = ' '.join(translated_sentences)

                    # Write the translated row
                    writer.writerow({"Speaker": speaker, "Text": translated_text})

        print(f"Translation completed for {input_file}. Output saved to {output_path}.")
