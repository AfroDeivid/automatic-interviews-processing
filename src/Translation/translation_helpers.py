import csv
import docx # pip install python-docx
import unicodedata
import os
import torch
from tqdm import tqdm

# Function to read the Word document and extract dialogue
def extract_dialogue_from_docx(docx_file):
    separator = ":"
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
                    user_input = input(f"Do you recognize this speaker '{speaker}'? (yes/no): ").strip().lower()
                    if user_input == "yes":
                        known_speakers.add(speaker)
                        print("Thanks! We set this speaker as valid name.")
                    elif user_input == "no" and last_speaker:
                        # Reassign the text to the previous speaker
                        dialogues[-1]["Text"] += " " + text
                        print("Thanks! Therefore we assing this text to the previous speaker.")
                        continue

                dialogues.append({"Speaker": speaker, "Text": dialogue_text})
                last_speaker = speaker
        else:
            # If no separator, assume it belongs to the last speaker
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


def translation(source_lang, target_lang, text, model, processor, use_cuda = True):

    if use_cuda:
        text_inputs = processor(text, return_tensors="pt", src_lang=source_lang).to("cuda")
    else:
        text_inputs = processor(text, return_tensors="pt", src_lang=source_lang)
        
    output_tokens = model.generate(**text_inputs, tgt_lang=target_lang)
    translated_text = processor.decode(output_tokens[0], skip_special_tokens=True)

    return translated_text

# CSV translation function with line-by-line saving
def translate_csv(input_csv, source_lang, target_lang, model, processor, use_cuda = True):
    encoding = 'utf-8'
    output_csv = f"{os.path.splitext(input_csv)[0]}_{target_lang}.csv"

    use_cuda = use_cuda and torch.cuda.is_available()

    # Count the number of rows already processed in the output file
    processed_rows = 0
    try:
        with open(output_csv, mode='r', encoding=encoding) as outfile:
            reader = csv.reader(outfile)
            processed_rows = sum(1 for row in reader) - 1  # Subtract 1 for the header row
    except FileNotFoundError:
        pass

    # Open the input file for reading
    with open(input_csv, mode='r', encoding=encoding) as infile:
        reader = csv.DictReader(infile)

        # Open the output file in append mode so that progress is saved after each row
        with open(output_csv, mode='a', newline='', encoding=encoding) as outfile:
            writer = csv.DictWriter(outfile, fieldnames=["Speaker", "Translated_Text"])
    
            # Check if the file is empty to avoid writing headers multiple times
            if infile.tell() == 0: # File is empty
                writer.writeheader() # Header == Columns names
  
            # Skip the already processed rows in the input file
            for _ in range(processed_rows):
                next(reader)
            
            # Use tqdm to display a progress bar
            rows = list(reader)
            for row in tqdm(rows, desc="Translating", unit="row"):
                speaker = row["Speaker"]
                text = row["Text"]

                # Translate the text
                translated_text = translation(source_lang, target_lang, text, model, processor, use_cuda)

                # Write the speaker and translated text to the new CSV file immediately
                writer.writerow({"Speaker": speaker, "Translated_Text": translated_text})