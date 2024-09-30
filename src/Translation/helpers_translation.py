import csv
import docx # pip install python-docx
import unicodedata
import os
import torch
from tqdm import tqdm

# Function to read the Word document and extract Dialogue & Speaker
def extract_dialogue_from_docx(docx_file):
    # Load the Word document
    doc = docx.Document(docx_file)
    dialogues = []
    current_text = []
    
    # Loop through each paragraph in the document
    for paragraph in doc.paragraphs:
        text = unicodedata.normalize('NFKD', paragraph.text.strip())

        # Only look after the separator after encountering an empty line
        # Allow to handle with the combination of a line break and separator within the text at the same time !
        if not text:
            # If there's accumulated text, check for the separator and save the dialogue
            if current_text:
                combined_text = ' '.join(current_text).strip()
                if " : " in combined_text:
                    # Only look at the first separator, to avoid separator within the text
                    parts = combined_text.split(" : ", 1)
                    if len(parts) == 2:
                        speaker = parts[0].strip()
                        dialogue_text = parts[1].strip()
                        dialogues.append({"Speaker": speaker, "Text": dialogue_text})
                current_text = []
            continue

        # Accumulate text for the current speaker
        current_text.append(text)

    # Add the last accumulated dialogue
    if current_text:
        combined_text = ' '.join(current_text).strip()
        if " : " in combined_text:
            parts = combined_text.split(" : ", 1)
            if len(parts) == 2:
                speaker = parts[0].strip()
                dialogue_text = parts[1].strip()
                dialogues.append({"Speaker": speaker, "Text": dialogue_text})

    return dialogues

# Function to save the extracted dialogue to a CSV file
def save_to_csv(data, output_file):
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Speaker", "Text"])
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def translation(source_lang, target_lang, text, model, processor, cuda = False):

    if cuda:
        text_inputs = processor(text, return_tensors="pt", src_lang=source_lang).to("cuda")
    else:
        text_inputs = processor(text, return_tensors="pt", src_lang=source_lang)
        
    output_tokens = model.generate(**text_inputs, tgt_lang=target_lang)
    translated_text = processor.decode(output_tokens[0], skip_special_tokens=True)

    return translated_text

# CSV translation function with line-by-line saving
def translate_csv(input_csv, source_lang, target_lang, model, processor, cuda = False):
    encoding = 'utf-8'
    output_csv = f"{os.path.splitext(input_csv)[0]}_{target_lang}.csv"

    if cuda and torch.cuda.is_available():
        cuda = True

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
                translated_text = translation(source_lang, target_lang, text, model, processor, cuda)

                # Write the speaker and translated text to the new CSV file immediately
                writer.writerow({"Speaker": speaker, "Translated_Text": translated_text})