import csv


def split_text_into_chunks(input_csv, max_tokens, processor, src_lang):
    chunks = []
    current_chunk = ''
    add_speaker_label = True  # Flag to control when to add the speaker label

    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            speaker = row['Speaker']
            text = row['Text']
            
            # Only add the speaker label if it's a new chunk or a new row
            if add_speaker_label:
                text_with_label = f"[{speaker}]: {text}"
            else:
                text_with_label = text  # Skip adding speaker label if continuing the same row

            # Split text into sentences for controlled chunking
            sentences = re.split(r'(?<=[.!?])\s+', text_with_label)
            for sentence in sentences:
                # Estimate tokens if we add the new sentence
                combined_text = (current_chunk + " " + sentence).strip()
                tokens = processor(combined_text, return_tensors="pt", src_lang=src_lang)
                num_tokens = len(tokens['input_ids'][0])

                if num_tokens <= max_tokens:
                    current_chunk = combined_text  # Continue building the chunk
                    add_speaker_label = False  # Do not add speaker label again for this row
                else:
                    # Add the current chunk to the list if it reaches the limit
                    chunks.append(current_chunk.strip())
                    # Start a new chunk with the current sentence (without speaker label)
                    current_chunk = sentence.strip()
                    add_speaker_label = True  # Reset to add speaker label for the next row

            # Reset label addition for the next row
            add_speaker_label = True

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks

def parse_and_write_translated_text(translated_chunks, output_csv):
    dialogue = []
    pattern = r'\[(.*?)\]:\s*(.*?)(?=\[|$)' # Regex pattern to match speaker and text
    for translated_chunk in translated_chunks:
        matches = re.findall(pattern, translated_chunk, re.DOTALL)
        for speaker, text in matches:
            dialogue.append({'Speaker': speaker.strip(), 'Translated_Text': text.strip()})
    # Write to CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
        fieldnames = ['Speaker', 'Translated_Text']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for turn in dialogue:
            writer.writerow(turn)

def translate_with_chunks(input_csv, source_lang, target_lang, model, processor, use_cuda=True, buffer_size=200):

    # Split the text into chunks for context & performance reasons.
    max_length = 200 # Minus a buffer for special tokens or potential expansion during translation.
    chunks = split_text_into_chunks(input_csv, max_length, processor, source_lang)
    print(f"Number of chunks: {len(chunks)}")
    for text in chunks:
        tokens = processor(text, return_tensors="pt", src_lang=source_lang)
        print(f"Number of tokens: {len(tokens['input_ids'][0])}")

    # Translate the chunks
    translated_chunks = []
    for chunk in chunks:
        translated_text = translation(source_lang, target_lang, chunk, model, processor, use_cuda)
        translated_chunks.append(translated_text)

    # Parse and write the translated text
    output_csv = f"{os.path.splitext(input_csv)[0]}_{source_lang}_to_{target_lang}.csv"
    parse_and_write_translated_text(translated_chunks, output_csv)
