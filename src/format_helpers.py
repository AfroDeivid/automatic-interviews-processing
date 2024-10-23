import os
import re
import csv


def get_audio_files(directory, extensions):
    """Get a list of audio files in the specified directory and its subdirectories with given extensions."""
    audio_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    return audio_files


def extract_id(name):
    """Extract numeric ID from the file name."""
    match = re.search(r'(\d+)', name)
    if match:
        participant_id = int(match.group(0))
    else:
        participant_id = None

    return participant_id


def convert_str_to_csv(str_file, directory):
    """Convert a single .str file to a CSV file."""
    csv_file = os.path.splitext(str_file)[0] + '.csv'

    with open(str_file, 'r') as file:
        content = file.read()

    # Regular expression to match each entry in the .str file
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2}),\d{3} --> (\d{2}:\d{2}:\d{2}),\d{3}\nSpeaker (\d+): (.+?)(?=\n\d+\n|\Z)', re.DOTALL)
    matches = pattern.findall(content)

    # Write to CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Index', 'Experiment', 'File Name', 'Id', 'Start Time', 'End Time', 'Speaker', 'Content']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)#, quoting=csv.QUOTE_ALL)

        writer.writeheader()
        for match in matches:
            index, start_time, end_time, speaker, text = match
            name, _ = os.path.splitext(os.path.basename(str_file)) # os.path.basename(str_file)
            writer.writerow({
                'Index': index,
                'Experiment': os.path.basename(directory),
                'File Name': name,
                'Id': extract_id(name),
                'Start Time': start_time,
                'End Time': end_time,
                'Speaker': speaker,
                'Content': text.replace('\n', '')
            })