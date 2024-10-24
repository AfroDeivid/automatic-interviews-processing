import os
import re
import csv

def get_files(directory, extensions):
    """Get a list of files in the specified directory and its subdirectories with given extensions."""
    files = []

    for root, dirs, files_in_dir in os.walk(directory):
        for file in files_in_dir:
            if any(file.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, file))
                
    return files

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
    pattern = re.compile(r'\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\nSpeaker (\d+): (.+?)(?=\n\d+\n|\Z)', re.DOTALL)
    matches = pattern.findall(content)

    # Write to CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Experiment', 'File Name', 'Id', 'Start Time', 'End Time', 'Speaker', 'Content']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)#, quoting=csv.QUOTE_ALL)

        writer.writeheader()
        for match in matches:
            start_time, end_time, speaker, text = match
            name = os.path.splitext(os.path.basename(str_file))[0]
            writer.writerow({
                'Experiment': os.path.basename(directory),
                'File Name': name,
                'Id': extract_id(name),
                'Start Time': start_time,
                'End Time': end_time,
                'Speaker': speaker,
                'Content': text.replace('\n', '')
            })