import os
import re
import csv
from pydub.utils import mediainfo
import pandas as pd

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

def convert_str_to_csv(str_file, directory='Not Specified'):
    """Convert a single .str file to a CSV file."""
    csv_file = os.path.splitext(str_file)[0] + '.csv'

    with open(str_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expression to match each entry in the .str file
    pattern = re.compile(r'\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\nSpeaker (\d+): (.+?)(?=\n\d+\n|\Z)', re.DOTALL)
    matches = pattern.findall(content)

    # Write to CSV
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as csvfile: # encoding='utf-8-sig' to add BOM signature and being recognized as UTF-8 format by Excel
                                                                       # One possible solution to handle special characters in Excel.
        fieldnames = ['Experiment', 'File Name', 'Id','Content Type' ,'Start Time', 'End Time', 'Speaker', 'Content']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)#, quoting=csv.QUOTE_ALL)

        writer.writeheader()
        for match in matches:
            start_time, end_time, speaker, text = match
            name = os.path.splitext(os.path.basename(str_file))[0]
            writer.writerow({
                'Experiment': os.path.basename(directory),
                'File Name': name,
                'Id': extract_id(name),
                'Content Type': "Audio",
                'Start Time': start_time,
                'End Time': end_time,
                'Speaker': speaker,
                'Content': text.replace('\n', '')
            })

def analyze_audio_files(directories, extensions):
    """Analyze audio files and collect their properties."""
    data = []
    for directory in directories:
        audio_files = get_files(directory, extensions)
        for audio_file in audio_files:
            info = mediainfo(audio_file)

            # Duration
            duration = round(float(info['duration']), 2) if 'duration' in info else 0
            duration_min, duration_sec = divmod(int(duration), 60)
            duration_hr, duration_min = divmod(duration_min, 60)
            duration_string = f"{duration_hr:02d}:{duration_min:02d}:{duration_sec:02d}" # Format as hh:mm:ss

            # File name
            name, ext = os.path.splitext(os.path.basename(audio_file))
            
            data.append({
                'File_name': name,
                "Format": ext,
                "ID": extract_id(name),
                'Duration': duration_string,
                'Duration_sec': duration,
                'Experiment': os.path.basename(directory),

            })
    
    return pd.DataFrame(data)