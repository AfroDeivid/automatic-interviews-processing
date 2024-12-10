import os
import re
import csv
from pydub.utils import mediainfo
import pandas as pd
from datetime import timedelta

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

def convert_str_to_csv(str_file, experiment='Not Specified'):
    """Convert a single .str file to a CSV file."""
    csv_file = os.path.splitext(str_file)[0] + '.csv'

    with open(str_file, 'r', encoding='utf-8-sig') as file:
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
                'Experiment': experiment,
                'File Name': name,
                'Id': extract_id(name),
                'Content Type': "Audio",
                'Start Time': start_time,
                'End Time': end_time,
                'Speaker': speaker,
                'Content': text.replace('\n', '')
            })

def format_timedelta(td):
    """Format a timedelta object as HH:MM:SS."""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def analyze_audio_files(directories, extensions):
    """Analyze audio files and collect their properties."""
    data = []
    for directory in directories:
        audio_files = get_files(directory, extensions)
        for audio_file in audio_files:
            info = mediainfo(audio_file)

            # Duration
            duration_seconds = round(float(info['duration']), 2) if 'duration' in info else 0
            duration_timedelta = timedelta(seconds=duration_seconds)  # Convert to timedelta
            duration_string = format_timedelta(duration_timedelta)  # Format as HH:MM:SS

            # File name
            name, ext = os.path.splitext(os.path.basename(audio_file))
            
            data.append({
                "File Name": name,
                "Format": ext,
                "Id": extract_id(name),
                'Duration': duration_string,
                'Duration_timedelta': duration_timedelta,  # Keep timedelta for calculations
                'Duration_sec': duration_seconds,
                'Experiment': os.path.basename(directory),
            })
    
    return pd.DataFrame(data)