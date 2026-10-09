import os
import re
import csv
from pydub.utils import mediainfo
import pandas as pd
from datetime import timedelta

def get_files(directory, extensions):
    """
    Get a list of files in the specified directory and its subdirectories with given extensions.
    Args:
        directory (str): The path to the directory to search for files.
        extensions (list of str): A list of file extensions to filter the files.
    Returns:
        list of str: A list of file paths that match the given extensions.
    """
    files = []

    for root, dirs, files_in_dir in os.walk(directory):
        for file in files_in_dir:
            if any(file.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, file))
                
    return files

def extract_id(name):
    """
    Extract numeric ID from the file name.
    Args:
        name (str): The file name from which to extract the numeric ID.
    Returns:
        int or None: The extracted numeric ID if found, otherwise None.
    """
    match = re.search(r'(\d+)', name)
    if match:
        participant_id = int(match.group(0))
    else:
        participant_id = None

    return participant_id

def convert_str_to_csv(str_file, experiment='Not Specified'):
    """
    Convert a single .str file to a CSV file.
    Args:
        str_file (str): The path to the .str file to be converted.
        experiment (str, optional): The name of the experiment. Defaults to 'Not Specified'.
    Returns:
        None
    The function reads the content of the .str file, extracts relevant information using a regular expression,
    and writes the extracted data to a CSV file. The CSV file will have the following columns:
    'Experiment', 'File Name', 'Id', 'Content Type', 'Start Time', 'End Time', 'Speaker', and 'Content'.
    """
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
    """
    Analyze audio files and collect their properties.
    Parameters:
    directories (list of str): List of directories to search for audio files.
    extensions (list of str): List of file extensions to filter audio files.
    Returns:
    pd.DataFrame: A DataFrame containing the properties of the audio files, including:
        - File Name (str): The name of the audio file without extension.
        - Format (str): The file extension of the audio file.
        - Id (str): An extracted identifier from the file name.
        - Duration (str): The duration of the audio file formatted as HH:MM:SS.
        - Duration_timedelta (timedelta): The duration of the audio file as a timedelta object.
        - Duration_sec (float): The duration of the audio file in seconds.
        - Experiment (str): The name of the directory containing the audio file.
    """
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