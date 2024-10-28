import os
import pandas as pd
import shutil
import re


def organize_csv_files_by_dir(source_dir, destination_dir):
    # Ensure destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"Created destination directory: {destination_dir}")

    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(source_dir, filename)
            
            try:
                print(file_path)
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if 'Experiment' column exists
                if 'Experiment' not in df.columns:
                    print(f"'Experiment' column not found in {filename}. Skipping this file.")
                    experiment = "Not Specified"
                else:
                    # Get unique Experiment values
                    experiment = df['Experiment'].unique()
                    if len(experiment) > 1:
                        print(f"Multiple Experiment values found in {filename}.")
                        print("Values: ", experiment, " Skipping this file.")
                        continue

                # Define subdirectory path based on Experiment value
                experiment_dir = os.path.join(destination_dir, str(experiment[0]))

                # Create subdirectory if it doesn't exist
                if not os.path.exists(experiment_dir):
                    os.makedirs(experiment_dir)
                    print(f"Created subdirectory: {experiment_dir}")

                # Define destination file path
                destination_file_path = os.path.join(experiment_dir, filename)

                # Copy the file
                shutil.copy(file_path, destination_file_path)
                print(f"Copied {filename} to {experiment_dir}")

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

# Function to remove filler words
def simpler_clean(text, filler_words = None):

    # Remove filler words from the text
    if filler_words :
        filler_words_pattern = r'\b(' + '|'.join(map(re.escape, filler_words)) + r')\b'
        text = re.sub(filler_words_pattern, '', text, flags=re.IGNORECASE)

    # If there is two words which repeat consecutively, remove one of them (ignoring case)
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    ## Visual cleaning
    # Convert lowercase "i" to uppercase "I" when it stands alone
    text = re.sub(r'\bi\b', 'I', text)
    # Remove double spaces
    text = re.sub(r'\s+', ' ', text)
    # If the firt character is a space, remove it
    if text[0] == ' ':
        text = text[1:]
    # If there is a point, put on upper case the next character
    text = re.sub(r'\.\s+(\w)', lambda x: x.group(0).upper(), text)

    return text


def clean_files(raw_folder, destination_folder, fillers_words):
    for subdir, _, files in os.walk(raw_folder):
        for file in files:
            if file.endswith(".csv"):
                # Create corresponding subdirectory in destination folder
                relative_path = os.path.relpath(subdir, raw_folder)
                destination_subdir = os.path.join(destination_folder, relative_path)
                os.makedirs(destination_subdir, exist_ok=True)
                
                # Define file paths
                raw_file_path = os.path.join(subdir, file)
                destination_file_path = os.path.join(destination_subdir, file)
                
                # Load the CSV file, process it, and save the result
                data = pd.read_csv(raw_file_path)
                if fillers_words:
                    data['Content'] = data['Content'].apply(simpler_clean, args=(fillers_words,))

                data.to_csv(destination_file_path, index=False)