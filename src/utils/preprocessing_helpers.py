import os
import pandas as pd
import shutil
import re
import csv
import numpy as np

def copy_csv_files_with_structure(source_dir, destination_dir):
    # Ensure destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"Created destination directory: {destination_dir}")

    # Walk through all files and subdirectories in the source directory
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith('.csv'):
                file_path = os.path.join(root, filename)
                
                try:
                    print(f"Processing file: {file_path}")
                    
                    # Determine the relative path from the source directory
                    relative_path = os.path.relpath(file_path, source_dir)
                    
                    # Define the destination path with the same structure
                    destination_path = os.path.join(destination_dir, relative_path)
                    
                    # Ensure the subdirectory exists in the destination
                    destination_subdir = os.path.dirname(destination_path)
                    if not os.path.exists(destination_subdir):
                        os.makedirs(destination_subdir)
                        print(f"Created subdirectory: {destination_subdir}")
                    
                    # Copy the file to the destination
                    shutil.copy(file_path, destination_path)
                    print(f"Copied {filename} to {destination_subdir}")

                except Exception as e:
                    print(f"Error processing file {filename}: {e}")

def organize_csv_files_by_experiment(source_dir, destination_dir):
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

def process_files(raw_folder, destination_folder, fillers_words= None, roles=False, text_format=False,time_stamps=False ,conditions=None, utterance=False):
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
                    data = data.dropna(subset=['Content'])
                    data['Content'] = data['Content'].apply(simpler_clean, args=(fillers_words,))
                    data = data.dropna(subset=['Content'])
                if roles:
                    df_role, _ = assign_roles(data, file_name=file)
                    data["Speaker"] = df_role["Role"]
                if text_format:
                    # if df have 'Original Content' column, use it to create a dialogue with both original and processed content
                    if 'Original Content' in data.columns:
                        convert_csv_to_dialogue_with_original(raw_file_path, destination_file_path, include_timestamps=time_stamps)
                    else:
                        convert_csv_to_dialogue_merge_speakers(raw_file_path, destination_file_path, include_timestamps=time_stamps)
                    continue
                if conditions is not None:
                    data["Condition"] = conditions[conditions["File Name"] == os.path.splitext(file)[0]]["Condition"].values[0]
                    data["Order Condition"] = conditions[conditions["File Name"] == os.path.splitext(file)[0]]["Order Condition"].values[0]
                if utterance:
                    add_utterance_index_from_csv(raw_file_path, destination_file_path)
                    continue
                
                data.to_csv(destination_file_path, index=False)

# Function to remove filler words and clean the text
def simpler_clean(text, filler_words = None):

    # Remove filler words from the text
    if filler_words :
        filler_words_pattern = r'\b(' + '|'.join(map(re.escape, filler_words)) + r')\b'
        text = re.sub(filler_words_pattern, '', text, flags=re.IGNORECASE)
    # If there is "-" alone after removing of filler words, remove it
    text = re.sub(r'\b-\b', '', text)

    # If there is two words which repeat consecutively, remove one of them (ignoring case)
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    ## Visual cleaning
    # If there only spaces, return None
    if text.isspace():
        return None
    # Convert lowercase "i" to uppercase "I" when it stands alone
    text = re.sub(r'\bi\b', 'I', text)
    # Remove double spaces
    text = re.sub(r'\s+', ' ', text)
    # If there is empty spaces at the beginning or end of the text, remove them
    text = text.strip()
    # If there is a point, put on upper case the next character
    text = re.sub(r'\.\s+(\w)', lambda x: x.group(0).upper(), text)
    # Put the first character of the text on upper case
    text = text[0].upper() + text[1:]
    # Put a point at the end of the text if there isn't a punctuation mark
    if text[-1] not in ['.', '!', '?']:
        text += '.'
    # Put a space between words and punctuation marks other than " . "
    text = re.sub(r'(\w)([!?])', r'\1 \2', text)
        
    return text

def assign_roles(data, file_name= None):
    """
    Assigns roles to speakers in the DataFrame based on participant and interviewer scores.
    
    Args:
    - df (pd.DataFrame): DataFrame containing 'Speaker' and 'Content' columns.
    
    Returns:
    - pd.DataFrame: DataFrame with an added 'Role' column.
    """
    df = data.copy()

    # Define regex patterns for participant and interviewer utterances
    participant_patterns = ["I", "me", "my", "mine", "myself"] # First person pronouns
                            
    interviewer_patterns = ['your', 'yours', 'yourself', # Second person pronouns
                            "could you", "can you", "would you", "do you", "please", "mind if I record",  # Common interviewer phrases
                            "question", "how"] # Questions
    
    participant_patterns = r'\b(' + '|'.join(map(re.escape, participant_patterns)) + r')\b'
    interviewer_patterns = r'\b(' + '|'.join(map(re.escape, interviewer_patterns)) + r')\b'
    
    # '?' does not have word boundaries like words do, so it won't be matched by patterns with \b.
    # So we handle it separately
    question_mark_weight = 1  # Default weight for '?'

    # Initialize a dictionary to store scores
    scores = {}
    
    # Calculate scores for each speaker
    for speaker in df['Speaker'].unique():
        speaker_texts = df[df['Speaker'] == speaker]['Content']
        participant_score = speaker_texts.str.count(participant_patterns, flags=re.IGNORECASE).sum()
        interviewer_score = speaker_texts.str.count(interviewer_patterns, flags=re.IGNORECASE).sum()

        # Add weighted '?' counts to interviewer score
        question_count = speaker_texts.str.count(r'\?').sum() * question_mark_weight
        interviewer_score += question_count

        scores[speaker] = {
            'participant_score': participant_score,
            'interviewer_score': interviewer_score
        }
    
    # Convert scores to DataFrame for easier manipulation
    scores_df = pd.DataFrame(scores).T.reset_index().rename(columns={'index': 'Speaker'})
    
    # Calculate score ratios
    scores_df['participant_ratio'] = scores_df['participant_score'] / (scores_df['interviewer_score'] + 1e-6)
    scores_df['interviewer_ratio'] = scores_df['interviewer_score'] / (scores_df['participant_score'] + 1e-6)
    scores_df['score_diff'] = scores_df['participant_score'] - scores_df['interviewer_score']
    
    # Initialize role assignments
    scores_df['Role'] = 'Unassigned'
    
    # Identify potential participants
    potential_participants = scores_df[scores_df['participant_ratio'] >= scores_df['interviewer_ratio']]
    
    role_dict = {}
    if not potential_participants.empty:
        # Select the speaker with the highest difference in participant score
        participant_speaker = potential_participants.sort_values(by='score_diff', ascending=False).iloc[0]['Speaker']
    else:
        # Calculate the score difference for all speakers and select the one with the highest difference
        print(f"File '{file_name}': Couldn't accurately predict the most probable participant. Define the mosts probable interviewers and select by default the participant as a fallback.")
        participant_speaker = scores_df.sort_values(by='score_diff', ascending=False).iloc[0]['Speaker']

    # Assign roles
    interviewer_count = 1
    unique_interviewers = scores_df['Speaker'].nunique() - 1  # Assuming one unique participant speaker
    for speaker in scores_df['Speaker']:
        if speaker == participant_speaker:
            role_dict[speaker] = 'Participant'
        else:
            # Only add numbering if there's more than one interviewer
            if unique_interviewers > 1:
                role_dict[speaker] = f'Interviewer {interviewer_count}'
                interviewer_count += 1
            else:
                role_dict[speaker] = 'Interviewer'
    
    # Assign roles to scores_df
    scores_df['Role'] = scores_df['Speaker'].map(role_dict)

    # Map roles back to the original DataFrame
    df['Role'] = df['Speaker'].map(role_dict)

    #print(f"File '{file_name}': {scores_df}")   
    
    return df, scores_df

def add_utterance_index_from_csv(
    input_csv: str,
    output_csv: str,
    speaker_column: str = 'Speaker',
    file_column: str = 'File Name'
):
    """
    Adds an `utterance_index` column to the input CSV file, tracking when an utterance changes.
    
    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to save the output CSV with the new column.
    - speaker_column (str): Column indicating the speaker.
    - file_column (str): Column identifying the file (unique identifier for conversations).
    
    Returns:
    - None: Writes the modified CSV to `output_csv`.
    """
    with open(input_csv, mode='r', encoding='utf-8') as infile, \
         open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['utterance_index']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        previous_speaker = None
        previous_file = None
        utterance_index = -1

        for row in reader:
            current_speaker = row.get(speaker_column, '').strip()
            current_file = row.get(file_column, '').strip()

            # If the speaker or file changes, increment the utterance index
            if current_file != previous_file or current_speaker != previous_speaker:
                utterance_index += 1
                previous_file = current_file
                previous_speaker = current_speaker

            # Add the utterance index to the row
            row['utterance_index'] = utterance_index
            writer.writerow(row)


def convert_csv_to_dialogue_merge_speakers(input_csv, output_txt, include_timestamps=False):
    """
    Converts a CSV file to a dialogue-style text file with only Speaker and Content,
    merging consecutive entries from the same speaker. Optionally includes timestamps.

    Args:
        input_csv (str): Path to the input CSV file.
        output_txt (str): Path to the output text file.
        include_timestamps (bool): Whether to include timestamps for each merged dialogue.
    """
    output_txt = os.path.splitext(output_txt)[0] + '.txt'
    with open(input_csv, mode='r', encoding='utf-8') as csvfile, \
            open(output_txt, mode='w', encoding='utf-8') as txtfile:
        
        reader = csv.DictReader(csvfile)
        
        previous_speaker = None
        dialogue_buffer = ""
        start_time = None
        end_time = None
        
        for row in reader:
            speaker = row.get('Speaker', 'Unknown').strip()
            content = row.get('Content', '').strip()
            row_start_time = row.get('Start Time', '').strip()
            row_end_time = row.get('End Time', '').strip()
            
            if not speaker or not content:
                continue  # Skip rows with missing speaker or content
            
            if speaker == previous_speaker:
                # Append to the existing dialogue buffer
                dialogue_buffer += f" {content}"
                if include_timestamps:
                    end_time = row_end_time  # Update the end time
            else:
                if previous_speaker is not None:
                    # Write the previous dialogue buffer with optional timestamps
                    if include_timestamps and start_time and end_time:
                        dialogue_line = f"{start_time} --> {end_time}\n{previous_speaker}: {dialogue_buffer}\n\n"
                    else:
                        dialogue_line = f"{previous_speaker}: {dialogue_buffer}\n\n"
                    txtfile.write(dialogue_line)
                
                # Start a new dialogue buffer
                previous_speaker = speaker
                dialogue_buffer = content
                start_time = row_start_time
                end_time = row_end_time
        
        # Write the last dialogue buffer after the loop ends
        if previous_speaker is not None and dialogue_buffer:
            if include_timestamps and start_time and end_time:
                dialogue_line = f"{start_time} --> {end_time}\n{previous_speaker}: {dialogue_buffer}"
            else:
                dialogue_line = f"{previous_speaker}: {dialogue_buffer}"
            txtfile.write(dialogue_line)


def convert_csv_to_dialogue_with_original(input_csv, output_txt, include_timestamps=False):
    """
    Converts a CSV file to a dialogue-style text file with Content and Original Content side by side,
    merging consecutive entries from the same speaker. Optionally includes timestamps for Content.

    Args:
        input_csv (str): Path to the input CSV file.
        output_txt (str): Path to the output text file.
        include_timestamps (bool): Whether to include timestamps for Content.
    """
    output_txt = os.path.splitext(output_txt)[0] + '.txt'
    with open(input_csv, mode='r', encoding='utf-8') as csvfile, \
            open(output_txt, mode='w', encoding='utf-8') as txtfile:
        
        reader = csv.DictReader(csvfile)
        
        previous_speaker = None
        dialogue_buffer_content = ""
        dialogue_buffer_original = ""
        start_time = None
        end_time = None

        for row in reader:
            speaker = row.get('Speaker', 'Unknown').strip()
            content = row.get('Content', '').strip()
            original_content = row.get('Original Content', '').strip()
            row_start_time = row.get('Start Time', '').strip()
            row_end_time = row.get('End Time', '').strip()
            
            # Skip rows with missing content
            if not speaker or not content:
                continue
            
            if speaker == previous_speaker:
                # Append to the existing dialogue buffers
                dialogue_buffer_content += f" {content}"
                dialogue_buffer_original += f" {original_content}"
                if include_timestamps:
                    end_time = row_end_time  # Update the end time
            else:
                if previous_speaker is not None:
                    # Write the previous dialogue buffers with timestamps for Content
                    if include_timestamps and start_time and end_time:
                        dialogue_line = (
                            f"{start_time} --> {end_time}\n"
                            f"{previous_speaker}: {dialogue_buffer_content}\n\n"
                            f"Original Content: {dialogue_buffer_original}\n\n"
                        )
                    else:
                        dialogue_line = (
                            f"Speaker {previous_speaker}: {dialogue_buffer_content}\n"
                            f"Original Content: {dialogue_buffer_original}\n\n"
                        )
                    txtfile.write(dialogue_line)
                
                # Start new dialogue buffers
                previous_speaker = speaker
                dialogue_buffer_content = content
                dialogue_buffer_original = original_content
                start_time = row_start_time
                end_time = row_end_time

        # Write the last dialogue buffer after the loop ends
        if previous_speaker is not None and dialogue_buffer_content:
            if include_timestamps and start_time and end_time:
                dialogue_line = (
                    f"{start_time} --> {end_time}\n"
                    f"Speaker {previous_speaker}:\n"
                    f"Content: {dialogue_buffer_content}\n"
                    f"Original Content: {dialogue_buffer_original}\n"
                )
            else:
                dialogue_line = (
                    f"Speaker {previous_speaker}:\n"
                    f"Content: {dialogue_buffer_content}\n"
                    f"Original Content: {dialogue_buffer_original}\n"
                )
            txtfile.write(dialogue_line)



def merge_csv_in_subdirectories(source_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Iterate over each subdirectory in the source directory
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        
        # Ensure it's a directory
        if os.path.isdir(subdir_path):
            csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
            
            # Check if there are exactly two CSV files to merge
            if len(csv_files) == 2:
                csv_paths = [os.path.join(subdir_path, f) for f in csv_files]
                
                # Read and concatenate the CSV files
                try:
                    df_list = [pd.read_csv(file) for file in csv_paths]
                    merged_df = pd.concat(df_list, ignore_index=True)
                    
                    # Save the merged file in the output directory
                    merged_file_name = f"{subdir}_merged.csv"
                    merged_file_path = os.path.join(output_dir, merged_file_name)
                    merged_df.to_csv(merged_file_path, index=False)
                    print(f"Merged files from {subdir_path} into {merged_file_path}")
                
                except Exception as e:
                    print(f"Error merging files in {subdir_path}: {e}")
            else:
                print(f"Skipping {subdir_path}: Expected 2 CSV files, found {len(csv_files)}")

def preprocessing_csv(str_file):
    """Create a processed version of the CSV directly from the .str file path."""
    # List of common filler words to remove
    fillers_words = ["uh", "huh", "um", "hmm", "Mm"]
    
    # Change the extension from .str to .csv
    csv_path = str_file.replace('.str', '.csv')
    
    # Load the CSV file generated from the .str file
    df = pd.read_csv(csv_path)
    
    # Apply preprocessing functions
    ## Remove fillers words & basic cleaning
    df = df.dropna(subset=['Content'])
    df['Content'] = df['Content'].apply(simpler_clean, args=(fillers_words,))
    df = df.dropna(subset=['Content'])

    ## Assign roles
    df_role, _ = assign_roles(df, file_name=csv_path)
    df["Speaker"] = df_role["Role"]

    # Define the output directory within `results/processed`, mirroring the original structure
    relative_path = os.path.relpath(str_file, "results")  # Calculate the relative path from `results`
    preproced_dir = os.path.join("results", "processed", os.path.dirname(relative_path))
    os.makedirs(preproced_dir, exist_ok=True)  # Create necessary directories

    preproced_csv_path = os.path.join(preproced_dir, os.path.basename(csv_path))
    df.to_csv(preproced_csv_path, index=False)
    print(f"Saved preprocessed file to {preproced_csv_path}")

def match_transcripts(reference_df, target_df):
    """
    Align transcripts by largest overlap or closest proximity.
    Adds speaker annotations when merging target content with different speakers.
    Returns a list of of the target content strings, aligned with the index of reference_df.
    """
    # Copy dataframes to avoid modifying originals
    reference_df = reference_df.copy()
    target_df = target_df.copy()

    def time_to_ms(timestamp):
        h, m, s_ms = timestamp.strip().split(":")
        s, ms = s_ms.split(",")
        total_ms = (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)
        return total_ms

    # Convert timestamps to milliseconds
    reference_df['Start_ms'] = reference_df['Start Time'].apply(time_to_ms)
    reference_df['End_ms'] = reference_df['End Time'].apply(time_to_ms)
    target_df['Start_ms'] = target_df['Start Time'].apply(time_to_ms)
    target_df['End_ms'] = target_df['End Time'].apply(time_to_ms)

    
    ref_intervals = pd.IntervalIndex.from_arrays(reference_df['Start_ms'], reference_df['End_ms'], closed='both')    

    ref_df_intervals = reference_df[['Start_ms', 'End_ms']].copy()
    ref_df_intervals['ref_index'] = ref_df_intervals.index
    tar_df_intervals = target_df[['Start_ms', 'End_ms']].copy()
    tar_df_intervals['tar_index'] = tar_df_intervals.index

    # Build a dataframe of overlaps
    overlaps = []
    for idx, row in tar_df_intervals.iterrows():
        # Find overlapping reference intervals
        overlapping_refs = ref_intervals.overlaps(pd.Interval(row['Start_ms'], row['End_ms'], closed='both'))
        if overlapping_refs.any():
            # Compute overlap durations
            overlap_durations = (np.minimum(reference_df.loc[overlapping_refs, 'End_ms'], row['End_ms']) - 
                                 np.maximum(reference_df.loc[overlapping_refs, 'Start_ms'], row['Start_ms']))
            # Select the reference with the maximum overlap
            max_overlap_idx = overlap_durations.idxmax()
            overlaps.append((row['tar_index'], max_overlap_idx))
        else:
            # No overlap; will handle later
            overlaps.append((row['tar_index'], None))

    # Convert overlaps to DataFrame
    overlaps_df = pd.DataFrame(overlaps, columns=['tar_index', 'ref_index'])

    # Handle non-overlapping target intervals by assigning to the nearest reference interval
    no_overlap_mask = overlaps_df['ref_index'].isnull()
    if no_overlap_mask.any():
        # For target intervals with no overlaps
        no_overlap_tar_indices = overlaps_df.loc[no_overlap_mask, 'tar_index']
        tar_no_overlap = target_df.loc[no_overlap_tar_indices]
        tar_no_overlap['mid_point'] = (tar_no_overlap['Start_ms'] + tar_no_overlap['End_ms']) / 2

        # Compute distances to reference intervals' midpoints
        ref_mid_points = (reference_df['Start_ms'] + reference_df['End_ms']) / 2
        distances = ref_mid_points.values.reshape(-1, 1) - tar_no_overlap['mid_point'].values.reshape(1, -1)
        distances = np.abs(distances)

        # Find the closest reference interval for each target interval
        closest_refs = distances.argmin(axis=0)
        overlaps_df.loc[no_overlap_mask, 'ref_index'] = closest_refs

    # Group target rows by their matched reference rows
    target_df['ref_index'] = overlaps_df['ref_index'].values
    grouped = target_df.groupby('ref_index')

    # Prepare the target content list aligned with reference_df
    target_content_list = [None] * len(reference_df)

    for ref_idx, group in grouped:
        # Get the sorted target contents
        group_sorted = group.sort_values('Start_ms')
        speakers = group_sorted['Speaker'].unique()
        if len(speakers) > 1:
            # Annotate speakers
            speaker_contents = group_sorted.groupby('Speaker')['Content'].apply(' '.join)
            target_content_str = ' '.join(f'Speaker {sp}: {cnt}' for sp, cnt in speaker_contents.items())
        else:
            # Single speaker
            target_content_str = ' '.join(group_sorted['Content'])
        target_content_list[int(ref_idx)] = target_content_str

    return target_content_list

def match_transcripts_folder(reference_dir, target_dir, output_dir):
    """
    Walks through reference_dir and target_dir, matches CSV files with the same name,
    applies match_transcripts_refined function to each pair, and saves the output in output_dir.
    
    Parameters:
        reference_dir (str): The directory containing the reference English CSV files.
        target_dir (str): The directory containing the target French CSV files.
        output_dir (str): The directory where the matched CSV files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all CSV files from the reference directory
    for root, dirs, files in os.walk(reference_dir):
        for file in files:
            if file.endswith('.csv'):
                # Construct full path for the reference file
                reference_file_path = os.path.join(root, file)
                
                # Compute the relative path to maintain directory structure
                relative_path = os.path.relpath(reference_file_path, reference_dir)
                
                # Construct the corresponding target file path
                target_file_path = os.path.join(target_dir, relative_path)
                
                # Check if the corresponding target file exists
                if os.path.exists(target_file_path):
                    # Load the data
                    df_ref = pd.read_csv(reference_file_path)
                    df_target = pd.read_csv(target_file_path)
                    
                    # Perform the matching
                    df_ref["Original Content"] = match_transcripts(df_ref, df_target)
                    
                    # Construct the output file path
                    output_file_path = os.path.join(output_dir, relative_path)
                    
                    # Ensure the output subdirectory exists
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    
                    # Save the updated DataFrame
                    df_ref.to_csv(output_file_path, index=False, encoding='utf-8-sig')
                else:
                    print(f"Matching file not found for {reference_file_path}")
