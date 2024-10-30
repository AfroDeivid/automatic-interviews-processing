import os
import pandas as pd
import shutil
import re
import csv


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

def clean_files(raw_folder, destination_folder, fillers_words= None, roles=False, text_format=False):
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
                    # Remove rows with None values
                    data = data.dropna(subset=['Content'])
                if roles:
                    df_role, _ = assign_roles(data, file_name=file)
                    data["Speaker"] = df_role["Role"]
                if text_format:
                    convert_csv_to_dialogue_merge_speakers(raw_file_path, destination_file_path)
                else:
                    data.to_csv(destination_file_path, index=False)

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
    for speaker in scores_df['Speaker']:
        if speaker == participant_speaker:
            role_dict[speaker] = 'Participant'
        else:
            role_dict[speaker] = f'Interviewer {interviewer_count}'
            interviewer_count += 1
    
    # Assign roles to scores_df
    scores_df['Role'] = scores_df['Speaker'].map(role_dict)

    # Map roles back to the original DataFrame
    df['Role'] = df['Speaker'].map(role_dict)

    #print(f"File '{file_name}': {scores_df}")   
    
    return df, scores_df

def convert_csv_to_dialogue_merge_speakers(input_csv, output_txt):
    """
    Converts a CSV file to a dialogue-style text file with only Speaker and Content,
    merging consecutive entries from the same speaker.

    Args:
        input_csv (str): Path to the input CSV file.
        output_txt (str): Path to the output text file.
    """
    output_txt = os.path.splitext(output_txt)[0] + '.txt'
    with open(input_csv, mode='r', encoding='utf-8') as csvfile, \
            open(output_txt, mode='w', encoding='utf-8') as txtfile:
        
        reader = csv.DictReader(csvfile)
        
        previous_speaker = None
        dialogue_buffer = ""
        
        for row in reader:
            speaker = row.get('Speaker', 'Unknown').strip()
            content = row.get('Content', '').strip()
            
            if not speaker or not content:
                continue  # Skip rows with missing speaker or content
            
            if speaker == previous_speaker:
                # Append to the existing dialogue buffer
                dialogue_buffer += f" {content}"
            else:
                if previous_speaker is not None:
                    dialogue_line = f"{previous_speaker}: {dialogue_buffer}\n\n"
                    txtfile.write(dialogue_line)
                
                # Start a new dialogue buffer
                previous_speaker = speaker
                dialogue_buffer = content
        
        # Write the last dialogue buffer after the loop ends
        if previous_speaker is not None and dialogue_buffer:
            dialogue_line = f"{previous_speaker}: {dialogue_buffer}"
            txtfile.write(dialogue_line)