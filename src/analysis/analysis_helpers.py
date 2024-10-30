import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary libraries for word frequency analysis
from collections import Counter
#nltk.download('stopwords') # Install NLTK stopwords if not already installed
from nltk.corpus import stopwords
import string

def load_and_combine_csv(directory_path, pattern='*.csv'):
    """
    Load and combine CSV files from a specified directory.

    Parameters:
    - directory_path (str): Path to the directory containing CSV files.
    - pattern (str): Glob pattern to match CSV files. Default is '*.csv'.

    Returns:
    - pd.DataFrame: Combined DataFrame containing data from all CSV files.
    """
    # Construct the full file path pattern
    csv_files_path = os.path.join(directory_path, pattern)
    # Retrieve all matching CSV file paths
    csv_files = glob.glob(csv_files_path)

    if not csv_files:
        print(f"No CSV files found in {directory_path} with pattern '{pattern}'.")
        return pd.DataFrame()  # Return empty DataFrame if no files found

    print(f"Found {len(csv_files)} CSV files.")

    # Load each CSV file into a DataFrame and store in a list
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
            #print(f"Loaded '{os.path.basename(file)}' successfully.")
        except Exception as e:
            print(f"Error loading '{os.path.basename(file)}': {e}")

    # Combine all DataFrames into one
    combined_df = pd.concat(df_list, ignore_index=True)

    return combined_df

def standardize_data(df):
    """
    Standardize Speaker labels & normalize the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    - pd.DataFrame: Cleaned and preprocessed DataFrame.
    """

    # Standardize Speaker labels (e.g., merge all Interviewers into one category)
    speaker_replacements = {
        'Interviewer 1': 'Interviewer',
        'Interviewer 2': 'Interviewer',
        'Interviewer 3': 'Interviewer',
        # Add more replacements if necessary
    }
    df['Speaker'] = df['Speaker'].replace(speaker_replacements)
    print("Standardized speaker labels.")

    # Normalize text: convert to lowercase and strip whitespace
    df['Content'] = df['Content'].str.lower().str.strip()
    print("Normalized text in 'Content' column.")

    return df

def calculate_word_char_counts(df):
    """
    Calculate word and character counts for each 'Content' entry.

    Parameters:
    - df (pd.DataFrame): The DataFrame with 'Content' column.

    Returns:
    - pd.DataFrame: DataFrame with added 'Word_Count' and 'Character_Count' columns.
    """

    df['Word_Count'] = df['Content'].apply(lambda x: len(x.split()))
    df['Character_Count'] = df['Content'].apply(len)
    print("Calculated 'Word_Count' and 'Character_Count' for each response.")

    return df

def aggregate_counts(df, groupby_columns):
    """
    Aggregate word and character counts based on specified grouping columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - groupby_columns (list of str): List of column names to group by.

    Returns:
    - pd.DataFrame: Aggregated DataFrame with sum and mean of word and character counts.
    """
    
    # Perform the aggregation
    aggregated = df.groupby(groupby_columns).agg(
        Word_Count=('Word_Count', 'sum'),
        Average_Words=('Word_Count', 'mean'),
        Character_Count=('Character_Count', 'sum'),
        Average_Characters=('Character_Count', 'mean')
    ).reset_index()
    
    return aggregated

def box_plot(df,x_column,y_column, hue_column=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x_column, y=y_column, data=df, hue=hue_column)
    plt.title(f'{x_column} Distribution by {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

def word_frequency_plot(df, stop_words=False, top_n=20):

    if not stop_words:
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = set()
    punctuation = set(string.punctuation)

    df['Tokens'] = df['Content'].apply(
        lambda x: [
            word.strip(string.punctuation) 
            for word in x.split() 
            if word not in stop_words and word not in punctuation
        ]
    )

    all_tokens = [token for tokens in df['Tokens'] for token in tokens]
    word_freq = Counter(all_tokens)
    top_n_words = word_freq.most_common(top_n)
    top_n_df = pd.DataFrame(top_n_words, columns=['Word', 'Frequency'])

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Frequency', y='Word', data=top_n_df, palette='viridis', hue='Word', legend=False)
    plt.title('Top 20 Most Frequent Words')
    plt.xlabel('Frequency')
    plt.ylabel('Word')

    plt.tight_layout()
    plt.show()