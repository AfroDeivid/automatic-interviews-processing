import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from typing import Optional, List, Set

import spacy
# Load spaCy model
#!python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


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

def standardize_speaker_labels(df):
    """
    Standardize Speaker labels.
    """

    # Standardize Speaker labels (e.g., merge all Interviewers into one category)
    speaker_replacements = {
        'Interviewer 1': 'Interviewer',
        'Interviewer 2': 'Interviewer',
        'Interviewer 3': 'Interviewer',
        # Add more replacements if necessary
    }
    df["Speaker_original"] = df["Speaker"]
    df['Speaker'] = df['Speaker'].replace(speaker_replacements)

    return df

def calculate_word_counts(df):
    """
    Calculate word and character counts for each 'Content' entry.

    Parameters:
    - df (pd.DataFrame): The DataFrame with 'Content' column.

    Returns:
    - pd.DataFrame: DataFrame with added 'Word_Count' and 'Character_Count' columns.
    """

    df['Word Count'] = df['Content'].apply(lambda x: len(x.split()))
    print("Calculated 'Word Count' for each 'Content' entry.")

    return df

def aggregate_counts(df, groupby_columns):
    """
    Aggregate word counts based on specified grouping columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - groupby_columns (list of str): List of column names to group by.

    Returns:
    - pd.DataFrame: Aggregated DataFrame with the sum of word counts.
    """
    
    # Perform the aggregation
    aggregated = df.groupby(groupby_columns).agg({'Word Count': 'sum'}).reset_index()

    return aggregated

def stripplot(df,x_column,y_column, hue_column=None):

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x_column, y=y_column, data=df, hue=hue_column)
    sns.stripplot(x=x_column, y=y_column, data=df, hue=hue_column, linewidth=1, edgecolor="k",
                  dodge=True, jitter=True, legend=False)
    plt.title(f'{x_column} Distribution by {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

def stripplot_with_counts(df, x_column, y_column, hue_column=None, id_column=None, legend_labels= None, file_name=None):

    plt.figure(figsize=(10, 6))

    # Draw boxplot
    ax = sns.boxplot(x=x_column, y=y_column, data=df, hue=hue_column)

    # Draw stripplot
    strip = sns.stripplot(x=x_column, y=y_column, data=df, hue=hue_column, linewidth=1, edgecolor="k",
                          dodge=True, jitter=True, legend=False)

    # Extend y-limits to include space for counts
    y_min, y_max = plt.ylim()
    y_range = y_max - y_min
    plt.ylim(y_min - (0.02 * y_range), y_max)

    # Update the legend with custom labels if provided
    if legend_labels:
        handles, labels = ax.get_legend_handles_labels()
        # Ensure the number of labels matches
        if len(legend_labels) != len(handles):
            raise ValueError("Number of legend labels does not match the number of hues.")
        ax.legend(handles[:len(legend_labels)], legend_labels, title=hue_column)
    else:
        ax.legend(title=hue_column)

    # Calculate and annotate counts
    if hue_column:
        # Group by both x and hue columns
        group_counts = df.groupby([x_column, hue_column]).size().unstack().fillna(0)
        for i, level in enumerate(group_counts.index):
            for j, hue_level in enumerate(group_counts.columns):
                count = int(group_counts.loc[level, hue_level])
                x_pos = i + (j - 0.5 * (len(group_counts.columns) - 1)) * 0.8 / len(group_counts.columns)
                plt.text(x_pos, y_min + 25 , f'n={count}',
                         ha='center', va='top', fontsize=10, fontweight='bold')
    else:
        # Group by x_column only
        group_counts = df[x_column].value_counts().sort_index()
        for i, level in enumerate(group_counts.index):
            count = group_counts[level]
            plt.text(i, y_min + 25, f'n={count}', ha='center', va='top', fontsize=13)

    # Connect points belonging to the same ID
    if id_column:
        x_levels = df[x_column].unique()
        x_dict = {level: i for i, level in enumerate(x_levels)}
        if hue_column:
            hue_levels = df[hue_column].unique()
            hue_dict = {level: i for i, level in enumerate(hue_levels)}
            n_hues = len(hue_levels)
            width = 0.8  # Total width allocated to hues
        else:
            hue_levels = [None]
            hue_dict = {None: 0}
            n_hues = 1
            width = 0

        positions = {}
        for idx, row in df.iterrows():
            x_level = row[x_column]
            x_index = x_dict[x_level]
            hue_level = row[hue_column] if hue_column else None
            hue_index = hue_dict[hue_level]
            # Compute the adjusted x position
            x_pos = x_index + (hue_index - (n_hues - 1) / 2) * width / n_hues
            y_val = row[y_column]
            key = row[id_column]
            positions.setdefault(key, []).append((x_pos, y_val))

        # Plot lines connecting the positions for each ID
        for key, pos_list in positions.items():
            # Sort the positions by x positions to ensure correct line plotting
            pos_list.sort(key=lambda x: x[0])
            x_vals, y_vals = zip(*pos_list)
            plt.plot(x_vals, y_vals, color='gray', alpha=0.5)

    # Set titles and labels
    plt.title(f'{y_column} Distribution by {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)

    plt.tight_layout()

    if file_name:
        plt.savefig(file_name, dpi=600)

    plt.show()

### Text Analysis Functions

def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    extra_stopwords: Optional[Set[str]] = None,
    retain_stopwords: Optional[Set[str]] = None
) -> str:
    """
    Preprocess text using spaCy, including tokenization, lemmatization,
    stopword removal, and lowercasing.

    Args:
        text (str): The input text to preprocess.
        remove_stopwords (bool): Whether to remove stopwords, punctuation, and whitespace. Default is True.
        extra_stopwords (Optional[Set[str]]): Additional custom stopwords to remove. Default is None.
        retain_stopwords (Optional[Set[str]]): Specific stopwords to retain even if they are stopwords. Default is None.

    Returns:
        str: The preprocessed text as a single string.
    """
    if not text:
        return ""

    # Process the text using spaCy
    doc = nlp(text)

    # Initialize an empty list to hold processed tokens
    tokens = []
    for token in doc:
        # Lemmatize and lowercase
        lemma = token.lemma_.lower()

        # Apply stopword filter
        if remove_stopwords and token.is_stop:
            # Retain specific stopwords if listed
            if retain_stopwords and lemma in retain_stopwords:
                tokens.append(lemma)
            continue

        # Remove tokens in extra stopwords
        if extra_stopwords and lemma in extra_stopwords:
            continue

        # Filter punctuation and spaces
        if token.is_punct or token.is_space:
            continue

        # Add valid tokens to the list
        tokens.append(lemma)

    return " ".join(tokens)


def count_word_frequencies(
    df: pd.DataFrame,
    tokenized_column: str = 'preprocessed_content',
    groupby_columns: Optional[List[str]] = None,
    normalize: bool = False
) -> pd.DataFrame:
    """
    Count word frequencies for the specified groupings.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - tokenized_column (str): Column containing pre-tokenized text content (list of words).
    - groupby_columns (list of str, optional): Columns to group by (e.g., Participant, File).
    - normalize (bool): Whether to normalize word counts by total word count in each group.

    Returns:
    - pd.DataFrame: DataFrame with word frequencies and optional normalization.
    """
    results = []

    # Group by specified columns
    grouped = df.groupby(groupby_columns) if groupby_columns else [(None, df)]

    for group_values, group_df in grouped:
        tokens = []
        for entry in group_df[tokenized_column].dropna():
            tokens.extend(entry.split())

        word_counts = Counter(tokens)
        total_words = sum(word_counts.values())

        # Normalize if requested
        if normalize and total_words > 0:
            word_counts = {word: count / total_words for word, count in word_counts.items()}

        # Prepare results
        group_dict = dict(zip(groupby_columns, group_values)) if groupby_columns else {}
        for word, freq in word_counts.items():
            row = {**group_dict, 'Word': word, 'Frequency': freq}
            results.append(row)

    return pd.DataFrame(results)

def plot_word_frequencies(
    df: pd.DataFrame,
    top_n: int = 20,
    groupby_column: Optional[str] = None,
    frequency_column: str = 'Frequency',
    level_column: str = 'Id'

):
    """
    Plot word frequencies by group (e.g., Experiment, Condition).

    Parameters:
    - df (pd.DataFrame): DataFrame containing word frequencies.
    - top_n (int): Number of top words to display per group.
    - groupby_column (str, optional): Column to group plots by (e.g., Experiment).
    - frequency_column (str): Column containing frequency values.
    - level_column (str): Column defining the granularity level for averaging (e.g., File Name or ID).

    Returns:
    - None
    """
    grouped = df.groupby(groupby_column) if groupby_column else [(None, df)]

    num_groups = len(grouped) if groupby_column else 1
    fig, axes = plt.subplots(num_groups, 1, figsize=(10, 5 * num_groups), squeeze=False)
    axes = axes.flatten()

    for idx, (group_name, group_df) in enumerate(grouped):
        # For each word sum their frequency and divide by column of interest
        n_level = group_df[level_column].nunique()
        group_df = group_df.groupby('Word').agg({frequency_column: 'sum'}).reset_index()
        group_df[frequency_column] = group_df[frequency_column] / n_level


        # Get top N words for the group
        top_words = group_df.nlargest(top_n, frequency_column)

        sns.barplot(data=top_words, x=frequency_column, y='Word', ax=axes[idx])

        title = f'Top {top_n} Words in {groupby_column}: {group_name} (n={n_level})' if groupby_column else 'Top Words'
        axes[idx].set_title(title)
        axes[idx].set_xlabel('Frequency')
        axes[idx].set_ylabel('Word')

    plt.tight_layout()
    plt.show()

def count_unique_words(
    df: pd.DataFrame,
    groupby_columns: List[str],
    unique_column: str = "Id",
    tokenized_column: str = 'preprocessed_content'
) -> pd.DataFrame:
    """
    Counts unique words that appear across different categories within specified groupings.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - groupby_columns (list of str): List of column names to group by.
    - unique_column (str): Column name for the unique identifier (e.g., ID).
    - tokenized_column (str): Column name for the pre-tokenized text content.

    Returns:
    - pd.DataFrame: DataFrame with group columns, 'Word', and 'Participant_Count'.
    """
    results = []

    for group_values, group in df.groupby(groupby_columns):
        word_participant_count = Counter()
        for participant_id, participant_data in group.groupby(unique_column):
            unique_words = set()

            for tokens in participant_data[tokenized_column].dropna():
                unique_words.update(tokens.split())
            word_participant_count.update(unique_words)

        group_dict = dict(zip(groupby_columns, group_values)) if isinstance(group_values, tuple) else {groupby_columns[0]: group_values}
        for word, count in word_participant_count.items():
            row = {**group_dict, 'Word': word, 'Participant_Count': count}
            results.append(row)

    results_df = pd.DataFrame(results)
    return results_df

def generate_word_clouds(
    df: pd.DataFrame,
    groupby_columns: Optional[List[str]] = None,
    filter_values: Optional[List[List[str]]] = None,
    max_words: int = 15,
    min_count: int = 2,
    save_fig: bool = False

):
    """
    Generates word clouds based on participant counts for each unique combination
    of the specified grouping columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing unique word counts.
    - groupby_columns (list of str, optional): List of columns to group by.
    - filter_values (list of lists, optional): Specific values to filter by.
    - max_words (int): Maximum number of words to display in the word cloud.
    - min_count (int): Minimum count of word repetitions to include in the word cloud.
    """
    if filter_values and groupby_columns:
        for i, column in enumerate(groupby_columns):
            if i < len(filter_values) and filter_values[i]:
                df = df[df[column].isin(filter_values[i])]

    # Filter words based on min_count
    df = df[df['Participant_Count'] >= min_count]

    groups = df.groupby(groupby_columns) if groupby_columns else [(None, df)]

    for group_values, group in groups:
        word_counts = dict(zip(group['Word'], group['Participant_Count']))

        wordcloud = WordCloud(
            width=1600, height=400,
            background_color='white',
            colormap='viridis',
            max_words=max_words,
            contour_width=1,
            contour_color='black'
        ).generate_from_frequencies(word_counts)

        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        title = ", ".join(f"{col}: {val}" for col, val in zip(groupby_columns, group_values)) if groupby_columns else "Word Cloud"
        plt.title(title)
        plt.tight_layout()

        if save_fig:
            # Sanitize filename
            sanitized_title = title.replace(":", "_").replace(",", "_").replace(" ", "_")
            plt.savefig(f'./outputs/wordcloud_{sanitized_title}.png', dpi=800,bbox_inches="tight")
        plt.show()
