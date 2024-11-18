import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from wordcloud import WordCloud
import string
from collections import Counter
from typing import Optional, List, Set

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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

def stripplot(df,x_column,y_column, hue_column=None):

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x_column, y=y_column, data=df, hue=hue_column)
    sns.stripplot(x=x_column, y=y_column, data=df, hue=hue_column, linewidth=1, edgecolor="k",
                  dodge=True, jitter=False, legend=False)
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

def word_frequency_analysis(
    df: pd.DataFrame,
    content_column: str = 'Content',
    groupby_column: Optional[str] = None,
    omit_stop_words: bool = True,
    extra_stopwords: Optional[Set[str]] = None,
    top_n: int = 20
):
    """
    Plot word frequency analysis with subplots for each group in a specified column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - content_column (str): Column containing text content.
    - groupby_column (str, optional): The column name to group by. If None, no grouping.
    - omit_stop_words (bool): Whether to remove English stopwords.
    - extra_stopwords (set of str, optional): Additional words to exclude.
    - top_n (int): Number of top words to display.

    Returns:
    - None
    """
    stop_words_set = _prepare_stopwords(omit_stop_words, extra_stopwords)
    grouped = df.groupby(groupby_column) if groupby_column else [(None, df)]
    num_groups = len(grouped) if groupby_column else 1
    fig, axes = plt.subplots(num_groups, 1, figsize=(10, 5 * num_groups), squeeze=False)
    axes = axes.flatten()

    for idx, (group_name, group_df) in enumerate(grouped):
        tokens = _tokenize_texts(group_df[content_column], stop_words_set)
        top_n_df = _get_top_n_words(tokens, top_n)
        sns.barplot(
            data=top_n_df, x='Frequency', y='Word', hue="Word",
            dodge=False, ax=axes[idx], palette='viridis'
        )
        title = f'Top {top_n} Words in {groupby_column}: {group_name}' if groupby_column else 'Top Words'
        axes[idx].set_title(title)
        axes[idx].set_xlabel('Frequency')
        axes[idx].set_ylabel('Word')

    plt.tight_layout()
    plt.show()

def _prepare_stopwords(omit_stop_words: bool, extra_stopwords: Optional[Set[str]]) -> Set[str]:
    """
    Prepare the set of stopwords.

    Parameters:
    - omit_stop_words (bool): Whether to include English stopwords.
    - extra_stopwords (set of str, optional): Additional stopwords to include.

    Returns:
    - Set[str]: Set of stopwords.
    """
    stop_words_set = set(stopwords.words('english')) if omit_stop_words else set()
    if extra_stopwords:
        stop_words_set.update(extra_stopwords)
    return stop_words_set

def _tokenize_texts(texts: pd.Series, stop_words_set: Set[str]) -> List[str]:
    """
    Tokenize a series of texts into words, excluding stopwords and punctuation.

    Parameters:
    - texts (pd.Series): Series of text strings.
    - stop_words_set (set of str): Set of stopwords to exclude.

    Returns:
    - List[str]: List of cleaned tokens.
    """
    tokens = []
    lemmatizer = WordNetLemmatizer()
    for text in texts.dropna():
        for word in text.split():
            word_clean = word.strip(string.punctuation).lower()
            if word_clean and word_clean not in stop_words_set:
                lemma = lemmatizer.lemmatize(word_clean)
                tokens.append(lemma)
    return tokens

def _get_top_n_words(tokens: List[str], top_n: int) -> pd.DataFrame:
    """
    Get the top N words from a list of tokens.

    Parameters:
    - tokens (List[str]): List of word tokens.
    - top_n (int): Number of top words to return.

    Returns:
    - pd.DataFrame: DataFrame containing top words and their frequencies.
    """
    top_n_words = Counter(tokens).most_common(top_n)
    return pd.DataFrame(top_n_words, columns=['Word', 'Frequency'])

def count_unique_words(
    df: pd.DataFrame,
    groupby_columns: List[str],
    unique_column: str = "Id",
    content_column: str = 'Content',
    omit_stop_words: bool = True,
    extra_stopwords: Optional[Set[str]] = None
) -> pd.DataFrame:
    """
    Counts unique words that appear across different categories within specified groupings.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - groupby_columns (list of str): List of column names to group by.
    - unique_column (str): Column name for the unique identifier (e.g., ID).
    - content_column (str): Column name for the text content.
    - omit_stop_words (bool): Whether to remove English stopwords.
    - extra_stopwords (set of str, optional): Additional words to exclude.

    Returns:
    - pd.DataFrame: DataFrame with group columns, 'Word', and 'Participant_Count'.
    """
    stop_words_set = _prepare_stopwords(omit_stop_words, extra_stopwords)
    results = []

    for group_values, group in df.groupby(groupby_columns):
        word_participant_count = Counter()
        for participant_id, participant_data in group.groupby(unique_column):
            unique_words = set()
            for content in participant_data[content_column].dropna():
                unique_words.update(_tokenize_text(content, stop_words_set))
            word_participant_count.update(unique_words)
        group_dict = dict(zip(groupby_columns, group_values)) if isinstance(group_values, tuple) else {groupby_columns[0]: group_values}
        for word, count in word_participant_count.items():
            row = {**group_dict, 'Word': word, 'Participant_Count': count}
            results.append(row)
    results_df = pd.DataFrame(results)
    return results_df

def _tokenize_text(text: str, stop_words_set: Set[str]) -> Set[str]:
    """
    Tokenize a single text into unique words, excluding stopwords and punctuation.

    Parameters:
    - text (str): Text string to tokenize.
    - stop_words_set (set of str): Set of stopwords to exclude.

    Returns:
    - Set[str]: Set of unique cleaned tokens.
    """
    tokens = set()
    lemmatizer = WordNetLemmatizer()
    for word in text.split():
        word_clean = word.strip(string.punctuation).lower()
        if word_clean and word_clean not in stop_words_set:
            lemma = lemmatizer.lemmatize(word_clean)
            tokens.add(lemma)
    return tokens

def generate_word_clouds(
    df: pd.DataFrame,
    groupby_columns: Optional[List[str]] = None,
    filter_values: Optional[List[List[str]]] = None,
    max_words: int = 50
):
    """
    Generates word clouds based on participant counts for each unique combination
    of the specified grouping columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing unique word counts.
    - groupby_columns (list of str, optional): List of columns to group by.
    - filter_values (list of lists, optional): Specific values to filter by.
    - max_words (int): Maximum number of words to display in the word cloud.
    """
    if filter_values and groupby_columns:
        for i, column in enumerate(groupby_columns):
            if i < len(filter_values) and filter_values[i]:
                df = df[df[column].isin(filter_values[i])]

    groups = df.groupby(groupby_columns) if groupby_columns else [(None, df)]

    for group_values, group in groups:
        word_counts = dict(zip(group['Word'], group['Participant_Count']))
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=max_words,
            contour_width=1,
            contour_color='black'
        ).generate_from_frequencies(word_counts)

        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        title = ", ".join(f"{col}: {val}" for col, val in zip(groupby_columns, group_values)) if groupby_columns else "Word Cloud"
        plt.title(title)
        plt.show()