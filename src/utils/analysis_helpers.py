import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from typing import Optional, Union, List, Set

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
    retain_stopwords: Optional[Set[str]] = None,
    return_stopwords: bool = False
) -> Union[str, List[str]]:
    """
    Preprocess text using spaCy, including tokenization, lemmatization,
    stopword removal, and lowercasing.

    Args:
        text (str): The input text to preprocess.
        remove_stopwords (bool): Whether to remove stopwords, punctuation, and whitespace. Default is True.
        extra_stopwords (Optional[Set[str]]): Additional custom stopwords to remove. Default is None.
        retain_stopwords (Optional[Set[str]]): Specific stopwords to retain even if they are stopwords. Default is None.
        return_stopwords (bool): Whether to return the list of stopwords used in preprocessing. Default is False.

    Returns:
        Union[str, List[str]]: The preprocessed text as a single string, or the list of stopwords used if return_stopwords is True.
    """
    if not text:
        return "" if not return_stopwords else []

    # Process the text using spaCy
    doc = nlp(text)

    # Get the default stopwords from spaCy
    default_stopwords = set(nlp.Defaults.stop_words)

    # Combine default stopwords with extra stopwords
    all_stopwords = default_stopwords.union(extra_stopwords or set())

    # Remove retained stopwords from the final stopword list
    if retain_stopwords:
        all_stopwords.difference_update(retain_stopwords)

    # Initialize an empty list to hold processed tokens
    tokens = []

    for token in doc:
        # Lemmatize and lowercase
        lemma = token.lemma_.lower()

        # Apply stopword filter
        if remove_stopwords and lemma in all_stopwords:
            continue

        # Filter punctuation and spaces
        if token.is_punct or token.is_space:
            continue

        # Add valid tokens to the list
        tokens.append(lemma)

    if return_stopwords:
        return list(all_stopwords)

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

### Topics Analysis
import networkx as nx
import math
import matplotlib.colors as mcolors


def build_network_from_interviews(df_interviews, include_self_loops=True):
    """
    Build a directed network graph from interviews, aggregating topic transitions across all interviews.
    
    Parameters:
    - df_interviews (pd.DataFrame): DataFrame containing interview data with columns 'File Name', 'turn_index', and 'one_topic_name'.
    - include_self_loops (bool): Whether to include self-loops in the graph.
    
    Returns:
    - G (nx.DiGraph): Directed graph with aggregated topic transitions and edge weights and node attributes (topic counts).

    Note:
    The nodes in the graph built by build_network_from_interviews contain attributes that correspond to 
    the options available for the size_by parameter in ``plot_topic_transition_network``:
    - occurrence: Calculated as the total number of times a topic appears in the dataset (total mentions).
    - appearance: Represents the number of unique interviews in which a topic appears.
    """
    # Initialize directed graph
    G = nx.DiGraph()

    # Count topic occurrences (total mentions) across all interviews
    occurrence_counts = Counter(df_interviews["one_topic_name"])

    # Count topic appearances (number of files where a topic appears)
    appearance_counts = (
        df_interviews.groupby("one_topic_name")["File Name"].nunique().to_dict()
    )

    # Add nodes with both counts as attributes
    for topic in occurrence_counts:
        G.add_node(
            topic,
            occurrence=occurrence_counts[topic],
            appearance=appearance_counts.get(topic, 0),
        )

    # Group by interview ID to process each interview path
    grouped = df_interviews.sort_values("turn_index").groupby("File Name")

    for interview_id, group in grouped:
        # Get the ordered sequence of topics for the interview
        topics = group["one_topic_name"].tolist()

        # Add edges for consecutive topics in the sequence
        for i in range(len(topics) - 1):
            u = topics[i]
            v = topics[i + 1]
            if u != v or include_self_loops:
                # Add or update edge with weight
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1
                else:
                    G.add_edge(u, v, weight=1)

    return G
def plot_topic_transition_network(
    G,
    title="Topic Transition Network",
    show_edge_labels=True,
    file_name=None,
    background_color="white",
    size_by="occurrence",  # Options: 'occurrence', 'appearance', or 'degree_centrality'
    min_size=1000,          # Minimum node size
    max_size=4000,          # Maximum node size
    palette=None,
    legend_outside=False,  # Option to place legend outside the plot
):
    """
    Plot the directed topic transition network with numeric labels, enhanced arrow visibility, and adjusted edge margins.
    
    Parameters:
        G (Graph): The directed graph to plot (topics as nodes, transitions as edges).
        title (str): The title of the plot.
        show_edge_labels (bool): Whether to show edge weight labels.
        file_name (str, optional): Path to save the plot as an image file. If None, the plot is not saved.
        background_color (str): Background color of the plot ('white' or 'transparent').
        size_by (str): Criterion for node size ('occurrence', 'appearance', or 'degree_centrality').
        min_size (int): Minimum node size after normalization.
        max_size (int): Maximum node size after normalization.
        palette (dict, optional): Predefined color palette (e.g., {0: '#FF5733', 1: '#33FF57'}). If None, a palette will be generated.
        legend_outside (bool): Whether to place the legend outside the plot.
    """

    # Determine the background color
    facecolor = "none" if background_color == "transparent" else background_color

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(15, 10), facecolor=facecolor)
    pos = nx.spring_layout(G, seed=42, k=0.5, scale=3)

    # Extract numeric IDs from node names (e.g., "2_see_myself_awake_love" -> "2")
    numeric_labels = {node: node.split("_")[0] for node in G.nodes()}

    # Generate a palette if none is provided
    if palette is None:
        cmap = plt.get_cmap("tab20", len(G.nodes()))  # Default colormap with distinct colors
        palette = {node: mcolors.to_hex(cmap(i)) for i, node in enumerate(G.nodes())}
        node_colors = [palette[node] for node in G.nodes()]
    else:
        node_colors = [palette[int(numeric_labels[node])] for node in G.nodes()]

    # Node sizes based on the selected criterion
    if size_by == "occurrence":
        values = [G.nodes[node].get("occurrence", 0) for node in G.nodes()]
    elif size_by == "appearance":
        values = [G.nodes[node].get("appearance", 0) for node in G.nodes()]
    elif size_by == "degree_centrality":
        centrality = nx.degree_centrality(G)
        values = [centrality[node] for node in G.nodes()]
    else:
        raise ValueError("size_by must be 'occurrence', 'appearance', or 'degree_centrality'.")

    # Normalize values to the specified range
    if values:
        min_val, max_val = min(values), max(values)
        if max_val > min_val:  # Avoid division by zero
            normalized_sizes = [
                min_size + (value - min_val) / (max_val - min_val) * (max_size - min_size)
                for value in values
            ]
        else:
            normalized_sizes = [min_size for _ in values]  # All nodes the same size if no variation
    else:
        normalized_sizes = [min_size for _ in G.nodes()]  # Default sizes if no data

    node_sizes_dict = {node: size for node, size in zip(G.nodes(), normalized_sizes)}

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_size=normalized_sizes, node_color=node_colors, edgecolors="white", linewidths=1.5, ax=ax
    )
    
    # Draw numeric node labels
    nx.draw_networkx_labels(G, pos, labels=numeric_labels, font_size=16, font_color="white", ax=ax)

    # Draw edges with enhanced arrows and adjusted margins
    all_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_weight = max(all_weights) if all_weights else 1

    for u, v, data in G.edges(data=True):
        weight = data["weight"]
        edge_width = weight / max_weight * 3
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=edge_width,
            arrowstyle="-|>",  # Arrow style for sharp arrowheads
            arrowsize=20,  # Increased arrow size for visibility
            edge_color="#2C3E50",  # Dark edge color
            alpha=0.8,  # Slight transparency for polished look
            min_source_margin=math.sqrt(node_sizes_dict[u] / math.pi),  # Adjust for source node size
            min_target_margin=math.sqrt(node_sizes_dict[v] / math.pi),  # Adjust for target node size
            ax=ax
        )

    # Draw edge labels if enabled
    if show_edge_labels:
        edge_labels = {(u, v): f"{data['weight']}" for u, v, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, ax=ax)

    # Add legend with full topic names
    legend_labels = {numeric_labels[node]: node for node in G.nodes()}
    handles = [
        plt.Line2D([0], [0], marker="o", color=palette[int(key)], linestyle="", markersize=10)
        for key in sorted(legend_labels.keys())
    ]
    labels = [f"{value}" for _, value in sorted(legend_labels.items())]

    # # Adjust legend position
    if legend_outside:
        plt.legend(
            handles, labels, title="Topics", loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=14, title_fontsize=16
        )
    else:
        plt.legend(
            handles, labels, title="Topics", loc="best", fontsize=14, title_fontsize=16
        )

    # Final plot adjustments
    plt.title(title, fontsize=20, color="#3b3b3b")
    plt.axis("off")
    plt.tight_layout()

    # Save plot if requested
    if file_name:
        plt.savefig(file_name, dpi=600, bbox_inches="tight")

    plt.show()