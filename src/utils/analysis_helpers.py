import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import string
from collections import Counter

# Install NLTK stopwords if not already installed
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')
from nltk.corpus import stopwords


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


def word_frequency_plot(df, groupby_column=None, omit_stop_words=True, extra_stopwords=None, top_n=20):
    """
    Plot word frequency analysis with subplots for each group in a specified column.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - groupby_column (str or None): The column name to group by (e.g., 'Experiment'). If None, no grouping.
    - stop_words (bool): Whether to remove English stopwords.
    - top_n (int): Number of top words to display.
    
    Returns:
    - combined_top_n_df (pd.DataFrame): DataFrame containing the top words and their frequencies.
    """
    
    # Set up stopwords and punctuation filters
    stop_words_set = set(stopwords.words('english')) if omit_stop_words else set()
    stop_words_set.update(extra_stopwords or [])  # Add any additional stopwords if provided
    
    # Function to clean and tokenize text
    def tokenize(text):
        tokens = []
        for word in text.split():
            word_clean = word.strip(string.punctuation).lower()
            if word_clean and word_clean not in stop_words_set:
                tokens.append(word_clean)
        return tokens
    
    # Prepare for grouping or use the entire DataFrame if no grouping column is specified
    grouped = df.groupby(groupby_column) if groupby_column else [(None, df)]
    num_groups = len(grouped) if groupby_column else 1
    
    # Initialize storage for results
    top_words_list = []

    # Set up subplots
    fig, axes = plt.subplots(num_groups, 1, figsize=(10, 5 * num_groups), squeeze=False)
    axes = axes.flatten() if groupby_column else [axes[0, 0]]  # Flatten axes only if grouped
    
    for idx, (group_name, group_df) in enumerate(grouped):
        # Tokenize and count word frequencies
        tokens = [token for content in group_df['Content'] for token in tokenize(content)]
        top_n_df = pd.DataFrame(Counter(tokens).most_common(top_n), columns=['Word', 'Frequency'])
        if groupby_column:
            top_n_df[groupby_column] = group_name  # Add group name if grouping
        
        top_words_list.append(top_n_df)

        sns.barplot(data=top_n_df, x='Frequency', y='Word', hue="Word", legend=False, ax=axes[idx], palette='viridis')
        title = f'Top {top_n} Words in {groupby_column}: {group_name}' if groupby_column else 'Top Words'
        axes[idx].set_title(title)
        axes[idx].set_xlabel('Frequency')
        axes[idx].set_ylabel('Word')
    
    plt.tight_layout()
    plt.show()

def count_unique_words(df, groupby_columns, unique_column="Id", content_column='Content', omit_stop_words=True, extra_stopwords=None):
    """
    Counts unique words that appear across different categories (e.g., IDs, participants) within specified groupings,
    excluding stopwords.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - groupby_columns (list of str): List of column names to group by (e.g., ['Experiment', 'Condition']).
    - unique_column (str): Column name for the unique identifier (e.g., ID).
    - content_column (str): Column name for the text content.
    - omit_stop_words (bool): Whether to remove English stopwords.
    - extra_stopwords (set of str, optional): Set of additional words to exclude from the count.

    Returns:
    - pd.DataFrame: A DataFrame with columns specified in `groupby_columns` plus 'Word' and 'Participant_Count'.
                    'Participant_Count' indicates how many unique IDs used each word.
    """
    # Set up stopwords
    stop_words_set = set(stopwords.words('english')) if omit_stop_words else set()
    stop_words_set.update(extra_stopwords or [])  # Add any additional stopwords if provided

    # Initialize a list to store results
    results = []

    # Function to clean and tokenize text
    def tokenize(text):
        tokens = []
        for word in text.split():
            word_clean = word.strip(string.punctuation).lower()
            if word_clean and word_clean not in stop_words_set:
                tokens.append(word_clean)
        return tokens

    # Group by specified columns
    for group_values, group in df.groupby(groupby_columns):
        # Dictionary to count words across unique IDs
        word_participant_count = Counter()

        # Group by unique participant identifier to get unique words per participant
        for participant_id, participant_data in group.groupby(unique_column):
            #print(participant_id)
            # Collect unique words used by this participant
            unique_words = set()
            for content in participant_data[content_column].dropna():
                # Filter out stopwords
                filtered_words = tokenize(content)
                unique_words.update(filtered_words)
            
            # Update word counts across participants
            word_participant_count.update(unique_words)
        
        # Prepare a dictionary of group values (e.g., Experiment and Condition)
        group_dict = dict(zip(groupby_columns, group_values)) if isinstance(group_values, tuple) else {groupby_columns[0]: group_values}
        
        # Append results with the group values, word, and participant count
        for word, count in word_participant_count.items():
            row = {**group_dict, 'Word': word, 'Participant_Count': count}
            results.append(row)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def generate_word_clouds(df, groupby_columns=None, filter_values=None, max_words=50):
    """
    Generates word clouds based on participant counts for each unique combination of the specified grouping columns.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing unique word counts (e.g., `unique_words_df`).
    - groupby_columns (list of str): List of columns to group by (e.g., ['Experiment', 'Condition']).
    - filter_values (list of lists): Specific values to filter by, where each inner list corresponds to the values
      for the corresponding column in `groupby_columns`.
    """
    # Apply filtering based on filter_values if provided
    if filter_values:
        for i, column in enumerate(groupby_columns):
            if i < len(filter_values) and filter_values[i]:  # Check if filter is provided for this column
                df = df[df[column].isin(filter_values[i])]

    # If no grouping columns are specified, create a single word cloud for the entire dataset
    if not groupby_columns:
        groups = [(None, df)]
    else:
        # Group by specified columns
        groups = df.groupby(groupby_columns)

    # Iterate over each group and generate a word cloud
    for group_values, group in groups:
        # Create a dictionary of words and their participant counts for this group
        word_counts = dict(zip(group['Word'], group['Participant_Count']))
        
        # Generate the word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=max_words,
            contour_width=1,
            contour_color='black'
        ).generate_from_frequencies(word_counts)
        
        # Display the word cloud
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')  # Hide the axes
        
        # Create a title based on group values
        if groupby_columns:
            title = ", ".join(f"{col}: {val}" for col, val in zip(groupby_columns, group_values))
        else:
            title = "Word Cloud"
        
        plt.title(title)
        plt.show()


# Cluster
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from sklearn.cluster import KMeans

# Function to preprocess text (with extended stopwords)
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    # Remove punctuation and lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Extend stopwords with domain-specific filler words
    stop_words = set(stopwords.words('english'))

    #domain_specific_stopwords = {'yeah', 'yes', 'like', 'you know', 'um', 'uh', 'dont', 'really', 'think', 'know', 'feel'}
    #stop_words = stop_words.union(domain_specific_stopwords)

    # Tokenize and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in stop_words]
    return ' '.join(tokens)


# Functions to perform clustering and return the clustered DataFrame

def cluster_tfidf_kmeans(df, n_clusters=2):
    vectorizer = TfidfVectorizer(max_features=1000)
    pca = PCA(n_components=2)
    
    clustered_dfs = []
    tfidf_matrices = []  # Store X_tfidf matrices for each experiment
    
    for experiment in df['Experiment'].unique():
        experiment_df = df[df['Experiment'] == experiment].copy()
        
        # TF-IDF transformation
        X_tfidf = vectorizer.fit_transform(experiment_df['preprocessed_content']).toarray()
        tfidf_matrices.append(X_tfidf)
        
        # PCA for dimensionality reduction
        X_pca = pca.fit_transform(X_tfidf)
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        experiment_df.loc[:, 'cluster'] = kmeans.fit_predict(X_tfidf)
        
        # Append the clustered DataFrame and PCA data for visualization
        clustered_dfs.append((experiment_df, X_pca))
    
    return clustered_dfs, tfidf_matrices  # Return both clustered DataFrames and TF-IDF matrices


from sentence_transformers import SentenceTransformer

def cluster_bert_kmeans(df, n_clusters=2):
    pca = PCA(n_components=2)
    clustered_dfs = []
    embedding_matrices = []  # Store X_embeddings for each experiment
    
    # Initialize BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for experiment in df['Experiment'].unique():
        experiment_df = df[df['Experiment'] == experiment].copy()
        
        # Compute BERT embeddings
        X_embeddings = model.encode(experiment_df['preprocessed_content'].tolist())
        embedding_matrices.append(X_embeddings)
        
        # Dimensionality reduction with PCA for visualization
        X_pca = pca.fit_transform(X_embeddings)
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        experiment_df['cluster'] = kmeans.fit_predict(X_embeddings)
        
        # Store experiment-specific DataFrame and PCA-transformed embeddings
        clustered_dfs.append((experiment_df, X_pca))
    
    return clustered_dfs, embedding_matrices  # Return both clustered DataFrames and embeddings


# Function to visualize clusters
def visualize_clusters(clustered_dfs):
    for experiment_df, X_pca in clustered_dfs:
        experiment = experiment_df['Experiment'].iloc[0]
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=experiment_df['cluster'], cmap='viridis', alpha=0.6)
        plt.title(f'Clustering for Experiment {experiment}')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(label='Cluster')
        plt.show()
