from typing import Optional, Tuple, Set, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from sentence_transformers import SentenceTransformer

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# Ensure NLTK resources are downloaded
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

def preprocess_text(
    text: str,
    lemmatize: bool = True,
    remove_stopwords: bool = True,
    ngrams: int = 1,
    extra_stopwords: Optional[Set[str]] = None
) -> str:
    """
    Preprocess the input text with options for lemmatization, stopword removal, and n-grams.
    """
    # Lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        if extra_stopwords:
            stop_words.update(extra_stopwords)
        tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Generate n-grams
    if ngrams > 1:
        tokens = ['_'.join(tokens[i:i+ngrams]) for i in range(len(tokens)-ngrams+1)]
    
    return ' '.join(tokens)


def cluster_text_kmeans(
    df: pd.DataFrame,
    text_column: str = 'preprocessed_content',
    method: str = 'tfidf',
    n_clusters: Optional[int] = None,
    n_components: int = 2,
    max_features: int = 1000,
    random_state: int = 42
) -> Tuple[List[Tuple[pd.DataFrame, np.ndarray]], List[np.ndarray]]:
    """
    Clusters text data using KMeans clustering with TF-IDF or BERT embeddings.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the text data.
    - text_column (str): Name of the column containing preprocessed text.
    - method (str): Embedding method, 'tfidf' or 'bert'.
    - n_clusters (int, optional): Number of clusters. If None, the optimal number is determined.
    - n_components (int): Number of components for dimensionality reduction.
    - max_features (int): Maximum number of features for TF-IDF vectorizer.
    - random_state (int): Random state for reproducibility.

    Returns:
    - clustered_dfs (list of tuples): List of tuples containing the DataFrame and reduced embeddings per experiment.
    - embeddings_list (list of np.ndarray): List of embeddings for each experiment.
    """
    if method not in ['tfidf', 'bert']:
        raise ValueError("Method must be 'tfidf' or 'bert'")

    clustered_dfs = []
    embeddings_list = []

    # Initialize vectorizer or model
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features)
    else:
        model = SentenceTransformer('all-MiniLM-L6-v2')

    # Iterate over experiments
    for experiment in df['Experiment'].unique():
        experiment_df = df[df['Experiment'] == experiment].copy()

        if method == 'tfidf':
            embeddings = vectorizer.fit_transform(experiment_df[text_column]).toarray()
        else:
            embeddings = model.encode(experiment_df[text_column].tolist())

        embeddings_list.append(embeddings)

        # Determine optimal number of clusters if n_clusters is None
        if n_clusters is None:
            range_n_clusters = list(range(2, min(10, len(experiment_df))))
            silhouette_avg_scores = []
            for n in range_n_clusters:
                kmeans = KMeans(n_clusters=n, random_state=random_state)
                cluster_labels = kmeans.fit_predict(embeddings)
                silhouette_avg = silhouette_score(embeddings, cluster_labels)
                silhouette_avg_scores.append(silhouette_avg)
            # Select the number of clusters with the highest silhouette score
            optimal_n_clusters = range_n_clusters[np.argmax(silhouette_avg_scores)]
            print(f"Optimal number of clusters for experiment {experiment}: {optimal_n_clusters}")
        else:
            optimal_n_clusters = n_clusters

        # KMeans clustering
        kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=random_state)
        experiment_df['cluster'] = kmeans.fit_predict(embeddings)

        # Dimensionality reduction for visualization
        if n_components > 0:
            pca = PCA(n_components=n_components, random_state=random_state)
            embeddings_reduced = pca.fit_transform(embeddings)
        else:
            embeddings_reduced = None

        # Append the clustered DataFrame and reduced embeddings
        clustered_dfs.append((experiment_df, embeddings_reduced))

    return clustered_dfs, embeddings_list


def visualize_clusters(
    clustered_dfs: List[Tuple[pd.DataFrame, np.ndarray]],
    n_components: int = 2,
    title_prefix: str = 'Clustering for Experiment'
):
    """
    Visualizes clusters for each experiment.

    Parameters:
    - clustered_dfs (list of tuples): List containing tuples of DataFrame and embeddings per experiment.
    - n_components (int): Number of dimensions in embeddings (2 or 3).
    - title_prefix (str): Prefix for the plot titles.
    """
    if n_components not in [2, 3]:
        print(f"Visualization is only supported for n_components=2 or 3.")
        return

    for experiment_df, embeddings_reduced in clustered_dfs:
        experiment = experiment_df['Experiment'].iloc[0]
        clusters = experiment_df['cluster']
        num_clusters = clusters.nunique()

        if n_components == 2:
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1],
                                  c=clusters, cmap='viridis', alpha=0.6)
            plt.title(f'{title_prefix} {experiment}')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.colorbar(label='Cluster')
            plt.show()
        elif n_components == 3:
            from mpl_toolkits.mplot3d import Axes3D  # For 3D plots
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1],
                                 embeddings_reduced[:, 2], c=clusters, cmap='viridis', alpha=0.6)
            ax.set_title(f'{title_prefix} {experiment}')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            plt.show()