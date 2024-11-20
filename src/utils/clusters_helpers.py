import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, Tuple, Set, List

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from sentence_transformers import SentenceTransformer


def embed_text_tfidf(
    texts: List[str],
    max_features: int = 1000
) -> np.ndarray:
    """
    Embeds text using TF-IDF vectorization.

    Parameters:
    - texts (List[str]): List of preprocessed texts.
    - max_features (int): Maximum number of features for the TF-IDF vectorizer.

    Returns:
    - embeddings (np.ndarray): TF-IDF embeddings.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    embeddings = vectorizer.fit_transform(texts).toarray()
    return embeddings

def embed_text_bert(
    texts: List[str],
    model_name: str = 'all-MiniLM-L6-v2'
) -> np.ndarray:
    """
    Embeds text using a pre-trained BERT model.

    Parameters:
    - texts (List[str]): List of preprocessed texts.
    - model_name (str): Name of the SentenceTransformer model to use.

    Returns:
    - embeddings (np.ndarray): BERT embeddings.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings

def cluster_text_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> Tuple[np.ndarray, KMeans]:
    """
    Clusters text embeddings using KMeans clustering.

    Parameters:
    - embeddings (np.ndarray): Text embeddings.
    - n_clusters (int): Number of clusters.
    - random_state (int): Random state for reproducibility.

    Returns:
    - cluster_labels (np.ndarray): Cluster labels for each text.
    - kmeans_model (KMeans): Fitted KMeans model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels, kmeans

def find_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int = 10,
    random_state: int = 42
) -> int:
    """
    Finds the optimal number of clusters using silhouette analysis and plots the scores.

    Parameters:
    - embeddings (np.ndarray): Text embeddings.
    - max_clusters (int): Maximum number of clusters to test.
    - random_state (int): Random state for reproducibility.

    Returns:
    - optimal_n_clusters (int): Optimal number of clusters.
    """
    range_n_clusters = list(range(2, min(max_clusters, len(embeddings))))
    silhouette_avg_scores = []

    # Compute silhouette scores for each number of clusters
    for n in range_n_clusters:
        kmeans = KMeans(n_clusters=n, random_state=random_state)
        cluster_labels = kmeans.fit_predict(embeddings)
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        silhouette_avg_scores.append(silhouette_avg)

    # Plot silhouette scores
    plt.figure(figsize=(8, 6))
    plt.plot(range_n_clusters, silhouette_avg_scores, marker='o', linestyle='-', color='b')
    plt.title("Silhouette Scores for Different Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.xticks(range_n_clusters)
    plt.grid(True)
    plt.show()

    # Find the optimal number of clusters
    optimal_n_clusters = range_n_clusters[np.argmax(silhouette_avg_scores)]
    print(f"Optimal number of clusters: {optimal_n_clusters}")
    #return optimal_n_clusters
    
def visualize_clusters_from_df(
    df: pd.DataFrame,
    embedding_col: str = 'Embedding',
    cluster_col: str = 'Cluster',
    shape_by: Optional[str] = None,
    n_components: int = 2,
    title: str = 'Cluster Visualization with Metadata'
):
    """
    Visualizes text clusters using embeddings and metadata with simplified shape legends.

    Parameters:
    - df (pd.DataFrame): DataFrame containing embeddings, cluster labels, and metadata.
    - embedding_col (str): Column name containing embeddings (default: 'Embedding').
    - cluster_col (str): Column name containing cluster labels (default: 'Cluster').
    - shape_by (str, optional): Column name to determine marker shapes (default: None).
    - n_components (int): Number of dimensions to reduce to (2 or 3).
    - title (str): Title for the plot.
    """
    if n_components not in [2, 3]:
        raise ValueError("Visualization supports only 2 or 3 dimensions.")

    # Extract embeddings and cluster labels
    embeddings = np.vstack(df[embedding_col])
    cluster_labels = df[cluster_col]

    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Prepare color mapping
    unique_clusters = cluster_labels.unique()
    cmap = plt.cm.get_cmap('viridis', len(unique_clusters))

    # Handle shape-based metadata if provided
    if shape_by:
        shape_labels = df[shape_by]
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X', 'h']
        unique_shapes = shape_labels.unique()
        shape_map = {label: markers[i % len(markers)] for i, label in enumerate(unique_shapes)}
    else:
        unique_shapes = [None]
        shape_map = {None: 'o'}

    # 2D Visualization
    if n_components == 2:
        plt.figure(figsize=(8, 6))

        # Plot points
        for shape, marker in shape_map.items():
            shape_mask = df[shape_by] == shape if shape_by else np.array([True] * len(df))
            for cluster in unique_clusters:
                cluster_mask = cluster_labels == cluster
                combined_mask = shape_mask & cluster_mask if shape_by else cluster_mask
                plt.scatter(
                    reduced_embeddings[combined_mask, 0],
                    reduced_embeddings[combined_mask, 1],
                    c=[cmap(cluster / len(unique_clusters))],
                    marker=marker,
                    alpha=0.7,
                )

        # Add color legend (for clusters)
        color_legend = plt.colorbar(label='Cluster')

        # Add shape legend
        if shape_by:
            for shape, marker in shape_map.items():
                plt.scatter([], [], color='gray', marker=marker, label=f'{shape}')
            plt.legend(title=shape_by, loc='best', fontsize=8)

        # Finalize plot
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()

    # 3D Visualization
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot points
        for shape, marker in shape_map.items():
            shape_mask = df[shape_by] == shape if shape_by else np.array([True] * len(df))
            for cluster in unique_clusters:
                cluster_mask = cluster_labels == cluster
                combined_mask = shape_mask & cluster_mask if shape_by else cluster_mask
                ax.scatter(
                    reduced_embeddings[combined_mask, 0],
                    reduced_embeddings[combined_mask, 1],
                    reduced_embeddings[combined_mask, 2],
                    c=[cmap(cluster / len(unique_clusters))],
                    marker=marker,
                    alpha=0.7,
                )

        # Add shape legend
        if shape_by:
            for shape, marker in shape_map.items():
                ax.scatter([], [], color='gray', marker=marker, label=f'{shape}')
            ax.legend(title=shape_by, loc='best', fontsize=8)

        ax.set_title(title)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plt.show()
