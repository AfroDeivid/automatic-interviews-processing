
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
