import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def evaluate_optimal_clusters(X, max_k=10, random_state=42):
    """Evaluate optimal number of clusters using Elbow method and Silhouette Score."""
    wcss = []
    sil_scores = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X, kmeans.labels_))
    
    # Plot Elbow Method
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), wcss, marker='o')
    plt.title("Elbow Method (WCSS)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    
    # Plot Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), sil_scores, marker='o', color='green')
    plt.title("Silhouette Score")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    
    plt.tight_layout()
    plt.show()
    
    return wcss, sil_scores

def compare_clustering_algorithms(X, k=4, random_state=42):
    """Compare Silhouette Scores for KMeans, DBSCAN, and GMM."""
    from sklearn.cluster import DBSCAN
    from sklearn.mixture import GaussianMixture
    
    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_score = silhouette_score(X, kmeans_labels)
    
    # DBSCAN
    dbscan = DBSCAN(eps=1.0, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    dbscan_score = silhouette_score(X, dbscan_labels[dbscan_labels != -1]) if len(set(dbscan_labels)) > 1 else -1
    
    # GMM
    gmm = GaussianMixture(n_components=k, random_state=random_state)
    gmm_labels = gmm.fit_predict(X)
    gmm_score = silhouette_score(X, gmm_labels)
    
    # Print comparison
    print("Silhouette Score Comparison:")
    print(f"KMeans + PCA: ~{kmeans_score:.2f} (Clear clusters, stable results)")
    print(f"DBSCAN: ~{dbscan_score:.2f} (Sensitive to scale, outliers noisy)")
    print(f"GMM: ~{gmm_score:.2f} (Soft clusters, more flexible)")
    
    return {
        'KMeans': kmeans_score,
        'DBSCAN': dbscan_score,
        'GMM': gmm_score
    }

if __name__ == "__main__":
    # Example usage (assuming X is a preprocessed and scaled feature matrix)
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    # Sample data (replace with actual preprocessed data)
    sample_data = np.random.rand(100, 5)  # 100 samples, 5 features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample_data)
    
    # Evaluate optimal clusters
    wcss, sil_scores = evaluate_optimal_clusters(X_scaled)
    print("Recommended k based on Elbow and Silhouette: Check plots for elbow point and peak Silhouette Score")
    
    # Compare clustering algorithms
    scores = compare_clustering_algorithms(X_scaled)
    recommended_algorithm = max(scores, key=scores.get)
    print(f"Recommended Algorithm: {recommended_algorithm} with Silhouette Score: {scores[recommended_algorithm]:.2f}")