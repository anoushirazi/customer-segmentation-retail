# src/utils.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import joblib
import os


def load_data(filepath):
    """
    Loads the CSV dataset.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Data loaded successfully from {filepath}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def save_model(model, filename):
    """
    Saves a model to a .joblib file.
    """
    try:
        joblib.dump(model, filename)
        print(f"✅ Model saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")


def save_dataframe(df, filename):
    """
    Saves a dataframe to a CSV file.
    """
    try:
        df.to_csv(filename, index=False)
        print(f"✅ DataFrame saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving dataframe: {e}")


def plot_elbow_and_silhouette(X_scaled, max_k=10, save_path=None):
    """
    Plots the elbow curve and silhouette scores.
    """
    from sklearn.cluster import KMeans

    wcss = []
    sil_scores = []

    K = range(2, max_k + 1)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        wcss.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(K, wcss, marker='o')
    plt.title('Elbow Method (WCSS)')
    plt.xlabel('k')
    plt.ylabel('WCSS')

    plt.subplot(1, 2, 2)
    plt.plot(K, sil_scores, marker='o', color='green')
    plt.title('Silhouette Scores')
    plt.xlabel('k')
    plt.ylabel('Score')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()


def ensure_dir(directory):
    """
    Create a directory if it doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📂 Created directory: {directory}")

