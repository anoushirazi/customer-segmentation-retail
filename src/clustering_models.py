# src/clustering_models.py

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import pandas as pd


def run_kmeans(X_scaled, n_clusters=4):
    """
    Fits KMeans on scaled data and returns the model and labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels


def run_dbscan(X_scaled, eps=1.0, min_samples=5):
    """
    Applies DBSCAN clustering and returns labels.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    return labels


def run_gmm(X_scaled, n_components=4):
    """
    Applies Gaussian Mixture Model clustering and returns the model and labels.
    """
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(X_scaled)
    return gmm, labels


def apply_pca(X_scaled, n_components=2):
    """
    Reduces features to 2D using PCA for visualization.
    """
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    return components
