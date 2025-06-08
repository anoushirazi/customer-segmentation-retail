import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Import custom modules
from preprocess import preprocess_data
from feature_engineering import feature_engineering
from evaluation import evaluate_optimal_clusters, compare_clustering_algorithms
from visualization import plot_spending_by_gender, plot_elbow_and_silhouette, plot_clusters_pca
from profiling import profile_clusters, generate_marketing_actions
from model_saving import save_model_and_scaler, save_clustered_data

def main():
    # Define file path (adjust as needed)
    file_path = 'C:/Users/hh/Desktop/Target_Retail_Sales_Forecasting.csv'
    
    # Step 1: Preprocess data
    customer_df, X_scaled, scaler = preprocess_data(file_path)
    
    # Step 2: Feature engineering (alternative approach if needed)
    # customer_df = feature_engineering(pd.read_csv(file_path))  # Uncomment if using standalone feature engineering
    
    # Step 3: Evaluate optimal number of clusters
    wcss, sil_scores = evaluate_optimal_clusters(X_scaled)
    print("Recommended k based on Elbow and Silhouette: Check plots for elbow point and peak Silhouette Score")
    
    # Step 4: Apply KMeans with k=4 (based on PDF recommendation)
    kmeans = KMeans(n_clusters=4, random_state=42)
    customer_df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Step 5: Visualize results
    plot_spending_by_gender(customer_df)
    plot_elbow_and_silhouette(wcss, sil_scores)
    plot_clusters_pca(X_scaled, customer_df['Cluster'], title="Customer Segments (PCA Reduced)")
    
    # Step 6: Profile clusters and generate marketing actions
    cluster_summary = profile_clusters(customer_df)
    print("Cluster Summary Statistics:")
    print(cluster_summary)
    
    marketing_actions = generate_marketing_actions()
    print("\nMarketing Action Table:")
    print(marketing_actions)
    
    # Step 7: Save model, scaler, and clustered data
    save_model_and_scaler(kmeans, scaler)
    save_clustered_data(customer_df)

if __name__ == "__main__":
    main()