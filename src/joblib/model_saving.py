import joblib

def save_model_and_scaler(model, scaler, model_path="model.joblib", scaler_path="scaler.joblib"):
    """Save the trained model and scaler to disk."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"{model_path} and {scaler_path} saved successfully.")

def save_clustered_data(customer_df, output_path="clustered_customers.csv"):
    """Save the clustered customer data to a CSV file."""
    customer_df.to_csv(output_path, index=False)
    print(f"{output_path} saved successfully.")

if __name__ == "__main__":
    # Example usage (assuming sample model, scaler, and customer data)
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import numpy as np
    
    # Sample data
    sample_data = pd.DataFrame({
        'Customer_ID': ['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005'],
        'Age': [34, 26, 50, 37, 30],
        'Gender': [0, 1, 0, 0, 0],
        'Total_Spent': [150, 1000, 30, 500, 100],
        'Avg_Spent': [150.0, 1000.0, 30.0, 500.0, 100.0],
        'Num_Transactions': [1, 1, 1, 1, 1]
    })
    
    # Sample features and scaling
    features = ['Age', 'Gender', 'Total_Spent', 'Avg_Spent', 'Num_Transactions']
    X = sample_data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Sample model
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_scaled)
    sample_data['Cluster'] = kmeans.predict(X_scaled)
    
    # Save model, scaler, and data
    save_model_and_scaler(kmeans, scaler)
    save_clustered_data(sample_data)