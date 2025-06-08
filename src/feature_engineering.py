import pandas as pd

def feature_engineering(df):
    """Aggregate transactional data to per-customer level and encode gender."""
    # Group data by Customer ID and aggregate key metrics
    customer_df = df.groupby('Customer ID').agg({
        'Age': 'first',
        'Gender': 'first',
        'Total Amount': ['sum', 'mean'],
        'Transaction ID': 'count'
    }).reset_index()

    # Flatten the multi-level columns
    customer_df.columns = ['Customer_ID', 'Age', 'Gender', 'Total_Spent', 'Avg_Spent', 'Num_Transactions']
    
    # Encode Gender
    customer_df['Gender'] = customer_df['Gender'].map({'Male': 0, 'Female': 1})
    
    return customer_df

if __name__ == "__main__":
    # Example usage (assuming a sample DataFrame structure)
    import pandas as pd
    sample_data = pd.DataFrame({
        'Customer ID': ['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005'],
        'Transaction ID': [1, 2, 3, 4, 5],
        'Date': ['2023-11-24', '2023-02-27', '2023-01-13', '2023-05-21', '2023-05-06'],
        'Gender': ['Male', 'Female', 'Male', 'Male', 'Male'],
        'Age': [34, 26, 50, 37, 30],
        'Product Category': ['Beauty', 'Clothing', 'Electronics', 'Clothing', 'Beauty'],
        'Quantity': [3, 2, 1, 1, 2],
        'Price per Unit': [50, 500, 30, 500, 50],
        'Total Amount': [150, 1000, 30, 500, 100]
    })
    customer_df = feature_engineering(sample_data)
    print(customer_df.head())