# src/data_processing.py

import pandas as pd


def load_and_clean_data(filepath):
    """
    Load and clean the retail transactional dataset.
    - Drop missing values
    - Convert dates
    - Remove duplicates
    """
    df = pd.read_csv(filepath)
    df.dropna(subset=['Customer ID', 'Age', 'Gender', 'Total Amount'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.drop_duplicates(inplace=True)
    return df


def aggregate_customer_level(df):
    """
    Aggregate transaction data to customer level.
    Returns a customer DataFrame with total spent, avg spent, etc.
    """
    customer_df = df.groupby('Customer ID').agg({
        'Age': 'first',
        'Gender': 'first',
        'Total Amount': ['sum', 'mean'],
        'Transaction ID': 'count'
    }).reset_index()

    # Flatten MultiIndex columns
    customer_df.columns = ['Customer_ID', 'Age', 'Gender', 'Total_Spent', 'Avg_Spent', 'Num_Transactions']

    # Encode gender
    customer_df['Gender'] = customer_df['Gender'].map({'Male': 0, 'Female': 1})
    return customer_df
