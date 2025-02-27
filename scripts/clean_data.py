import pandas as pd

def clean_and_understand_data(data):
    """
    Clean and understand the data.
    
    Parameters:
        data (dict): Dictionary of DataFrames containing historical data for each ticker.
    
    Returns:
        dict: A dictionary of cleaned DataFrames.
    """
    cleaned_data = {}
    for ticker, df in data.items():
        # Check for missing values
        print(f"Missing values in {ticker}:")
        print(df.isnull().sum())
        
        # Handle missing values (fill with forward fill)
        df.fillna(method='ffill', inplace=True)
        
        # Ensure correct data types
        df['Date'] = df.index
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Check basic statistics
        print(f"\nBasic statistics for {ticker}:")
        print(df.describe())
        
        cleaned_data[ticker] = df
    return cleaned_data