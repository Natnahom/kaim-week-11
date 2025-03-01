import yfinance as yf
import pandas as pd

def fetch_data(ticker, start_date, end_date):
    """
    Fetch historical data for a given ticker using yfinance.
    
    Parameters:
        ticker (str): Ticker symbol (e.g., 'TSLA').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: Historical data for the ticker.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Close']]  # Use only the 'Close' column

def preprocess_data(data):
    """
    Preprocess the data by splitting into training and testing sets.
    
    Parameters:
        data (pd.DataFrame): Historical data with 'Close' prices.
    
    Returns:
        tuple: (train_data, test_data)
    """
    # Ensure the data is a pandas Series with a DateTime index
    if isinstance(data, pd.DataFrame):
        data = data['Close']  # Extract the 'Close' column if it's a DataFrame
    
    # Split data into training and testing sets
    train_size = int(len(data) * 0.8)  # 80% training, 20% testing
    train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]
    
    return train_data, test_data