import pandas as pd
import yfinance as yf

def fetch_historical_data(tickers, start_date, end_date):
    """
    Fetch historical data for given tickers using yfinance.
    
    Parameters:
        tickers (list): List of ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: Historical data for the tickers.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def create_combined_dataframe(tsla_forecast, bnd_data, spy_data):
    """
    Create a combined DataFrame with forecasted TSLA prices and historical BND and SPY prices.
    
    Parameters:
        tsla_forecast (pd.Series): Forecasted TSLA prices.
        bnd_data (pd.Series): Historical BND prices.
        spy_data (pd.Series): Historical SPY prices.
    
    Returns:
        pd.DataFrame: Combined DataFrame with TSLA, BND, and SPY prices.
    """
    combined_data = pd.DataFrame({
        'TSLA': tsla_forecast,
        'BND': bnd_data[-len(tsla_forecast):],  # Align BND data with TSLA forecast
        'SPY': spy_data[-len(tsla_forecast):]   # Align SPY data with TSLA forecast
    })
    return combined_data

def compute_annual_returns_and_covariance(data):
    """
    Compute annual returns and covariance matrix for the assets.
    
    Parameters:
        data (pd.DataFrame): Combined DataFrame with TSLA, BND, and SPY prices.
    
    Returns:
        tuple: (annual_returns, covariance_matrix)
    """
    # Compute daily returns
    daily_returns = data.pct_change().dropna()
    
    # Compute annual returns (assuming 252 trading days in a year)
    annual_returns = daily_returns.mean() * 252
    
    # Compute covariance matrix
    covariance_matrix = daily_returns.cov() * 252
    
    return annual_returns, covariance_matrix