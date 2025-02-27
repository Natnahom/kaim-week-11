import yfinance as yf

def fetch_historical_data(tickers, start_date, end_date):
    """
    Fetch historical data for given tickers using yfinance.
    
    Parameters:
        tickers (list): List of ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        dict: A dictionary of DataFrames containing historical data for each ticker.
    """
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = df
    return data