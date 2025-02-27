import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler

def plot_closing_price(data):
    """
    Plot the closing price over time for each ticker.
    
    Parameters:
        data (dict): Dictionary of cleaned DataFrames.
    """
    plt.figure(figsize=(14, 8))
    for ticker, df in data.items():
        plt.plot(df['Date'], df['Close'], label=ticker)
    plt.title('Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

def calculate_daily_returns(data):
    """
    Calculate and plot daily percentage change for each ticker.
    
    Parameters:
        data (dict): Dictionary of cleaned DataFrames.
    """
    plt.figure(figsize=(14, 8))
    for ticker, df in data.items():
        df['Daily Return'] = df['Close'].pct_change()
        plt.plot(df['Date'], df['Daily Return'], label=ticker)
    plt.title('Daily Percentage Change')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.show()

def analyze_volatility(data, window=30):
    """
    Analyze volatility using rolling means and standard deviations.
    
    Parameters:
        data (dict): Dictionary of cleaned DataFrames.
        window (int): Rolling window size.
    """
    plt.figure(figsize=(14, 8))
    for ticker, df in data.items():
        df['Rolling Mean'] = df['Close'].rolling(window=window).mean()
        df['Rolling Std'] = df['Close'].rolling(window=window).std()
        
        plt.plot(df['Date'], df['Rolling Mean'], label=f'{ticker} Rolling Mean')
        plt.plot(df['Date'], df['Rolling Std'], label=f'{ticker} Rolling Std', linestyle='--')
    plt.title('Rolling Mean and Standard Deviation')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def decompose_time_series(data, ticker):
    """
    Decompose the time series into trend, seasonal, and residual components.
    
    Parameters:
        data (dict): Dictionary of cleaned DataFrames.
        ticker (str): Ticker symbol to decompose.
    """
    df = data[ticker]
    decomposition = seasonal_decompose(df['Close'], period=365, model='additive')
    
    plt.figure(figsize=(14, 8))
    decomposition.plot()
    plt.suptitle(f'Time Series Decomposition for {ticker}')
    plt.show()

