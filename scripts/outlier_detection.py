def detect_outliers(data, ticker):
    """
    Detect outliers in daily returns using the IQR method.
    
    Parameters:
        data (dict): Dictionary of cleaned DataFrames.
        ticker (str): Ticker symbol to analyze.
    """
    df = data[ticker]
    df['Daily Return'] = df['Close'].pct_change()
    
    Q1 = df['Daily Return'].quantile(0.25)
    Q3 = df['Daily Return'].quantile(0.75)
    IQR = Q3 - Q1
    
    outliers = df[(df['Daily Return'] < (Q1 - 1.5 * IQR)) | (df['Daily Return'] > (Q3 + 1.5 * IQR))]
    print(f"Outliers in {ticker}:")
    print(outliers[['Date', 'Daily Return']])