import matplotlib.pyplot as plt

def plot_portfolio_performance(data, weights):
    """
    Plot the cumulative returns of the optimized portfolio.
    
    Parameters:
        data (pd.DataFrame): Combined DataFrame with TSLA, BND, and SPY prices.
        weights (np.array): Portfolio weights for each asset.
    """
    daily_returns = data.pct_change().dropna()
    portfolio_daily_returns = daily_returns.dot(weights)
    cumulative_returns = (1 + portfolio_daily_returns).cumprod()
    
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label='Optimized Portfolio')
    plt.title('Cumulative Returns of Optimized Portfolio')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()