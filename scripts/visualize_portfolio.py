import matplotlib.pyplot as plt
import numpy as np

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

def plot_asset_allocation(weights, tickers):
    """
    Plot a pie chart showing the optimized asset allocation.
    
    Parameters:
        weights (np.array): Optimized portfolio weights.
        tickers (list): List of asset tickers.
    """
    plt.figure(figsize=(6, 6))
    plt.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Optimized Asset Allocation')
    plt.show()

def calculate_portfolio_risk(weights, covariance_matrix):
    """
    Calculate the portfolio risk (volatility).
    
    Parameters:
        weights (np.array): Portfolio weights for each asset.
        covariance_matrix (pd.DataFrame): Covariance matrix of asset returns.
    
    Returns:
        float: Portfolio risk (volatility).
    """
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

def portfolio_return(weights, annual_returns):
    """
    Calculate the portfolio return.
    
    Parameters:
        weights (np.array): Portfolio weights for each asset.
        annual_returns (pd.Series): Annual returns for each asset.
    
    Returns:
        float: Portfolio return.
    """
    return np.dot(weights, annual_returns)

def plot_risk_return(annual_returns, covariance_matrix, optimized_weights):
    """
    Plot risk (volatility) vs. return for each asset and the optimized portfolio.
    
    Parameters:
        annual_returns (pd.Series): Annual returns for each asset.
        covariance_matrix (pd.DataFrame): Covariance matrix of asset returns.
        optimized_weights (np.array): Optimized portfolio weights.
    """
    # Calculate individual asset risks and returns
    asset_risks = np.sqrt(np.diag(covariance_matrix))
    asset_returns = annual_returns
    
    # Calculate portfolio risk and return
    portfolio_risk_value = calculate_portfolio_risk(optimized_weights, covariance_matrix)
    portfolio_return_value = portfolio_return(optimized_weights, annual_returns)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(asset_risks, asset_returns, s=200, label='Individual Assets', color='blue')
    plt.scatter(portfolio_risk_value, portfolio_return_value, s=200, label='Optimized Portfolio', color='red', marker='*')
    
    # Annotate points
    for i, ticker in enumerate(['TSLA', 'BND', 'SPY']):
        plt.annotate(ticker, (asset_risks[i], asset_returns[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Return')
    plt.title('Risk-Return Scatter Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_var_distribution(data, weights):
    """
    Plot the distribution of portfolio returns and highlight the Value at Risk (VaR).
    
    Parameters:
        data (pd.DataFrame): Combined DataFrame with TSLA, BND, and SPY prices.
        weights (np.array): Portfolio weights for each asset.
    """
    # Compute daily returns
    daily_returns = data.pct_change().dropna()
    
    # Compute portfolio daily returns
    portfolio_daily_returns = daily_returns.dot(weights)
    
    # Calculate VaR at 95% confidence level
    var_95 = np.percentile(portfolio_daily_returns, 5)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(portfolio_daily_returns, bins=50, density=True, alpha=0.6, color='blue', label='Portfolio Returns')
    plt.axvline(var_95, color='red', linestyle='--', label=f'VaR (95%): {var_95:.2%}')
    plt.xlabel('Daily Returns')
    plt.ylabel('Density')
    plt.title('Distribution of Portfolio Returns and Value at Risk (VaR)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cumulative_returns_comparison(data, weights):
    """
    Plot the cumulative returns of the optimized portfolio and individual assets.
    
    Parameters:
        data (pd.DataFrame): Combined DataFrame with TSLA, BND, and SPY prices.
        weights (np.array): Portfolio weights for each asset.
    """
    # Compute daily returns
    daily_returns = data.pct_change().dropna()
    
    # Compute cumulative returns for individual assets
    cumulative_returns_individual = (1 + daily_returns).cumprod()
    
    # Compute cumulative returns for the optimized portfolio
    portfolio_daily_returns = daily_returns.dot(weights)
    cumulative_returns_portfolio = (1 + portfolio_daily_returns).cumprod()
    
    # Plot
    plt.figure(figsize=(10, 6))
    for ticker in ['TSLA', 'BND', 'SPY']:
        plt.plot(cumulative_returns_individual[ticker], label=ticker)
    plt.plot(cumulative_returns_portfolio, label='Optimized Portfolio', linewidth=2, color='black', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Cumulative Returns Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
