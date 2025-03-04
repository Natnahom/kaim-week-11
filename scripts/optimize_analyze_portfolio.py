import numpy as np
from scipy.optimize import minimize
from scripts.portfolio_metrics import portfolio_return, portfolio_risk, sharpe_ratio

def optimize_portfolio(annual_returns, covariance_matrix, risk_free_rate=0.02):
    """
    Optimize portfolio weights to maximize the Sharpe Ratio.
    
    Parameters:
        annual_returns (pd.Series): Annual returns for each asset.
        covariance_matrix (pd.DataFrame): Covariance matrix of asset returns.
        risk_free_rate (float): Risk-free rate (default is 2%).
    
    Returns:
        tuple: (optimized_weights, max_sharpe_ratio)
    """
    num_assets = len(annual_returns)
    initial_weights = np.array([1/num_assets] * num_assets)  # Equal weights
    bounds = tuple((0, 1) for _ in range(num_assets))  # Weights between 0 and 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Sum of weights = 1
    
    # Minimize negative Sharpe Ratio
    result = minimize(lambda weights: -sharpe_ratio(weights, annual_returns, covariance_matrix, risk_free_rate),
                      initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x, -result.fun

def analyze_portfolio(data, weights, annual_returns, covariance_matrix, risk_free_rate=0.02):
    """
    Analyze portfolio risk and return.
    
    Parameters:
        weights (np.array): Portfolio weights for each asset.
        annual_returns (pd.Series): Annual returns for each asset.
        covariance_matrix (pd.DataFrame): Covariance matrix of asset returns.
        risk_free_rate (float): Risk-free rate (default is 2%).
    
    Returns:
        dict: Portfolio metrics (return, volatility, VaR, Sharpe Ratio).
    """
    portfolio_ret = portfolio_return(weights, annual_returns)
    portfolio_vol = portfolio_risk(weights, covariance_matrix)
    sharpe = sharpe_ratio(weights, annual_returns, covariance_matrix, risk_free_rate)
    
    # Calculate Value at Risk (VaR) at 95% confidence level
    daily_returns = data.pct_change().dropna()
    portfolio_daily_returns = daily_returns.dot(weights)
    var_95 = np.percentile(portfolio_daily_returns, 5) * 100  # In percentage
    
    return {
        'Return': portfolio_ret,
        'Volatility': portfolio_vol,
        'VaR (95%)': var_95,
        'Sharpe Ratio': sharpe
    }