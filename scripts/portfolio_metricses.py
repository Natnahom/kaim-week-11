import numpy as np

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

def portfolio_risk(weights, covariance_matrix):
    """
    Calculate the portfolio risk (volatility).
    
    Parameters:
        weights (np.array): Portfolio weights for each asset.
        covariance_matrix (pd.DataFrame): Covariance matrix of asset returns.
    
    Returns:
        float: Portfolio risk (volatility).
    """
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

def sharpe_ratio(weights, annual_returns, covariance_matrix, risk_free_rate=0.02):
    """
    Calculate the Sharpe Ratio.
    
    Parameters:
        weights (np.array): Portfolio weights for each asset.
        annual_returns (pd.Series): Annual returns for each asset.
        covariance_matrix (pd.DataFrame): Covariance matrix of asset returns.
        risk_free_rate (float): Risk-free rate (default is 2%).
    
    Returns:
        float: Sharpe Ratio.
    """
    portfolio_ret = portfolio_return(weights, annual_returns)
    portfolio_vol = portfolio_risk(weights, covariance_matrix)
    return (portfolio_ret - risk_free_rate) / portfolio_vol