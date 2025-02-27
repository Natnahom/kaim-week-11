import numpy as np
import pandas as pd
import joblib
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

def fit_sarima(train_data):
    """
    Fit a SARIMA model using auto_arima to find the best parameters.
    
    Parameters:
        train_data (pd.DataFrame): Training data with 'Close' prices.
    
    Returns:
        SARIMAX: Fitted SARIMA model.
    """
    model = auto_arima(train_data, seasonal=True, m=12, stepwise=True, trace=True)
    print(f"Best SARIMA parameters: {model.order}, {model.seasonal_order}")
    return model

def evaluate_sarima(model, test_data):
    """
    Evaluate the SARIMA model on the test data.
    
    Parameters:
        model: Fitted SARIMA model.
        test_data (pd.DataFrame or pd.Series): Test data with 'Close' prices.
    
    Returns:
        tuple: (predictions, mae, rmse, mape)
    """
    # Ensure test_data is a 1D array
    if isinstance(test_data, pd.DataFrame):
        test_data = test_data['Close']  # Extract the 'Close' column if it's a DataFrame
    
    # Generate predictions
    predictions = model.predict(n_periods=len(test_data))
    
    # Ensure predictions is a 1D array
    predictions = np.squeeze(predictions)  # Convert to 1D array
    
    # Align indices (if necessary)
    predictions = pd.Series(predictions, index=test_data.index)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(test_data, predictions)
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
    
    return predictions, mae, rmse, mape

def save_sarima_model(model, filename='sarima_model.pkl'):
    """
    Save the SARIMA model to a file.
    """
    joblib.dump(model, filename)
    print(f"SARIMA model saved to {filename}")

def load_sarima_model(filename='sarima_model.pkl'):
    """
    Load the SARIMA model from a file.
    """
    model = joblib.load(filename)
    print(f"SARIMA model loaded from {filename}")
    return model