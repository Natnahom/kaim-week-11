import numpy as np
import pandas as pd
import joblib
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def fit_sarima(train_data):
    # """
    # Fit a SARIMA model using auto_arima.
    # """
    # # Debug: Print the head of train_data
    # print("train_data head:")
    # print(train_data.head())
    
    # # Fit the SARIMA model
    # model = auto_arima(train_data, seasonal=True, m=12, stepwise=True, trace=True)
    # print(f"Best SARIMA parameters: {model.order}, {model.seasonal_order}")
    # return model
    """
    Fit a SARIMA model with manually set parameters.
    """
    # Manually set SARIMA parameters (p, d, q) and (P, D, Q, m)
    model = SARIMAX(train_data, order=(1, 1, 0), seasonal_order=(2, 0, 0, 12))
    model_fit = model.fit(disp=False)
    return model_fit

def evaluate_sarima(model, test_data):
    """
    Evaluate the SARIMA model on the test data.
    """
    # Debug: Print the head of test_data
    print("test_data head:")
    print(test_data.head())
    
    # Generate predictions
    predictions = model.predict(n_periods=len(test_data))
    
    # Debug: Print the head of predictions
    print("predictions head:")
    print(predictions.head())
    
    # Ensure predictions is a 1D array
    predictions = np.squeeze(predictions)  # Convert to 1D array
    
    # Create a pandas Series for predictions with the same index as test_data
    predictions = pd.Series(predictions, index=test_data.index)
    
    # Check for NaN values in test_data and predictions
    if test_data.isnull().any().any():
        print("Warning: test_data contains NaN values. Filling NaN values with forward fill.")
        test_data = test_data.ffill()  # Forward fill NaN values
    
    if predictions.isnull().any().any():
        print("Warning: predictions contain NaN values. Filling NaN values with forward fill.")
        predictions = predictions.ffill()  # Forward fill NaN values
    
    # # Ensure no NaN values remain
    # if test_data.isnull().any().any() or predictions.isnull().any().any():
    #     raise ValueError("NaN values still exist in test_data or predictions after forward fill.")
    
    print(f"test_data head:\n{test_data.head()}")
    print(f"predictions head:\n{predictions.head()}")
    print(f"NaN values in test_data: {test_data.isnull().sum()}")
    print(f"NaN values in predictions: {predictions.isnull().sum()}")

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