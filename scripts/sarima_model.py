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
    
    # Generate predictions for the same time period as test_data
    start_index = len(model.model.endog)  # Start predicting from the end of training data
    end_index = start_index + len(test_data) - 1  # End at the last index of test_data
    predictions = model.predict(start=start_index, end=end_index)
    
    # Debug: Print the head of predictions
    print("predictions head:")
    print(predictions.head())
    
    # Ensure predictions is a 1D array
    predictions = np.squeeze(predictions)  # Convert to 1D array

    if test_data.isnull().any().any():
        print("NaN values found in test_data:")
        print(test_data[test_data.isnull()])

    # Check for NaN values in test_data and predictions
    if predictions.isnull().any():
        print("Warning: predictions contain NaN values. Filling NaN values with forward fill.")
        predictions = predictions.ffill()  # Forward fill NaN values
    
    # Calculate evaluation metrics only if there are no NaN values
    if not test_data.isnull().any().any() and not predictions.isnull().any():
        mae = mean_absolute_error(test_data, predictions)
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
    else:
        raise ValueError("Input contains NaN values in either test_data or predictions.")

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