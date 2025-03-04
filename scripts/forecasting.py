import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings for cleaner output

def sarima_forecast(model, steps=12):
    """
    Generate future forecasts using the SARIMA model.
    
    Parameters:
        model: Trained SARIMA model.
        steps (int): Number of steps (months) to forecast.
    
    Returns:
        tuple: (forecast, confidence_intervals)
    """
    # Generate forecasts with confidence intervals
    forecast = model.get_forecast(steps=steps)
    forecast_values = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()
    
    return forecast_values, confidence_intervals

def lstm_forecast(model, last_sequence, steps=12, lookback=60, scaler_filename='scaler.pkl'):
    """
    Generate future forecasts using the LSTM model.
    
    Parameters:
        model: Trained LSTM model.
        last_sequence (np.array): Last sequence of historical data.
        steps (int): Number of steps (months) to forecast.
        lookback (int): Lookback period used for training the LSTM model.
        scaler_filename (str): Path to the saved scaler.
    
    Returns:
        np.array: Forecasted values.
    """
    # Load the scaler
    scaler = joblib.load(scaler_filename)
    
    predictions = []
    current_sequence = last_sequence.reshape(1, lookback, 1)
    
    for _ in range(steps):
        # Predict the next value
        next_value = model.predict(current_sequence, verbose=0)
        predictions.append(next_value[0][0])
        
        # Reshape next_value to match the dimensions of current_sequence
        next_value = next_value.reshape(1, 1, 1)  # Reshape to (1, 1, 1)
        
        # Update the sequence with the predicted value
        current_sequence = np.append(current_sequence[:, 1:, :], next_value, axis=1)
    
    # Inverse transform the predictions to original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

def plot_forecast(historical_data, forecast_values, confidence_intervals=None, title="Forecasted Stock Prices"):
    """
    Plot the forecasted values alongside historical data.
    
    Parameters:
        historical_data (pd.Series): Historical stock prices.
        forecast_values (pd.Series or np.array): Forecasted values.
        confidence_intervals (pd.DataFrame): Confidence intervals for the forecast.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(14, 7))
    
    # Plot historical data
    plt.plot(historical_data.index, historical_data, label='Historical Prices', color='blue')
    
    # Plot forecasted values
    forecast_index = pd.date_range(start=historical_data.index[-1], periods=len(forecast_values) + 1, freq='B')[1:]
    plt.plot(forecast_index, forecast_values, label='Forecasted Prices', color='red')
    
    # Plot confidence intervals (if available)
    if confidence_intervals is not None:
        plt.fill_between(forecast_index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Intervals')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()