import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def prepare_lstm_data(data, lookback=60):
    """
    Prepare data for LSTM by creating sequences of lookback periods.
    
    Parameters:
        data (pd.DataFrame): Data with 'Close' prices.
        lookback (int): Number of time steps to look back.
    
    Returns:
        tuple: (X, y)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_lstm_model(input_shape):
    """
    Build an LSTM model.
    
    Parameters:
        input_shape (tuple): Shape of the input data.
    
    Returns:
        Sequential: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_lstm(model, X_test, y_test, scaler):
    """
    Evaluate the LSTM model on the test data.
    
    Parameters:
        model: Trained LSTM model.
        X_test (np.array): Test features.
        y_test (np.array): Test labels.
        scaler: MinMaxScaler used to scale the data.
    
    Returns:
        tuple: (predictions, mae, rmse, mape)
    """
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    return predictions, mae, rmse, mape

def save_lstm_model(model, filename='lstm_model.h5'):
    """
    Save the LSTM model to a file.
    """
    model.save(filename)
    print(f"LSTM model saved to {filename}")

def load_lstm_model(filename='lstm_model.h5'):
    """
    Load the LSTM model from a file.
    """
    model = tf.keras.models.load_model(filename)
    print(f"LSTM model loaded from {filename}")
    return model