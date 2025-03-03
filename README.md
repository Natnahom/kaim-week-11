# This is Week-11 of 10 academy

# Task 1: Preprocess and Explore Financial Data
This repository contains the code and documentation for Task 1 of the financial data analysis project. The goal of this task is to preprocess and explore historical financial data for three key assets: Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY). The data is sourced from YFinance and covers the period from January 1, 2015, to January 31, 2025.

## Table of Contents
- Objective

- Data Description

- Steps Performed

- Code Structure

- Outputs

- How to Run the Code

- Dependencies

- Key Insights

## Objective
The objective of this task is to:

Fetch historical financial data for TSLA, BND, and SPY using the YFinance library.

Clean and preprocess the data to handle missing values and ensure proper data types.

Perform exploratory data analysis (EDA) to understand trends, volatility, and outliers.

Decompose the time series into trend, seasonal, and residual components.

Prepare the data for further modeling and analysis.

## Data Description
The dataset includes the following columns for each asset:

Date: Trading day timestamp.

Open, High, Low, Close: Daily price metrics.

Adj Close: Adjusted close price to account for dividends and splits.

Volume: Total number of shares/units traded each day.

Asset-Specific Descriptions
TSLA: High-growth, high-risk stock in the consumer discretionary sector (Automobile Manufacturing).

BND: A bond ETF tracking U.S. investment-grade bonds, providing stability and income.

SPY: An ETF tracking the S&P 500 Index, offering broad U.S. market exposure.

Steps Performed
Fetch Historical Data:

Use the YFinance library to download historical data for TSLA, BND, and SPY.

- Data Cleaning and Understanding:

Check for missing values and handle them using forward fill.

Ensure proper data types for all columns.

Compute basic statistics (mean, standard deviation, etc.) for each dataset.

- Exploratory Data Analysis (EDA):

Visualize the closing price over time for all assets.

Calculate and plot daily percentage changes to observe volatility.

Analyze volatility using rolling means and standard deviations.

Detect outliers in daily returns using the Interquartile Range (IQR) method.

- Time Series Decomposition:

Decompose the time series into trend, seasonal, and residual components for TSLA.

## Code Structure
The code is modular and organized into functions for better readability and reusability. Below is the structure of the code:

Functions
- fetch_historical_data:

Fetches historical data for the given tickers using YFinance.

- clean_and_understand_data:

Cleans the data by handling missing values and ensuring proper data types.

Computes basic statistics for each dataset.

- plot_closing_price:

Plots the closing price over time for all assets.

- calculate_daily_returns:

Calculates and plots daily percentage changes for all assets.

- analyze_volatility:

Analyzes volatility using rolling means and standard deviations.

- decompose_time_series:

Decomposes the time series into trend, seasonal, and residual components.

- detect_outliers:

Detects outliers in daily returns using the IQR method.

- analysis.ipynb:

Executes all the above functions in sequence.

## Outputs
- Closing Price Over Time:

A plot showing the closing prices of TSLA, BND, and SPY over time.

- Daily Percentage Change:

A plot showing the daily returns of the assets.

- Rolling Mean and Standard Deviation:

A plot showing the rolling mean and standard deviation for each asset.

- Time Series Decomposition:

A decomposition plot for TSLA showing trend, seasonal, and residual components.

- Outliers:

A list of outliers in TSLA's daily returns.

## How to Run the Code
- Clone the repository:

- Install the required dependencies:

    - pip install -r requirements.txt
- Run the Python script:
    - Follow analysis.ipynb

## Dependencies
The following Python libraries are required to run the code:

yfinance

pandas

numpy

matplotlib

seaborn

statsmodels

scikit-learn

You can install them using:

pip install yfinance pandas numpy matplotlib seaborn statsmodels scikit-learn

## Key Insights
- Trends:

TSLA shows significant growth over time, with high volatility.

BND and SPY exhibit more stable trends, with SPY showing moderate growth.

- Volatility:

TSLA has the highest volatility, as seen in the daily percentage changes and rolling standard deviations.

BND has the lowest volatility, reflecting its stability as a bond ETF.

- Outliers:

TSLA has several outliers in its daily returns, indicating days with unusually high or low returns.

- Seasonality:

The time series decomposition for TSLA reveals a clear trend and some seasonal patterns.

# Task 2: Develop Time Series Forecasting Models

This repository contains the code and documentation for Task 2 of the financial data analysis project. The goal of this task is to develop and evaluate time series forecasting models to predict Tesla's (TSLA) future stock prices. We will implement SARIMA and LSTM models, compare their performance, and use the best-performing model to optimize portfolio management strategies.

## Table of Contents
- Objective

- Data Description

- Steps Performed

- Code Structure

- Outputs

- How to Run the Code

- Dependencies

- Key Insights

## Objective
- The objective of this task is to:

Develop time series forecasting models (SARIMA and LSTM) to predict Tesla's stock prices.

Evaluate the models using metrics such as MAE, RMSE, and MAPE.

Compare the performance of SARIMA and LSTM models.

Use the best-performing model to optimize portfolio management strategies.

## Data Description
The dataset includes historical stock price data for Tesla (TSLA) from January 1, 2015, to January 31, 2025. The data is sourced from YFinance and includes the following columns:

- Date: Trading day timestamp.

- Open, High, Low, Close: Daily price metrics.

- Adj Close: Adjusted close price to account for dividends and splits.

- Volume: Total number of shares traded each day.

## Steps Performed
- Data Fetching and Preprocessing:

Fetch historical data for TSLA using the YFinance library.

Split the data into training and testing sets (80% training, 20% testing).

- SARIMA Model:

Fit a SARIMA model using auto_arima to automatically select the best parameters.

Generate predictions for the test data.

Evaluate the model using MAE, RMSE, and MAPE.

- LSTM Model:

Prepare the data for LSTM by creating sequences of lookback periods.

Build and train an LSTM model using TensorFlow/Keras.

Generate predictions for the test data.

Evaluate the model using MAE, RMSE, and MAPE.

- Model Comparison:

Compare the performance of SARIMA and LSTM models.

Visualize the actual prices, SARIMA predictions, and LSTM predictions.

## Code Structure
The code is modular and organized into functions for better readability and reusability. Below is the structure of the code:

## Functions
- fetch_data:

Fetches historical data for a given ticker using YFinance.

- preprocess_data:

Splits the data into training and testing sets.

- fit_sarima:

Fits a SARIMA model using auto_arima.

- evaluate_sarima:

Evaluates the SARIMA model on the test data.

- prepare_lstm_data:

Prepares the data for LSTM by creating sequences of lookback periods.

- build_lstm_model:

Builds an LSTM model using TensorFlow/Keras.

- evaluate_lstm:

Evaluates the LSTM model on the test data.

## Outputs
- SARIMA Metrics:

MAE, RMSE, and MAPE values for the SARIMA model.

- LSTM Metrics:

MAE, RMSE, and MAPE values for the LSTM model.

Plot:

A plot comparing actual prices, SARIMA predictions, and LSTM predictions.

## How to Run the Code
1. Clone the repository:

2. Install the required dependencies:

- pip install -r requirements.txt

3. Run the Python script:
- Follow the timeSeries_forcasting.ipynb

## Dependencies
The following Python libraries are required to run the code:

- yfinance

- pandas

- numpy

- matplotlib

- statsmodels

- pmdarima

- scikit-learn

- tensorflow

## You can install them using:

- pip install yfinance pandas numpy matplotlib statsmodels pmdarima scikit-learn tensorflow
Key Insights
- SARIMA Model:

Suitable for univariate time series with seasonality.

Automatically selects the best parameters using auto_arima.

Provides interpretable results but may struggle with highly volatile data.

- LSTM Model:

A deep learning model capable of capturing long-term dependencies.

Requires more data and computational resources but can handle complex patterns.

Often outperforms traditional statistical models for large datasets.

- Model Comparison:

SARIMA is faster to train and interpret but may underperform on highly volatile data.

LSTM is more flexible and accurate but requires more data and computational resources.

# Task 3: Forecast Future Market Trends
This repository contains the code and documentation for Task 3 of the financial data analysis project. The goal of this task is to use the best-performing model from Task 2 (either SARIMA or LSTM) to forecast Tesla's (TSLA) future stock prices for the next 6-12 months. We will analyze the forecasted trends, visualize the results, and provide insights on potential market opportunities and risks.

## Table of Contents
- Objective

- Data Description

- Steps Performed

- Code Structure

- Outputs

- How to Run the Code

- Dependencies

- Key Insights

## Objective
- The objective of this task is to:

Use the trained SARIMA or LSTM model to forecast Tesla's stock prices for the next 6-12 months.

Visualize the forecasted prices alongside historical data.

Analyze the forecasted trends to identify potential market opportunities and risks.

## Data Description
The dataset includes historical stock price data for Tesla (TSLA) sourced from YFinance. The data covers the period from January 1, 2015, to January 31, 2025, and includes the following columns:

- Date: Trading day timestamp.

Open, High, Low, Close: Daily price metrics.

- Adj Close: Adjusted close price to account for dividends and splits.

- Volume: Total number of shares traded each day.

## Steps Performed
- Load the Trained Model:

Load the best-performing model (SARIMA or LSTM) from Task 2.

- Generate Future Forecasts:

Use the trained model to generate forecasts for the next 6-12 months.

- Visualize the Forecast:

Plot the forecasted prices alongside historical data.

Include confidence intervals (for SARIMA) to show the range of expected prices.

- Interpret the Results:

Analyze the forecasted trends to identify long-term trends (upward, downward, or stable).

Assess volatility and risk based on confidence intervals.

Highlight potential market opportunities and risks.

## Code Structure
The code is modular and organized into functions for better readability and reusability. Below is the structure of the code:

## Functions
- load_sarima_model:

Load the trained SARIMA model from a file.

- load_lstm_model:

Load the trained LSTM model from a file.

- sarima_forecast:

Generate future forecasts using the SARIMA model.

- lstm_forecast:

Generate future forecasts using the LSTM model.

- plot_forecast:

Visualize the forecasted prices alongside historical data.

- interpret_forecast:

Analyze the forecasted trends to identify market opportunities and risks.

## Outputs
- Forecast Plot:

A plot showing historical prices and forecasted prices for the next 6-12 months.

Confidence intervals (for SARIMA) to indicate the range of expected prices.

- Forecast Analysis:

Trend analysis (upward, downward, or stable).

Volatility and risk assessment based on confidence intervals.

Market opportunities and risks based on the forecasted trend.

## How to Run the Code
1. Clone the repository:

2. Install the required dependencies:

- pip install -r requirements.txt
3. Run the Python script:
- Follow the forecasting.ipynb

## Dependencies
The following Python libraries are required to run the code:

- yfinance

- pandas

- numpy

- matplotlib

- statsmodels

- pmdarima

- scikit-learn

- tensorflow

- joblib

## Trend Analysis:

- The forecast shows whether Tesla's stock prices are expected to increase (upward trend), decrease (downward trend), or remain stable.

1.  Volatility and Risk:

Confidence intervals (for SARIMA) indicate the level of uncertainty in the forecast.

Higher volatility suggests greater risk and potential for larger price swings.

2. Market Opportunities and Risks:

An upward trend suggests potential opportunities for investment.

A downward trend suggests potential risks and the need for caution.

# Task 4: Optimize Portfolio Based on Forecast

## Author
- Natnahom Asfaw
- 27/02/2025