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

## Author
- Natnahom Asfaw
- 27/02/2025