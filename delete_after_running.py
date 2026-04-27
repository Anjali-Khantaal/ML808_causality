#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
financial_ts_causal.py

This script implements the financial time series data collection, preprocessing,
exploratory analysis, causal DAG construction, and Bayesian VAR modeling as described 
in the original Jupyter Notebook (see :contentReference[oaicite:0]{index=0}).

Usage:
    python financial_ts_causal.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pymc as pm
import arviz as az

import datetime

# ------------------------------------------------------------------------------
# SECTION 1: DATA COLLECTION, CLEANING, AND PREPROCESSING
# ------------------------------------------------------------------------------

def data_collection_preprocessing():
    # -------------------------------
    # 1. Data Collection
    # -------------------------------
    # Define the date range for the data: several decades to capture various regimes.
    start_date = "1950-01-01"
    end_date   = "2025-03-31"

    # -- A. Federal Funds Rate from FRED -------------------
    fedfunds_df = pdr.DataReader("FEDFUNDS", "fred", start_date, end_date)
    # Convert from a 1-column DataFrame into a Series.
    fedfunds_s = fedfunds_df['FEDFUNDS'].copy()

    # Shift index to month-end dates (the original index is often the 1st of the month)
    fedfunds_s.index = fedfunds_s.index + pd.offsets.MonthEnd(0)

    # -- B. S&P 500 from yfinance (daily data) -------------
    sp500_df = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    # Ensure we have the 'Close' column
    if 'Close' not in sp500_df.columns:
        raise ValueError("Could not find a 'Close' column in the downloaded S&P 500 DataFrame!")
    
    # -------------------------------
    # 2. Data Cleaning and Resampling
    # -------------------------------
    # Keep only the "Close" column for the S&P 500
    sp500_close_df = sp500_df[['Close']]
    # Resample the data to month-end frequency and rename the column to "SP500"
    sp500_close_df = sp500_close_df.resample('ME').last()
    sp500_close_df.columns = ['SP500']

    # -------------------------------
    # 3. Merge and Align the Time Series Data
    # -------------------------------
    data = pd.concat([fedfunds_s, sp500_close_df['SP500']], axis=1)
    data.columns = ['FedFundsRate', 'SP500']

    # Sort and restrict the date range to when both series are available (drop initial months with missing S&P500)
    data = data.sort_index()
    data = data.loc['1957':]

    # Forward-fill missing values (if any small gaps exist)
    data = data.fillna(method='ffill')
    # Remove any remaining NaNs
    data.dropna(inplace=True)

    print("Data shape after merging and cleaning:", data.shape)
    print("Preview of merged data:")
    print(data.head())

    # -------------------------------
    # 4. Feature Engineering and Stationarity Transformations
    # -------------------------------
    # Create stationarity-inducing transformations:
    # - S&P 500 Log Returns
    # - Fed Funds Rate differences (first difference)
    data['SP500_log_return'] = np.log(data['SP500'] / data['SP500'].shift(1))
    data['FedFunds_diff'] = data['FedFundsRate'] - data['FedFundsRate'].shift(1)
    # Drop the first row (which becomes NaN due to shift)
    data.dropna(inplace=True)
    
    print("Transformed features (first five rows):")
    print(data[['SP500_log_return', 'FedFunds_diff']].head())
    print("Data shape after transformation:", data.shape)

    # -------------------------------
    # 5. Lagged Features for Modeling
    # -------------------------------
    # Add lagged features (1-month and 2-month lags) for temporal dependence.
    data['SP500_log_return_lag1'] = data['SP500_log_return'].shift(1)
    data['FedFunds_diff_lag1'] = data['FedFunds_diff'].shift(1)

    data['SP500_log_return_lag2'] = data['SP500_log_return'].shift(2)
    data['FedFunds_diff_lag2'] = data['FedFunds_diff'].shift(2)

    # Drop rows that contain NaN values introduced by lagging
    data.dropna(inplace=True)
    print("Data shape after adding lagged features:", data.shape)
    print("Preview of lagged features:")
    print(data[['SP500_log_return', 'SP500_log_return_lag1', 'FedFunds_diff', 'FedFunds_diff_lag1']].head())

    # -------------------------------
    # 6. Save the Preprocessed Data
    # -------------------------------
    csv_filename = "financial_timeseries_preprocessed.csv"
    data.to_csv(csv_filename, index=True)
    print("\nPreprocessed data saved to:", csv_filename)
    print("First 5 rows of the saved file:")
    print(data.head(5))
    print("Last 5 rows of the saved file:")
    print(data.tail(5))

    return data

# ------------------------------------------------------------------------------
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA) & CAUSAL DAG CONSTRUCTION
# ------------------------------------------------------------------------------

def exploratory_analysis():
    # Load the preprocessed data
    csv_filename = "financial_timeseries_preprocessed.csv"
    data = pd.read_csv(csv_filename, index_col=0, parse_dates=True)
    
    print("\n--- Exploratory Analysis ---")
    print("Data Head:")
    print(data.head())

    # 3.1 Time Series Plots (Raw Data)
    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data['FedFundsRate'], label='Fed Funds Rate')
    plt.title("Fed Funds Rate Over Time")
    plt.xlabel("Date")
    plt.ylabel("Fed Funds Rate (%)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data['SP500'], label='S&P 500', color='orange')
    plt.title("S&P 500 Over Time")
    plt.xlabel("Date")
    plt.ylabel("S&P 500")
    plt.legend()
    plt.show()

    # 3.2 Plots of Transformed (Stationary) Series
    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data['SP500_log_return'], label='S&P 500 Log Returns', color='green')
    plt.title("S&P 500 Log Returns Over Time")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data['FedFunds_diff'], label='Fed Funds Rate Difference', color='red')
    plt.title("Fed Funds Rate Difference Over Time")
    plt.xlabel("Date")
    plt.ylabel("Difference")
    plt.legend()
    plt.show()

    # 3.3 Descriptive Statistics and Correlation Analysis
    print("\nDescriptive Statistics:")
    print(data[['FedFundsRate', 'SP500', 'SP500_log_return', 'FedFunds_diff']].describe())

    print("\nCorrelation Matrix:")
    corr_matrix = data[['FedFundsRate', 'SP500', 'SP500_log_return', 'FedFunds_diff']].corr()
    print(corr_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix Heatmap")
    plt.show()

    # 3.4 Stationarity Testing using the ADF Test
    def adf_test(series, series_name):
        result = adfuller(series.dropna())
        print(f"\nADF Test for {series_name}:")
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        print("Critical Values:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.4f}")
        if result[1] < 0.05:
            print(f"Conclusion: {series_name} is likely stationary.")
        else:
            print(f"Conclusion: {series_name} is likely non-stationary.")

    adf_test(data['SP500_log_return'], "S&P 500 Log Returns")
    adf_test(data['FedFunds_diff'], "Fed Funds Rate Difference")

    # 4. Causal DAG Construction
    # Hypothesis: The interest rate influences asset prices.
    G = nx.DiGraph()
    G.add_node("Interest_Rate")  # represents 'FedFundsRate'
    G.add_node("Asset_Price")    # represents 'SP500'
    G.add_edge("Interest_Rate", "Asset_Price")

    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue",
            arrows=True, arrowstyle='-|>', arrowsize=20, node_size=2000, font_size=12)
    plt.title("Causal DAG: Interest Rate → Asset Price")
    plt.show()

    print("\nCausal DAG constructed. The hypothesis is that the interest rate has a direct influence on asset prices.")

# ------------------------------------------------------------------------------
# SECTION 3: BAYESIAN VAR MODEL (STEP 4-A)
# ------------------------------------------------------------------------------

def bayesian_var_model():
    # Load the preprocessed data (make sure the CSV exists from the previous step)
    csv_filename = "financial_timeseries_preprocessed.csv"
    data = pd.read_csv(csv_filename, index_col=0, parse_dates=True)

    print("\n--- Bayesian VAR Model ---")

    # Validate that all required columns exist
    required_columns = [
        'FedFunds_diff', 'SP500_log_return',
        'FedFunds_diff_lag1', 'FedFunds_diff_lag2',
        'SP500_log_return_lag1', 'SP500_log_return_lag2'
    ]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Define the observed time series for the VAR model.
    # y1: FedFunds_diff (interest rate) | y2: SP500_log_return (asset price)
    Y = data[['FedFunds_diff', 'SP500_log_return']].values  # Shape (T, 2)

    # Extract the lagged predictor arrays.
    ff_lag1    = data['FedFunds_diff_lag1'].values
    ff_lag2    = data['FedFunds_diff_lag2'].values
    sp500_lag1 = data['SP500_log_return_lag1'].values
    sp500_lag2 = data['SP500_log_return_lag2'].values

    T = Y.shape[0]
    print("Number of observations available for VAR:", T)
    num_lags = 2

    # Build the Bayesian VAR model using PyMC.
    with pm.Model() as var_model:

        # 7.1 Intercepts for both equations (2-dimensional vector)
        intercept = pm.Normal('intercept', mu=0, sigma=10, shape=2)

        # 7.2 Coefficients for equation 1 (y1: FedFunds_diff)
        # Only own lagged values are allowed.
        a_ff = pm.Normal('a_ff', mu=0, sigma=1, shape=num_lags)

        # 7.3 Coefficients for equation 2 (y2: SP500_log_return)
        # Both the lagged FedFunds_diff and the lagged SP500_log_return are allowed.
        a_ff_effect = pm.Normal('a_ff_effect', mu=0, sigma=1, shape=num_lags)
        a_sp_effect = pm.Normal('a_sp_effect', mu=0, sigma=1, shape=num_lags)

        # 7.4 Priors for error standard deviations.
        sigma = pm.HalfCauchy('sigma', beta=2.5, shape=2)

        # 7.5 Construct the model equations.
        # Equation for y1 (FedFunds_diff): uses only its own lags.
        mu_1 = intercept[0] + a_ff[0] * ff_lag1 + a_ff[1] * ff_lag2

        # Equation for y2 (SP500_log_return): uses both lagged FedFunds_diff and lagged SP500_log_return.
        mu_2 = (intercept[1] + 
                a_ff_effect[0] * ff_lag1 +
                a_sp_effect[0] * sp500_lag1 +
                a_ff_effect[1] * ff_lag2 +
                a_sp_effect[1] * sp500_lag2)

        # Stack the two equations so that mu has shape (T, 2)
        mu = pm.math.stack([mu_1, mu_2], axis=1)

        # 7.6 Likelihood definition.
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)

        # 7.7 Inference: Run the MCMC sampler (NUTS).
        print("Sampling from the Bayesian VAR model...")
        trace = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=True)

    # 8. Diagnostics: Display posterior summaries and trace plots.
    print("\nPosterior Summary:")
    print(az.summary(trace, var_names=['intercept', 'a_ff', 'a_ff_effect', 'a_sp_effect', 'sigma']))
    az.plot_trace(trace, var_names=['intercept', 'a_ff', 'a_ff_effect', 'a_sp_effect', 'sigma'])
    plt.show()

# ------------------------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------------------------

def main():
    print("=== Financial Time Series Causal Analysis ===")
    # Step 1: Data Collection and Preprocessing
    data_collection_preprocessing()
    # Step 2: Exploratory Analysis and Causal DAG
    exploratory_analysis()
    # Step 3: Bayesian VAR Model (Step 4-A)
    bayesian_var_model()
    print("=== End of Analysis ===")

if __name__ == "__main__":
    main()
