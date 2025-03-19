import pandas as pd
import pandas_datareader.data as web
import datetime

start = datetime.datetime(2015, 3, 1)
end = datetime.datetime(2025, 2, 28)

# -------------------------------
# 1. Retrieve S&P 500 Data (Daily)
# -------------------------------
# The FRED series "SP500" provides daily data.
sp500_daily = web.DataReader('SP500', 'fred', start, end)

# Resample to monthly data by taking the last available closing value of each month
sp500_monthly = sp500_daily.resample('ME').last()

# -------------------------------
# 2. Retrieve Bond and Interest Rate Data
# -------------------------------
# 10-Year Treasury Constant Maturity Rate (GS10) - monthly averages
gs10 = web.DataReader('GS10', 'fred', start, end)
gs10_monthly = gs10.resample('M').last()

# Effective Federal Funds Rate (FEDFUNDS) - provided as monthly averages
fedfunds = web.DataReader('FEDFUNDS', 'fred', start, end)
fedfunds_monthly = fedfunds.resample('M').last()

# -------------------------------
# 3. Retrieve CPI Data and Calculate Inflation
# -------------------------------
# CPI for All Urban Consumers (CPIAUCSL) - seasonally adjusted monthly data
cpi = web.DataReader('CPIAUCSL', 'fred', start, end)
cpi_monthly = cpi.resample('M').last()

# Convert CPI to annual inflation rate (year-over-year percentage change)
cpi_inflation = cpi_monthly.pct_change(12) * 100

# -------------------------------
# 4. Combine All Data into a Single DataFrame
# -------------------------------
# Concatenate the series along columns
data = pd.concat([ sp500_monthly,gs10_monthly,fedfunds_monthly,cpi_inflation], axis=1)
data.to_csv('./test.csv')

# Rename columns for clarity
data.columns = ['SP500', 'GS10', 'FEDFUNDS', 'Inflation']

# Total number of rows in the DataFrame
total_rows = len(data)

# Count the missing values per column
missing_counts = data.isna().sum()

# Count the non-missing values per column (optional)
non_missing_counts = data.count()

# Calculate the percentage of missing values per column
missing_percentage = (missing_counts / total_rows) * 100

# Create a summary DataFrame
summary = pd.DataFrame({
    'Total Rows': total_rows,
    'Non-Missing': non_missing_counts,
    'Missing': missing_counts,
    'Missing (%)': missing_percentage
})

print(summary)

print('========================================')

# Extract rows with any NaN values
missing_data = data[data.isna().any(axis=1)]
print("Rows with missing data:")
print(missing_data)
print('========================================')

for date, row in missing_data.iterrows():
    missing_cols = row[row.isna()].index.tolist()
    print(f"Month end {date.strftime('%Y-%m-%d')} is missing data for: {missing_cols}")

