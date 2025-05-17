import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

# Load CSV file (make sure 'Date' is a column and set as index)
df = pd.read_csv('./Stock Market Data Analysis/stock_data.csv',
                 parse_dates=['Date'], index_col='Date')

# Ensure the data is sorted
df = df.sort_index()

# ================================
# Calculations
# ================================

# Moving Averages
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()

# Daily Returns
df['Daily Return'] = df['Close'].pct_change()

# Volatility (Rolling Std Dev of Daily Returns)
df['Volatility'] = df['Daily Return'].rolling(window=20).std()

# Bollinger Bands
df['20_MA'] = df['Close'].rolling(window=20).mean()
df['20_SD'] = df['Close'].rolling(window=20).std()
df['Upper_Band'] = df['20_MA'] + (2 * df['20_SD'])
df['Lower_Band'] = df['20_MA'] - (2 * df['20_SD'])

# RSI (Relative Strength Index)
df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

# Drop initial NaN values for clean plotting
df = df.dropna()

# ================================
# Plots
# ================================

# Plot 1: Close Price with Moving Averages
plt.subplot(3, 1, 2)
plt.plot(df['Close'], label='Close Price', linewidth=1)
plt.plot(df['MA_7'], label='7-Day MA', linestyle='--')
plt.plot(df['MA_30'], label='30-Day MA', linestyle='--')
plt.title('Stock Price with Moving Averages')
plt.legend()
plt.show()


# Plot 2: Bollinger Bands
plt.subplot(3, 1, 2)
plt.plot(df['Close'], label='Close Price')
plt.plot(df['Upper_Band'], label='Upper Band', color='r', linestyle='--')
plt.plot(df['Lower_Band'], label='Lower Band', color='g', linestyle='--')
plt.fill_between(df.index, df['Lower_Band'],
                 df['Upper_Band'], color='grey', alpha=0.1)
plt.title('Bollinger Bands')
plt.legend()
plt.show()

# Plot 3: RSI and Volatility
plt.subplot(3, 1, 2)
plt.plot(df['RSI'], label='RSI', color='purple')
plt.axhline(70, linestyle='--', color='red', alpha=0.5)
plt.axhline(30, linestyle='--', color='green', alpha=0.5)
plt.title('Relative Strength Index (RSI)')
plt.legend()
plt.show()

plt.close()
