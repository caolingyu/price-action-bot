import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mplfinance.original_flavor import candlestick_ohlc



# Connect to Binance using a proxy
exchange = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True,
    'proxies': {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890',
    },
})

# Function to fetch historical data for a symbol
def fetch_data(symbol, timeframe='1h', since=None, limit=500):
    # Fetch the candlestick data from Binance
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    # Create a DataFrame from the fetched data
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    # Convert timestamp to datetime for better readability
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # Set the timestamp as the index
    df.set_index('timestamp', inplace=True)
    return df

    
# Function to detect wedge patterns
def detect_wedge(df, window=3):
    # Define the rolling window
    roll_window = window
    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    # Create a boolean mask for Wedge Up pattern
    mask_wedge_up = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    # Create a boolean mask for Wedge Down pattern
        # Create a boolean mask for Wedge Down pattern
    mask_wedge_down = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['trend_high'] == -1) & (df['trend_low'] == -1)
    # Create a new column for Wedge Up and Wedge Down pattern and populate it using the boolean masks
    df['wedge_pattern'] = np.nan
    df.loc[mask_wedge_up, 'wedge_pattern'] = 'Wedge Up'
    df.loc[mask_wedge_down, 'wedge_pattern'] = 'Wedge Down'
    return df

def detect_wedge_and_calculate_win_rate(df, window=3, future_candles=10, risk_reward_ratio=2):
    # Your previous detect_wedge code goes here
    # Define the rolling window
    roll_window = window
    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    # Create a boolean mask for Wedge Up pattern
    mask_wedge_up = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    # Create a boolean mask for Wedge Down pattern
        # Create a boolean mask for Wedge Down pattern
    mask_wedge_down = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['trend_high'] == -1) & (df['trend_low'] == -1)
    # Create a new column for Wedge Up and Wedge Down pattern and populate it using the boolean masks
    df['wedge_pattern'] = np.nan
    df.loc[mask_wedge_up, 'wedge_pattern'] = 'Wedge Up'
    df.loc[mask_wedge_down, 'wedge_pattern'] = 'Wedge Down'
    # New calculation for win rate
    df['result'] = None
    
    # Loop through the DataFrame to check each trade's outcome based on its pattern
    for i in range(len(df) - (future_candles + 1)):
        if df['wedge_pattern'].iloc[i] == 'Wedge Up':
            entry_price = df['Open'].iloc[i + 1]
            max_high_price = df['High'].iloc[i + 1 : i + future_candles + 1].max()
            min_low_price = df['Low'].iloc[i + 1 : i + future_candles + 1].min()
            
            if (max_high_price - entry_price) >= 2 * (entry_price - min_low_price):
                df['result'].iloc[i] = 'Win'
            else:
                df['result'].iloc[i] = 'Loss'

        elif df['wedge_pattern'].iloc[i] == 'Wedge Down':
            entry_price = df['Open'].iloc[i + 1]
            max_high_price = df['High'].iloc[i + 1 : i + future_candles + 1].max()
            min_low_price = df['Low'].iloc[i + 1 : i + future_candles + 1].min()
            
            if (entry_price - min_low_price) >= 2 * (max_high_price - entry_price):
                df['result'].iloc[i] = 'Win'
            else:
                df['result'].iloc[i] = 'Loss'

    win_rate = df['result'].value_counts(normalize=True)['Win']
    return df, win_rate


# Function to plot wedge patterns on a chart
def plot_wedge(df, symbol):
    plot_df = df[-100:].copy()
    wedge_df = plot_df.dropna(subset=['wedge_pattern'])
    
    fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the figsize to make the plot larger

    # Convert timestamp to the format required by matplotlib
    dates = mdates.date2num(plot_df.index.to_pydatetime())

    # Prepare the data in the right format
    ohlc_data = [(dates[i], row[1], row[2], row[3], row[4]) for i, row in enumerate(plot_df.itertuples(index=False))]

    # Plot the candlestick chart
    candlestick_ohlc(ax, ohlc_data, width=0.02, colorup='green', colordown='red', alpha=0.8)

    # Plot Wedge Up and Wedge Down patterns
    for index, row in wedge_df.iterrows():
        date = mdates.date2num(index.to_pydatetime())
        if row['wedge_pattern'] == 'Wedge Up':
            ax.plot(date, row['High'], '^', color='blue', markersize=10)
        elif row['wedge_pattern'] == 'Wedge Down':
            ax.plot(date, row['Low'], 'v', color='orange', markersize=10)

    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # Set minor intervals to an hour
    ax.grid(True)
    
    # Rotate date labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Tight layout to use space efficiently
    plt.tight_layout()

    # Save the plot as a .png file
    plt.savefig(f'{symbol.replace("/", "_")}_wedge_pattern.png')
    plt.close(fig)

# Example for fetching data, detecting wedge patterns, calculating win rate, and plotting for a single symbol
symbol = 'BTC/USDT'
data = fetch_data(symbol)
wedge_data, wedge_win_rate = detect_wedge_and_calculate_win_rate(data)
plot_wedge(wedge_data, symbol)

# Print out a confirmation message with the calculated win rate
print(f'Detected wedge patterns for {symbol} have been plotted and saved as an image. Win rate: {wedge_win_rate:.2%}')