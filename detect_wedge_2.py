import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from scipy.stats import linregress
import os

plt.style.use('seaborn-darkgrid')

# Function Definitions (在这之前放置您已经定义的pivot_id, pivot_point_position等函数)

from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates
import matplotlib.pyplot as plt 

import numpy as np
import os
import pandas as pd
from scipy.stats import linregress

plt.style.use('seaborn-darkgrid')

def pivot_id(ohlc, l, n1, n2):
    """
    Get the pivot id 

    :params ohlc is a dataframe
    :params l is the l'th row
    :params n1 is the number of candles to the left
    :params n2 is the number of candles to the right
    :return int  
    """

    # Check if the length conditions met
    if l-n1 < 0 or l+n2 >= len(ohlc):
        return 0
    
    pivot_low  = 1
    pivot_high = 1

    for i in range(l-n1, l+n2+1):
        if(ohlc.loc[l,"Close"] > ohlc.loc[i, "Close"]):
            pivot_low = 0

        if(ohlc.loc[l, "Close"] < ohlc.loc[i, "Close"]):
            pivot_high = 0

    if pivot_low and pivot_high:
        return 3

    elif pivot_low:
        return 1

    elif pivot_high:
        return 2
    else:
        return 0


def pivot_point_position(row):
    """
    Get the Pivot Point position and assign a Close value

    :params row -> row of the ohlc dataframe
    :return float
    """
   
    if row['Pivot']==1:
        return row['Close']-1e-3
    elif row['Pivot']==2:
        return row['Close']+1e-3
    else:
        return np.nan


def find_wedge_points(ohlc, back_candles):
    """
    Find wedge points

    :params ohlc         -> dataframe that has OHLC data
    :params back_candles -> number of periods to lookback
    :return all_points
    """
    all_points = []
    for candle_idx in range(back_candles+10, len(ohlc)):

        maxim = np.array([])
        minim = np.array([])
        xxmin = np.array([])
        xxmax = np.array([])

        for i in range(candle_idx-back_candles, candle_idx+1):
            if ohlc.loc[i,"Pivot"] == 1:
                minim = np.append(minim, ohlc.loc[i, "Close"])
                xxmin = np.append(xxmin, i) 
            if ohlc.loc[i,"Pivot"] == 2:
                maxim = np.append(maxim, ohlc.loc[i,"Close"])
                xxmax = np.append(xxmax, i)

        
        if (xxmax.size <3 and xxmin.size <3) or xxmax.size==0 or xxmin.size==0:
            continue

        slmin, intercmin, rmin, pmin, semin = linregress(xxmin, minim)
        slmax, intercmax, rmax, pmax, semax = linregress(xxmax, maxim)
        

        # Check if the lines are in the same direction
        if abs(rmax)>=0.9 and abs(rmin)>=0.9 and ((slmin>=1e-3 and slmax>=1e-3 ) or (slmin<=-1e-3 and slmax<=-1e-3)):
                # Check if lines are parallel but converge fast 
                x_ =   (intercmin -intercmax)/(slmax-slmin)
                cors = np.hstack([xxmax, xxmin])  
                if (x_ - max(cors))>0 and (x_ - max(cors))<(max(cors) - min(cors))*3 and slmin/slmax > 0.75 and slmin/slmax < 1.25:  
                     all_points.append(candle_idx)
            

    return all_points


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
def fetch_data(symbol, timeframe='1h', limit=500):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['timestamp'] = df['timestamp'].map(mdates.date2num)
    return df

# Function to calculate win rate based on Wedge patterns
def calculate_win_rate(ohlc, all_points, look_forward=10, risk_reward_ratio=2):
    win_count = 0
    for point in all_points:
        entry_price = ohlc['Close'][point]
        stop_loss = ohlc['Low'].iloc[point:point+look_forward].min()
        take_profit = entry_price + risk_reward_ratio * (entry_price - stop_loss)
        forward_prices = ohlc['High'].iloc[point+1:point+look_forward+1]
        if any(forward_prices >= take_profit):
            win_count += 1
    return win_count / len(all_points) if all_points else 0

# Example for fetching data, detecting wedge patterns, calculating win rate
symbol = 'BTC/USDT'  # Replace with your desired symbol
df = fetch_data(symbol)
ohlc = df.loc[:, ["timestamp", "Open", "High", "Low", "Close"]]
ohlc.rename(columns={'timestamp': 'Date'}, inplace=True)

ohlc["Pivot"] = ohlc.apply(lambda row: pivot_id(ohlc, row.name, 3, 3), axis=1)
ohlc['PointPos'] = ohlc.apply(lambda row: pivot_point_position(row), axis=1)

# Find all wedge pattern points
back_candles = 20
all_points = find_wedge_points(ohlc, back_candles)

# Calculate win rate
win_rate = calculate_win_rate(ohlc, all_points)

# Save the plot with detected wedge points
dir_ = os.getcwd()  # or any directory you want to save to
save_plot(ohlc, all_points, back_candles)

# Print out a confirmation message with the calculated win rate
print(f'Detected wedge patterns for {symbol} have been plotted and saved as images. Win rate: {win_rate:.2%}')