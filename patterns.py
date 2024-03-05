import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import pywt


def detect_head_shoulder(df, window=3):
# Define the rolling window
    roll_window = window
    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    # Create a boolean mask for Head and Shoulder pattern
    mask_head_shoulder = ((df['high_roll_max'] > df['High'].shift(1)) & (df['high_roll_max'] > df['High'].shift(-1)) & (df['High'] < df['High'].shift(1)) & (df['High'] < df['High'].shift(-1)))
    # Create a boolean mask for Inverse Head and Shoulder pattern
    mask_inv_head_shoulder = ((df['low_roll_min'] < df['Low'].shift(1)) & (df['low_roll_min'] < df['Low'].shift(-1)) & (df['Low'] > df['Low'].shift(1)) & (df['Low'] > df['Low'].shift(-1)))
    # Create a new column for Head and Shoulder and its inverse pattern and populate it using the boolean masks
    df['head_shoulder_pattern'] = np.nan
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    return df 
    # return not df['head_shoulder_pattern'].isna().any().item()

def detect_multiple_tops_bottoms(df, window=3):
# Define the rolling window
    roll_window = window
    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['close_roll_max'] = df['Close'].rolling(window=roll_window).max()
    df['close_roll_min'] = df['Close'].rolling(window=roll_window).min()
    # Create a boolean mask for multiple top pattern
    mask_top = (df['high_roll_max'] >= df['High'].shift(1)) & (df['close_roll_max'] < df['Close'].shift(1))
    # Create a boolean mask for multiple bottom pattern
    mask_bottom = (df['low_roll_min'] <= df['Low'].shift(1)) & (df['close_roll_min'] > df['Close'].shift(1))
    # Create a new column for multiple top bottom pattern and populate it using the boolean masks
    df['multiple_top_bottom_pattern'] = np.nan
    df.loc[mask_top, 'multiple_top_bottom_pattern'] = 'Multiple Top'
    df.loc[mask_bottom, 'multiple_top_bottom_pattern'] = 'Multiple Bottom'
    return df

def calculate_support_resistance(df, window=3):
# Define the rolling window
    roll_window = window
    # Set the number of standard deviation
    std_dev = 2
    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    # Calculate the mean and standard deviation for High and Low
    mean_high = df['High'].rolling(window=roll_window).mean()
    std_high = df['High'].rolling(window=roll_window).std()
    mean_low = df['Low'].rolling(window=roll_window).mean()
    std_low = df['Low'].rolling(window=roll_window).std()
    # Create a new column for support and resistance
    df['support'] = mean_low - std_dev * std_low
    df['resistance'] = mean_high + std_dev * std_high
    return df

def detect_triangle_pattern(df, window=3):
    # Define the rolling window
    roll_window = window
    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    # Create a boolean mask for ascending triangle pattern
    mask_asc = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['Close'] > df['Close'].shift(1))
    # Create a boolean mask for descending triangle pattern
    mask_desc = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['Close'] < df['Close'].shift(1))
    # Create a new column for triangle pattern and populate it using the boolean masks
    df['triangle_pattern'] = np.nan
    df.loc[mask_asc, 'triangle_pattern'] = 'Ascending Triangle'
    df.loc[mask_desc, 'triangle_pattern'] = 'Descending Triangle'
    return df

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
def detect_channel(df, window=3):
    # Define the rolling window
    roll_window = window
    # Define a factor to check for the range of channel
    channel_range = 0.1
    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    # Create a boolean mask for Channel Up pattern
    mask_channel_up = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['high_roll_max'] - df['low_roll_min'] <= channel_range * (df['high_roll_max'] + df['low_roll_min'])/2) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    # Create a boolean mask for Channel Down pattern
    mask_channel_down = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['high_roll_max'] - df['low_roll_min'] <= channel_range * (df['high_roll_max'] + df['low_roll_min'])/2) & (df['trend_high'] == -1) & (df['trend_low'] == -1)
    # Create a new column for Channel Up and Channel Down pattern and populate it using the boolean masks
    df['channel_pattern'] = np.nan
    df.loc[mask_channel_up, 'channel_pattern'] = 'Channel Up'
    df.loc[mask_channel_down, 'channel_pattern'] = 'Channel Down'
    return df

def detect_double_top_bottom(df, window=3, threshold=0.05):
    # Define the rolling window
    roll_window = window
    # Define a threshold to check for the range of pattern
    range_threshold = threshold

    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()

    # Create a boolean mask for Double Top pattern
    mask_double_top = (df['high_roll_max'] >= df['High'].shift(1)) & (df['high_roll_max'] >= df['High'].shift(-1)) & (df['High'] < df['High'].shift(1)) & (df['High'] < df['High'].shift(-1)) & ((df['High'].shift(1) - df['Low'].shift(1)) <= range_threshold * (df['High'].shift(1) + df['Low'].shift(1))/2) & ((df['High'].shift(-1) - df['Low'].shift(-1)) <= range_threshold * (df['High'].shift(-1) + df['Low'].shift(-1))/2)
    # Create a boolean mask for Double Bottom pattern
    mask_double_bottom = (df['low_roll_min'] <= df['Low'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(-1)) & (df['Low'] > df['Low'].shift(1)) & (df['Low'] > df['Low'].shift(-1)) & ((df['High'].shift(1) - df['Low'].shift(1)) <= range_threshold * (df['High'].shift(1) + df['Low'].shift(1))/2) & ((df['High'].shift(-1) - df['Low'].shift(-1)) <= range_threshold * (df['High'].shift(-1) + df['Low'].shift(-1))/2)

    # Create a new column for Double Top and Double Bottom pattern and populate it using the boolean masks
    df['double_pattern'] = np.nan
    df.loc[mask_double_top, 'double_pattern'] = 'Double Top'
    df.loc[mask_double_bottom, 'double_pattern'] = 'Double Bottom'

    return df

def detect_trendline(df, window=2):
    # Define the rolling window
    roll_window = window
    # Create new columns for the linear regression slope and y-intercept
    df['slope'] = np.nan
    df['intercept'] = np.nan

    for i in range(window, len(df)):
        x = np.array(range(i-window, i))
        y = df['Close'][i-window:i]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        df.at[df.index[i], 'slope'] = m
        df.at[df.index[i], 'intercept'] = c

    # Create a boolean mask for trendline support
    mask_support = df['slope'] > 0

    # Create a boolean mask for trendline resistance
    mask_resistance = df['slope'] < 0

    # Create new columns for trendline support and resistance
    df['support'] = np.nan
    df['resistance'] = np.nan

    # Populate the new columns using the boolean masks
    df.loc[mask_support, 'support'] = df['Close'] * df['slope'] + df['intercept']
    df.loc[mask_resistance, 'resistance'] = df['Close'] * df['slope'] + df['intercept']

    return df

def find_pivots(df):
    # Calculate differences between consecutive highs and lows
    high_diffs = df['high'].diff()
    low_diffs = df['low'].diff()

    # Find higher high
    higher_high_mask = (high_diffs > 0) & (high_diffs.shift(-1) < 0)
    
    # Find lower low
    lower_low_mask = (low_diffs < 0) & (low_diffs.shift(-1) > 0)

    # Find lower high
    lower_high_mask = (high_diffs < 0) & (high_diffs.shift(-1) > 0)

    # Find higher low
    higher_low_mask = (low_diffs > 0) & (low_diffs.shift(-1) < 0)

    # Create signals column
    df['signal'] = ''
    df.loc[higher_high_mask, 'signal'] = 'HH'
    df.loc[lower_low_mask, 'signal'] = 'LL'
    df.loc[lower_high_mask, 'signal'] = 'LH'
    df.loc[higher_low_mask, 'signal'] = 'HL'

    return df



'''
Algorithm 1: Basic Head-Shoulder Detection
This algorithm implements a basic detection of the "Head and Shoulder" and "Inverse Head and Shoulder" patterns 
in a data frame. These patterns are significant in financial analysis as they indicate possible market reversals. 
In the code, the algorithm uses a rolling window to track high and low points in the 'High' and 'Low' columns of 
the data frame. It then creates boolean masks to identify where these patterns occur. The identified patterns are 
then added to the data frame in a new 'head_shoulder_pattern' column.
'''

def detect_head_shoulder(df, window=3):
# Define the rolling window
    roll_window = window
    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    # Create a boolean mask for Head and Shoulder pattern
    mask_head_shoulder = ((df['high_roll_max'] > df['High'].shift(1)) & (df['high_roll_max'] > df['High'].shift(-1)) & (df['High'] < df['High'].shift(1)) & (df['High'] < df['High'].shift(-1)))
    # Create a boolean mask for Inverse Head and Shoulder pattern
    mask_inv_head_shoulder = ((df['low_roll_min'] < df['Low'].shift(1)) & (df['low_roll_min'] < df['Low'].shift(-1)) & (df['Low'] > df['Low'].shift(1)) & (df['Low'] > df['Low'].shift(-1)))
    # Create a new column for Head and Shoulder and its inverse pattern and populate it using the boolean masks
    df['head_shoulder_pattern'] = np.nan
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    return df 

'''
Algorithm 2: Head-Shoulder Detection with Savitzky-Golay Filter

This algorithm is an improvement of the first one. It first applies the Savitzky-Golay filter to smooth the 'High' and 'Low' columns. 
This filter is used to reduce noise and improve the reliability of pattern detection. 
In addition to the head-shoulder pattern detection of the first algorithm, this algorithm also considers 
the height of the "Head" or "Inverse Head" and introduces a threshold to avoid false pattern 
recognition due to insignificant price changes.
'''
def detect_head_shoulder_filter(df, window=3, threshold=0.01, time_delay=1):
    roll_window = window
    df['High_smooth'] = savgol_filter(df['High'], roll_window, 2) # Apply Savitzky-Golay filter
    df['Low_smooth'] = savgol_filter(df['Low'], roll_window, 2)
    
    df['high_roll_max'] = df['High_smooth'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low_smooth'].rolling(window=roll_window).min()
    
    # Define the height of the head and inverse head
    df['head_height'] = df['high_roll_max'] - df['Low'].rolling(window=roll_window).min()
    df['inv_head_height'] = df['High'].rolling(window=roll_window).max() - df['low_roll_min']
    
    # Define the masks for head and shoulder and inverse head and shoulder
    mask_head_shoulder = ((df['head_height'] > threshold) & (df['high_roll_max'] > df['High_smooth'].shift(time_delay)) & (df['high_roll_max'] > df['High_smooth'].shift(-time_delay)) & (df['High_smooth'] < df['High_smooth'].shift(time_delay)) & (df['High_smooth'] < df['High_smooth'].shift(-time_delay)))
    mask_inv_head_shoulder = ((df['inv_head_height'] > threshold) & (df['low_roll_min'] < df['Low_smooth'].shift(time_delay)) & (df['low_roll_min'] < df['Low_smooth'].shift(-time_delay)) & (df['Low_smooth'] > df['Low_smooth'].shift(time_delay)) & (df['Low_smooth'] > df['Low_smooth'].shift(-time_delay)))
    
    df['head_shoulder_pattern'] = np.nan
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    
    return df

'''
Algorithm 3: Head-Shoulder Detection with Kalman Filter

In this algorithm, the Kalman Filter is used to smooth the 'High' and 'Low' columns. 
The Kalman Filter is a recursive filter that estimates the state of a system in real time, 
making it more suitable for financial data with its inherent noise and uncertainties. 
This can potentially improve the accuracy of pattern detection. 
The pattern detection process is similar to Algorithm 1, but it operates on the smoothed data.
'''


def kalman_smooth(series, n_iter=10):
    # Initialize Kalman filter
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

    # Use the EM algorithm to estimate the best values for the parameters
    kf = kf.em(series, n_iter=n_iter)

    # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(series.values)

    return state_means.flatten()


def detect_head_shoulder_kf(df, window=3):
    roll_window = window
    df['High_smooth'] = kalman_smooth(df['High'])
    df['Low_smooth'] = kalman_smooth(df['Low'])
    
    df['high_roll_max'] = df['High_smooth'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low_smooth'].rolling(window=roll_window).min()
    
    mask_head_shoulder = ((df['high_roll_max'] > df['High_smooth'].shift(1)) & (df['high_roll_max'] > df['High_smooth'].shift(-1)) & (df['High_smooth'] < df['High_smooth'].shift(1)) & (df['High_smooth'] < df['High_smooth'].shift(-1)))
    mask_inv_head_shoulder = ((df['low_roll_min'] < df['Low_smooth'].shift(1)) & (df['low_roll_min'] < df['Low_smooth'].shift(-1)) & (df['Low_smooth'] > df['Low_smooth'].shift(1)) & (df['Low_smooth'] > df['Low_smooth'].shift(-1)))
    
    df['head_shoulder_pattern'] = np.nan
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    
    return df

'''
Algorithm 4: Head-Shoulder Detection with Wavelet Denoising

In this algorithm, wavelet denoising is applied to the 'High' and 'Low' columns before 
the pattern detection process. Wavelet denoising is an effective technique for eliminating 
noise while preserving the key features in the data. This can make the pattern detection process 
more robust and reliable, especially in the presence of high frequency noise in the data. 
'''

def wavelet_denoise(series, wavelet='db1', level=1):
    # Perform wavelet decomposition
    coeff = pywt.wavedec(series, wavelet, mode="per")
    # Set detail coefficients for levels > level to zero
    for i in range(1, len(coeff)):
        coeff[i] = pywt.threshold(coeff[i], value=np.std(coeff[i])/2, mode="soft")
    # Perform inverse wavelet transform
    return pywt.waverec(coeff, wavelet, mode="per")


def detect_head_shoulder_wavelet(df, window=3):
    roll_window = window
    df['High_smooth'] = wavelet_denoise(df['High'], 'db1', level=1)
    df['Low_smooth'] = wavelet_denoise(df['Low'], 'db1', level=1)
    
    df['high_roll_max'] = df['High_smooth'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low_smooth'].rolling(window=roll_window).min()
    
    mask_head_shoulder = ((df['high_roll_max'] > df['High_smooth'].shift(1)) & (df['high_roll_max'] > df['High_smooth'].shift(-1)) & (df['High_smooth'] < df['High_smooth'].shift(1)) & (df['High_smooth'] < df['High_smooth'].shift(-1)))
    mask_inv_head_shoulder = ((df['low_roll_min'] < df['Low_smooth'].shift(1)) & (df['low_roll_min'] < df['Low_smooth'].shift(-1)) & (df['Low_smooth'] > df['Low_smooth'].shift(1)) & (df['Low_smooth'] > df['Low_smooth'].shift(-1)))
    
    df['head_shoulder_pattern'] = np.nan
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    
    return df