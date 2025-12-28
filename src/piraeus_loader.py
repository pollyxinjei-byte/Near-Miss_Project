# src/data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_vessel_data(df):
    """
    Preprocesses raw AIS data into Delta (relative displacement) sequences.
    This implements the "Delta-Prediction" strategy described in Chapter 4.3.2.
    """
    
    # 1. Resample to 1-minute intervals to normalize time gaps
    df_resampled = df[['lat', 'lon', 'sog']].resample('1T').mean()
    
    # 2. Circular mean for COG to handle the 0/360 degree boundary
    def circular_mean(angles):
        if len(angles) == 0: return np.nan
        rads = np.deg2rad(angles.dropna())
        return np.rad2deg(np.arctan2(np.mean(np.sin(rads)), np.mean(np.cos(rads)))) % 360
    
    df_cog = df[['cog']].resample('1T').apply(circular_mean)
    df_vessel = pd.concat([df_resampled, df_cog], axis=1).interpolate().dropna()

    # 3. CALCULATE DELTAS (The key to 7.98m accuracy)
    # This transforms absolute coordinates into movement vectors
    df_vessel['delta_lat'] = df_vessel['lat'].diff()
    df_vessel['delta_lon'] = df_vessel['lon'].diff()
    df_vessel = df_vessel.dropna()

    # 4. Feature Selection & Scaling
    # Features: [Delta_Lat, Delta_Lon, SOG, COG]
    feature_cols = ['delta_lat', 'delta_lon', 'sog', 'cog']
    data_values = df_vessel[feature_cols].values
    original_coords = df_vessel[['lat', 'lon']].values  # Saved for reconstruction

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data_values)

    return data_scaled, original_coords, scaler


def create_sequences(data, orig_pos, seq_len=10):
    """
    Creates overlapping windows for LSTM training.
    
    Args:
        data: Scaled feature array
        orig_pos: Original coordinates for position reconstruction
        seq_len: Sequence length (default=10, ~10 minutes of context)
    
    Returns:
        xs: Input sequences (batch, seq_len, features)
        ys: Target deltas (batch, 2)
        pos_targets: Ground truth absolute positions
    """
    xs, ys, pos_targets = [], [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i : i+seq_len])           # Input sequence
        ys.append(data[i+seq_len][:2])           # Target: [Delta_Lat, Delta_Lon]
        pos_targets.append(orig_pos[i+seq_len])  # True Absolute Position
    return np.array(xs), np.array(ys), np.array(pos_targets)