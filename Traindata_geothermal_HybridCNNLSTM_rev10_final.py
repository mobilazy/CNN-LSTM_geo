"""CNN-LSTM Model for Geothermal BHE Configuration Analysis

What's new:
- OE401 correction: subtract 8 research wells, divide by 112 production wells
- DST handling: timezone-aware parsing to deal with daylight savings

Three BHE types:
- 112x Single U45mm (production wells, corrected)
- 4x Double U45mm (research)
- 4x MuoviEllipse 63mm (research)

All at 300m depth with per-well values.
"""

from typing import List, Dict, Optional
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import json
import warnings
warnings.filterwarnings('ignore')

# Config
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model params
SEQ_LEN = 48
PRED_HORIZON = 1
BATCH_SIZE = 1024  # Maximum GPU utilization - can handle 1024 with only 0.7% VRAM usage
EPOCHS = 50
LR = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.25

# Time windows for evaluation (days)
FORECAST_WINDOW_DAYS = 21  # how far ahead we predict
VALIDATION_WINDOW_DAYS = 7  # validation hold-out period
TRAIN_HISTORY_WINDOW_DAYS = 21  # history to show in plots

FORECAST_WINDOW_HOURS = FORECAST_WINDOW_DAYS * 24
VALIDATION_WINDOW_HOURS = VALIDATION_WINDOW_DAYS * 24
TRAIN_HISTORY_WINDOW_HOURS = TRAIN_HISTORY_WINDOW_DAYS * 24

# Architecture
CONV_CHANNELS = [32, 64]
KERNEL_SIZE = 3
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
DROPOUT = 0.1
PATIENCE = 16

# Setup logging to file and console
log_file_path = os.path.join(OUTPUT_DIR, "comprehensive_analysis.log")

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handler
file_handler = logging.FileHandler(log_file_path, mode="w")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler],
    force=True
)

def load_complete_field_data():
    """Load complete field data with OE401 correction.
    
    OE401 measures 120 wells but 8 are research wells also in OE402/OE403.
    We subtract those 8 and divide by 112 production wells.
    
    Steps:
    1. Load all three sensors
    2. Handle DST timestamps  
    3. Subtract research wells from OE401
    4. Divide by 112 wells
    """
    
    INPUT_DIR = os.path.join(os.path.dirname(__file__), "input")
    
    logging.info("Loading data with OE401 correction...")
    
    # Load all three sensors
    oe401_path = os.path.join(INPUT_DIR, "MeterOE401_singleU45.csv")
    oe402_path = os.path.join(INPUT_DIR, "MeterOE402_Ellipse63.csv")
    oe403_path = os.path.join(INPUT_DIR, "MeterOE403_doubleU45.csv")
    
    if not os.path.exists(oe401_path):
        raise FileNotFoundError(f"Complete field data not found: {oe401_path}")
    if not os.path.exists(oe402_path):
        raise FileNotFoundError(f"MuoviEllipse data not found: {oe402_path}")
    if not os.path.exists(oe403_path):
        raise FileNotFoundError(f"Double U45mm data not found: {oe403_path}")
    
    try:
        # Load OE401 (120 wells including 8 research)
        logging.info("Loading OE401...")
        df401 = pd.read_csv(oe401_path, encoding='utf-8', sep=',', decimal='.')
        
        # Clean up column names
        df401.columns = df401.columns.str.strip()
        
        # Rename columns to match expected names
        df401 = df401.rename(columns={
            'Power [kW]': 'power_kw',
            'T_supply [°C]': 'supply_temp',
            'return [°C]': 'return_temp',
            'Flow [m³/h]': 'flow_rate'
        })
        
        # DST handling for OE401
        s = df401["Timestamp"].astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
        ts_parsed = pd.to_datetime(s, dayfirst=True, errors="coerce")
        df401["Timestamp"] = ts_parsed.dt.tz_localize(
            "Europe/Oslo",
            ambiguous=False,
            nonexistent="shift_forward"
        )
        df401 = df401.sort_values("Timestamp").drop_duplicates(subset=['Timestamp'], keep='first').reset_index(drop=True)
        
        for col in ['supply_temp', 'return_temp', 'power_kw', 'flow_rate']:
            if col in df401.columns:
                df401[col] = pd.to_numeric(df401[col], errors='coerce')
        
        # Load OE402 (MuoviEllipse, 4 wells)
        logging.info("Loading OE402...")
        df402 = pd.read_csv(oe402_path, encoding='utf-8', sep=',', decimal='.')
        
        # Remove leading/trailing spaces from column names first
        df402.columns = df402.columns.str.strip()
        
        # Rename columns to match expected names
        df402 = df402.rename(columns={
            'Power [kW]': 'power_kw',
            'T_supply [°C]': 'supply_temp',
            'T_return [°C]': 'return_temp',
            'Flow [m³/h]': 'flow_rate'
        })
        
        s = df402["Timestamp"].astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
        ts_parsed = pd.to_datetime(s, dayfirst=True, errors="coerce")
        df402["Timestamp"] = ts_parsed.dt.tz_localize(
            "Europe/Oslo",
            ambiguous=False,
            nonexistent="shift_forward"
        )
        df402 = df402.sort_values("Timestamp").drop_duplicates(subset=['Timestamp'], keep='first').reset_index(drop=True)
        
        for col in ['supply_temp', 'return_temp', 'power_kw', 'flow_rate']:
            if col in df402.columns:
                df402[col] = pd.to_numeric(df402[col], errors='coerce')
        
        # Divide by 4 wells
        df402['power_kw'] = df402['power_kw'] / 4
        df402['flow_rate'] = df402['flow_rate'] / 4
        
        # Load OE403 (Double U45mm, 4 wells)
        logging.info("Loading OE403...")
        df403 = pd.read_csv(oe403_path, encoding='utf-8', sep=',', decimal='.')
        
        # Remove leading/trailing spaces from column names first
        df403.columns = df403.columns.str.strip()
        
        # Rename columns to match expected names
        df403 = df403.rename(columns={
            'Power [kW]': 'power_kw',
            'T_supply [°C]': 'supply_temp',
            'T_return [°C]': 'return_temp',
            'Flow [m³/h]': 'flow_rate'
        })
        
        s = df403["Timestamp"].astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
        ts_parsed = pd.to_datetime(s, dayfirst=True, errors="coerce")
        df403["Timestamp"] = ts_parsed.dt.tz_localize(
            "Europe/Oslo",
            ambiguous=False,
            nonexistent="shift_forward"
        )
        df403 = df403.sort_values("Timestamp").drop_duplicates(subset=['Timestamp'], keep='first').reset_index(drop=True)
        
        for col in ['supply_temp', 'return_temp', 'power_kw', 'flow_rate']:
            if col in df403.columns:
                df403[col] = pd.to_numeric(df403[col], errors='coerce')
        
        # Divide by 4 wells
        df403['power_kw'] = df403['power_kw'] / 4
        df403['flow_rate'] = df403['flow_rate'] / 4
        
        # Find common timestamps
        common_timestamps = set(df401['Timestamp']).intersection(
            set(df402['Timestamp'])
        ).intersection(
            set(df403['Timestamp'])
        )
        
        logging.info(f"Common timestamps across all sensors: {len(common_timestamps):,}")
        
        # Align all datasets
        df401_aligned = df401[df401['Timestamp'].isin(common_timestamps)].sort_values('Timestamp').reset_index(drop=True)
        df402_aligned = df402[df402['Timestamp'].isin(common_timestamps)].sort_values('Timestamp').reset_index(drop=True)
        df403_aligned = df403[df403['Timestamp'].isin(common_timestamps)].sort_values('Timestamp').reset_index(drop=True)
        
        # Calculate research well contribution (per-well values × 4)
        research_power_total = (df402_aligned['power_kw'] * 4) + (df403_aligned['power_kw'] * 4)
        research_flow_total = (df402_aligned['flow_rate'] * 4) + (df403_aligned['flow_rate'] * 4)
        
        logging.info(f"Research wells total power: {research_power_total.min():.2f} to {research_power_total.max():.2f} kW")
        
        # Apply correction: subtract research wells and divide by 112
        df401_aligned['power_kw'] = (df401_aligned['power_kw'] - research_power_total) / 112
        df401_aligned['flow_rate'] = (df401_aligned['flow_rate'] - research_flow_total) / 112
        
        logging.info(f"Corrected OE401 per-well power: {df401_aligned['power_kw'].min():.3f} to {df401_aligned['power_kw'].max():.3f} kW")
        
        # Keep essential columns
        essential_cols = ['Timestamp', 'supply_temp', 'return_temp', 'flow_rate', 'power_kw']
        df = df401_aligned[essential_cols].copy()
        
        # Add BHE type
        df['bhe_type'] = 'single_u45mm'
        df['bhe_type_encoded'] = 0
        
        # Apply cleaning
        df = clean_bhe_data(df, "SingleU45mm_OE401_Corrected")
        
        # Calculate thermal metrics
        df = calculate_cumulative_energy(df, time_interval_minutes=5)
        
        logging.info(f"Loaded complete field data (corrected): {len(df)} records")
        return df
        
    except Exception as e:
        logging.error(f"Error loading complete field data: {e}")
        raise

def load_double_u45mm_research_data():
    """Load Double U45mm research wells from OE403."""
    
    oe403_path = os.path.join(os.path.dirname(__file__), "input/MeterOE403_doubleU45.csv")
    
    if not os.path.exists(oe403_path):
        logging.warning("Double U45mm research data file not found, skipping...")
        return pd.DataFrame()
    
    logging.info("Loading Double U45mm research wells data...")
    
    try:
        df = pd.read_csv(oe403_path, encoding='utf-8', sep=',', decimal='.')
        logging.info(f"Raw Double U45mm data loaded: {len(df)} records")
        
        # Remove leading/trailing spaces from column names first
        df.columns = df.columns.str.strip()
        
        # Rename columns to match expected names
        df = df.rename(columns={
            'Power [kW]': 'power_kw',
            'T_supply [°C]': 'supply_temp',
            'T_return [°C]': 'return_temp',
            'Flow [m³/h]': 'flow_rate'
        })
        
        # DST handling
        s = df["Timestamp"].astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
        ts_parsed = pd.to_datetime(s, dayfirst=True, errors="coerce")
        df["Timestamp"] = ts_parsed.dt.tz_localize(
            "Europe/Oslo",
            ambiguous=False,
            nonexistent="shift_forward"
        )
        df = df.sort_values("Timestamp").drop_duplicates(subset=['Timestamp'], keep='first').reset_index(drop=True)
        
        # Convert numeric columns
        for col in ['supply_temp', 'return_temp', 'power_kw', 'flow_rate']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Normalize by 4 wells
        df['power_kw'] = df['power_kw'] / 4
        df['flow_rate'] = df['flow_rate'] / 4
        
        # Keep essential columns
        essential_cols = ['Timestamp', 'supply_temp', 'return_temp', 'flow_rate', 'power_kw']
        df = df[essential_cols].copy()
        
        # Add BHE type
        df['bhe_type'] = 'double_u45mm'
        df['bhe_type_encoded'] = 1
        
        # Apply cleaning
        df = clean_bhe_data(df, "DoubleU45mm_OE403")
        
        # Calculate thermal metrics
        df = calculate_cumulative_energy(df, time_interval_minutes=5)
        
        logging.info(f"Processed Double U45mm data: {len(df)} records")
        return df
            
    except Exception as e:
        logging.error(f"Error loading Double U45mm research data: {e}")
        return pd.DataFrame()

def load_muovi_ellipse_research_data():
    """Load MuoviEllipse 63mm research wells from OE402."""
    
    csv_path = os.path.join(os.path.dirname(__file__), "input/MeterOE402_Ellipse63.csv")
    
    if not os.path.exists(csv_path):
        logging.warning("MuoviEllipse 63mm data file not found, skipping...")
        return pd.DataFrame()
    
    logging.info("Loading MuoviEllipse 63mm research data...")
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8', sep=',', decimal='.')
        logging.info(f"Raw MuoviEllipse data loaded: {len(df)} records")
        
        # Remove leading/trailing spaces from column names first
        df.columns = df.columns.str.strip()
        
        # Rename columns to match expected names
        df = df.rename(columns={
            'Power [kW]': 'power_kw',
            'T_supply [°C]': 'supply_temp',
            'T_return [°C]': 'return_temp',
            'Flow [m³/h]': 'flow_rate'
        })
        
        # DST handling
        s = df["Timestamp"].astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
        ts_parsed = pd.to_datetime(s, dayfirst=True, errors="coerce")
        df["Timestamp"] = ts_parsed.dt.tz_localize(
            "Europe/Oslo",
            ambiguous=False,
            nonexistent="shift_forward"
        )
        df = df.sort_values("Timestamp").drop_duplicates(subset=['Timestamp'], keep='first').reset_index(drop=True)
        
        # Convert numeric columns
        for col in ['supply_temp', 'return_temp', 'power_kw', 'flow_rate']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Divide by 4 wells
        df['power_kw'] = df['power_kw'] / 4
        df['flow_rate'] = df['flow_rate'] / 4
        
        # Keep essential columns
        essential_cols = ['Timestamp', 'supply_temp', 'return_temp', 'flow_rate', 'power_kw']
        df = df[essential_cols].copy()
        
        # Add BHE type
        df['bhe_type'] = 'muovi_ellipse_63mm'
        df['bhe_type_encoded'] = 2
        
        # Apply cleaning
        df = clean_bhe_data(df, "MuoviEllipse_OE402")
        
        # Calculate thermal metrics
        df = calculate_cumulative_energy(df, time_interval_minutes=5)
        
        logging.info(f"Processed MuoviEllipse 63mm data: {len(df)} records")
        return df
        
    except Exception as e:
        logging.error(f"Error loading MuoviEllipse 63mm data: {e}")
        return pd.DataFrame()

def create_muovi_ellipse_research_data():
    """Deprecated - use load_muovi_ellipse_research_data() instead."""
    
    logging.warning("Deprecated: use load_muovi_ellipse_research_data()")
    return load_muovi_ellipse_research_data()

def clean_bhe_data(df, dataset_name=""):
    """Clean sensor data using signal processing."""
    
    if len(df) == 0:
        return df

    logging.info(f"Applying data cleaning to {dataset_name}: {len(df)} initial records")

    df_clean = df.copy()
    
    # Sensor columns
    temp_cols = ['supply_temp', 'return_temp']
    power_cols = ['power_kw']
    flow_cols = ['flow_rate']
    
    # Convert to numeric (keep sensor precision)
    for col in temp_cols + power_cols + flow_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove physically unrealistic values
    for col in temp_cols:
        if col in df_clean.columns:
            # Temp range for geothermal: -10 to 50°C
            mask = (df_clean[col] >= -10) & (df_clean[col] <= 50)
            outliers_removed = (~mask).sum()
            df_clean.loc[~mask, col] = np.nan
            if outliers_removed > 0:
                logging.info(f"Removed {outliers_removed} temperature outliers from {col}")
    
    for col in power_cols:
        if col in df_clean.columns:
            # Power range for boreholes: -500 to 500kW
            mask = (df_clean[col] >= -500) & (df_clean[col] <= 500)
            outliers_removed = (~mask).sum()
            df_clean.loc[~mask, col] = np.nan
            if outliers_removed > 0:
                logging.info(f"Removed {outliers_removed} power outliers from {col}")
    
    for col in flow_cols:
        if col in df_clean.columns:
            # Flow should be positive and under 100 m³/h
            mask = (df_clean[col] > 0) & (df_clean[col] <= 100)
            outliers_removed = (~mask).sum()
            df_clean.loc[~mask, col] = np.nan
            if outliers_removed > 0:
                logging.info(f"Removed {outliers_removed} flow rate outliers from {col}")
    
    # Remove stuck sensor readings
    # Higher threshold for temps (60 = 5hrs) due to slow thermal dynamics
    for col in temp_cols:
        if col in df_clean.columns:
            consecutive_same = df_clean[col].groupby((df_clean[col] != df_clean[col].shift()).cumsum()).transform('size')
            stuck_mask = consecutive_same > 60
            stuck_removed = stuck_mask.sum()
            df_clean.loc[stuck_mask, col] = np.nan
            if stuck_removed > 0:
                logging.info(f"Removed {stuck_removed} stuck sensor readings from {col}")
    
    # Lower threshold for power/flow (18 = 90min) - faster response
    for col in power_cols + flow_cols:
        if col in df_clean.columns:
            consecutive_same = df_clean[col].groupby((df_clean[col] != df_clean[col].shift()).cumsum()).transform('size')
            stuck_mask = consecutive_same > 18
            stuck_removed = stuck_mask.sum()
            df_clean.loc[stuck_mask, col] = np.nan
            if stuck_removed > 0:
                logging.info(f"Removed {stuck_removed} stuck sensor readings from {col}")
    
    # Apply median filtering to reduce noise
    window_temp = 3  # 15min for temps
    window_other = 5  # 25min for power/flow
    for col in temp_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].rolling(window=window_temp, center=True, min_periods=1).median()

    for col in power_cols + flow_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].rolling(window=window_other, center=True, min_periods=1).median()

    logging.info(f"Applied median filtering (temps={window_temp}, other={window_other})")

    # Interpolate short gaps (up to 20min)
    max_gap = 4
    for col in temp_cols + power_cols + flow_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].interpolate(method='linear', limit=max_gap)
    
    logging.info(f"Interpolated gaps up to {max_gap} readings")
    
    # Physics check - temp diff should match power direction
    if all(col in df_clean.columns for col in ['supply_temp', 'return_temp', 'power_kw']):
        temp_diff = df_clean['return_temp'] - df_clean['supply_temp']
        power = df_clean['power_kw']
        
        # Heat extraction (neg power) -> cooler return (neg diff)
        # Heat rejection (pos power) -> warmer return (pos diff)
        extraction_mask = power < 0
        rejection_mask = power > 0
        
        # Remove physically inconsistent readings
        inconsistent_extraction = extraction_mask & (temp_diff > 2.0)  # Return much warmer during extraction
        inconsistent_rejection = rejection_mask & (temp_diff < -2.0)   # Return much cooler during rejection
        
        inconsistent_total = inconsistent_extraction.sum() + inconsistent_rejection.sum()
        if inconsistent_total > 0:
            df_clean.loc[inconsistent_extraction | inconsistent_rejection, ['supply_temp', 'return_temp']] = np.nan
            logging.info(f"Removed {inconsistent_total} physically inconsistent temperature readings")
    
    # Remove rows with too many missing values
    essential_cols = temp_cols + power_cols + flow_cols
    available_cols = [col for col in essential_cols if col in df_clean.columns]
    
    # Adaptive threshold
    if dataset_name == 'complete_field':
        missing_threshold = 0.8  # more lenient for production field
    else:
        missing_threshold = 0.5  # standard for research wells
    
    if available_cols:
        missing_ratio = df_clean[available_cols].isnull().sum(axis=1) / len(available_cols)
        rows_removed = (missing_ratio > missing_threshold).sum()
        df_clean = df_clean[missing_ratio <= missing_threshold].copy()
        
        if rows_removed > 0:
            logging.info(f"Removed {rows_removed} rows with >{missing_threshold*100}% missing data")
    
    # Final cleanup - drop remaining NaN rows
    initial_len = len(df_clean)
    df_clean = df_clean.dropna(subset=available_cols).copy()
    final_removed = initial_len - len(df_clean)
    
    if final_removed > 0:
        logging.info(f"Removed {final_removed} rows with remaining NaN values")
    
    logging.info(f"Cleaned {dataset_name} data: {len(df_clean)} records remaining ({len(df_clean)/len(df)*100:.1f}% of original)")
    
    return df_clean

def calculate_collector_efficiency(df):
    """Calculate collector efficiency = |Q_well| / |ΔT| [kW/°C].
    
    Higher = more efficient heat transfer.
    
    Note: Not currently used in the code.
    """
    
    if 'supply_temp' not in df.columns or 'return_temp' not in df.columns or 'power_kw' not in df.columns:
        logging.warning("Missing columns for collector efficiency calculation")
        return df
    
    delta_T = df['supply_temp'] - df['return_temp']
    Q_well = df['power_kw']  # Already in kW
    
    efficiency = np.where(
        np.abs(delta_T) > 0.1,  # Threshold in °C
        np.abs(Q_well) / np.abs(delta_T),
        np.nan
    )
    
    df['collector_efficiency'] = efficiency
    
    valid_count = (~np.isnan(efficiency)).sum()
    logging.info(f"Calculated collector efficiency for {valid_count} valid measurements")
    
    return df

def calculate_cumulative_energy(df, time_interval_minutes=5):
    """Calculate cumulative energy using trapezoidal integration.
    
    E ≈ Σ[(Q_i + Q_{i+1})/2] * Δt
    """
    
    if 'power_kw' not in df.columns:
        logging.warning("Missing power_kw column for energy integration")
        return df
    
    delta_t = time_interval_minutes / 60.0  # Convert to hours
    Q = df['power_kw'].values
    cumulative_energy = np.zeros(len(Q))
    
    for i in range(1, len(Q)):
        if not (np.isnan(Q[i-1]) or np.isnan(Q[i])):
            energy_increment = ((Q[i-1] + Q[i]) / 2.0) * delta_t
            cumulative_energy[i] = cumulative_energy[i-1] + energy_increment
        else:
            cumulative_energy[i] = cumulative_energy[i-1]
    
    df['cumulative_energy_kwh'] = cumulative_energy
    
    total_energy = cumulative_energy[-1] if len(cumulative_energy) > 0 else 0
    logging.info(f"Total cumulative energy: {total_energy:.2f} kWh")
    
    return df

class BHEDataset(Dataset):
    """Dataset for collector-specific training."""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class SimpleNeuralNetwork(nn.Module):
    """Feedforward network for collector training."""
    
    def __init__(self, input_features, hidden_size=128, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

class ComprehensiveDataset(Dataset):
    """Dataset for comprehensive BHE analysis."""
    
    def __init__(self, df, seq_len, horizon, feature_cols, target_col, mean=None, std=None):
        self.seq_len = seq_len
        self.horizon = horizon
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Prepare data
        self.data = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)
        
        # Standardization
        if mean is None or std is None:
            self.mean = np.mean(self.data, axis=0)
            self.std = np.std(self.data, axis=0) + 1e-8
        else:
            self.mean = mean
            self.std = std
        
        # Standardize features
        self.data = (self.data - self.mean) / self.std
        
        # Create sequences
        self.sequences = []
        self.labels = []
        
        for i in range(len(self.data) - seq_len - horizon + 1):
            self.sequences.append(self.data[i:i+seq_len])
            self.labels.append(self.targets[i+seq_len+horizon-1])
        
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)
        
        logging.info(f"Created dataset with {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.labels[idx]])

class ComprehensiveCNNLSTM(nn.Module):
    """CNN-LSTM model for BHE analysis."""
    
    def __init__(self, input_features, conv_channels, kernel_size, lstm_hidden, lstm_layers, dropout):
        super(ComprehensiveCNNLSTM, self).__init__()
        
        self.input_features = input_features
        
        # CNN layers
        layers = []
        in_channels = input_features
        
        for out_channels in conv_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        batch_size, seq_len, features = x.shape
        
        # Reshape for CNN: (batch_size, features, seq_len)
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        x = self.conv_layers(x)
        
        # Reshape back for LSTM: (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        x = lstm_out[:, -1, :]
        
        # Output layers
        x = self.output_layers(x)
        
        return x

def train_model(model, train_loader, val_loader, epochs, lr, device, patience):
    """Train the CNN-LSTM model with optimized performance and GPU monitoring."""
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience, verbose=True
    )
    
    # Use mixed precision if on GPU
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    # GPU info if available
    if device == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
    
    logging.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        # Check GPU memory on first epoch
        if device == 'cuda' and epoch == 1:
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"GPU memory allocated before training: {allocated_before:.2f} GB")
        
        # Training loop with potential mixed precision
        for batch_data, batch_targets in train_loader:
            batch_data = batch_data.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler is not None:  # mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:  # Standard training
                outputs = model(batch_data)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_data, batch_targets in val_loader:
                batch_data = batch_data.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_data)
                        loss = criterion(outputs, batch_targets)
                else:
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_targets)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Adjust learning rate
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Log with GPU memory info
        if epoch % 2 == 0 or epoch == 1 or patience_counter >= patience:
            gpu_info = ""
            if device == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                gpu_info = f", GPU: {allocated:.1f}GB/{cached:.1f}GB"
            
            logging.info(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                        f"Best Val: {best_val_loss:.6f}, Patience: {patience_counter}/{patience}{gpu_info}")
        
        if patience_counter >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # GPU memory summary
    if device == 'cuda':
        final_allocated = torch.cuda.memory_allocated() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        logging.info(f"Training complete - GPU memory: {final_allocated:.2f}GB allocated, {max_allocated:.2f}GB peak")
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}

def evaluate_model(model, data_loader, device):
    """Evaluate model and return predictions and targets."""
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data, batch_targets in data_loader:
            batch_data = batch_data.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            
            outputs = model(batch_data)
            
            # Keep on GPU, convert at the end
            all_predictions.append(outputs.cpu())
            all_targets.append(batch_targets.cpu())
    
    # Concatenate then convert to numpy
    predictions = torch.cat(all_predictions, dim=0).numpy().flatten()
    targets = torch.cat(all_targets, dim=0).numpy().flatten()
    
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    
    return predictions, targets, mae, rmse

def create_comprehensive_collector_analysis(
    full_df,
    test_df,
    predictions,
    targets,
    feature_cols,
    config_analysis,
    train_history_hours=TRAIN_HISTORY_WINDOW_HOURS,
    forecast_hours=FORECAST_WINDOW_HOURS
):
    """Generate per-collector plots with training history and forecast."""

    logging.info("Creating collector analysis with train/forecast split...")

    if 'Timestamp' not in test_df.columns:
        logging.warning("Timestamp column missing in test data; cannot build timeline plot")
        return None

    test_data = test_df.copy()

    if len(predictions) < len(test_data):
        padded_predictions = np.full(len(test_data), np.nan)
        padded_predictions[-len(predictions):] = predictions
        test_data['predicted_temp'] = padded_predictions
    else:
        test_data['predicted_temp'] = predictions[:len(test_data)]

    test_data['actual_temp'] = test_data['return_temp']

    valid_data = test_data.dropna(subset=['predicted_temp']).copy()
    if valid_data.empty:
        logging.warning("No valid data for comprehensive collector analysis")
        return None

    prediction_start = valid_data['Timestamp'].min()
    prediction_end = valid_data['Timestamp'].max()
    logging.info("Forecast span: %s to %s", prediction_start, prediction_end)

    colors = {
        'single_u45mm': '#2E86AB',
        'double_u45mm': '#A23B72',
        'muovi_ellipse_63mm': '#F18F01'
    }

    labels = {
        'single_u45mm': 'Single U45mm (Complete Field)',
        'double_u45mm': 'Double U45mm (Research)',
        'muovi_ellipse_63mm': 'MuoviEllipse 63mm (Research)'
    }

    default_order = ['single_u45mm', 'double_u45mm', 'muovi_ellipse_63mm']
    available_types = list(valid_data['bhe_type'].unique())
    collector_order = [t for t in default_order if t in available_types]
    collector_order.extend([t for t in available_types if t not in collector_order])

    collector_segments: Dict[str, Dict[str, pd.DataFrame]] = {}
    collector_metrics: Dict[str, Dict[str, float]] = {}

    for bhe_type in collector_order:
        type_forecast = valid_data[valid_data['bhe_type'] == bhe_type].sort_values('Timestamp')
        if type_forecast.empty:
            continue

        forecast_segment = type_forecast.copy()
        if forecast_hours is not None:
            forecast_limit = prediction_start + pd.Timedelta(hours=forecast_hours)
            limited_segment = forecast_segment[forecast_segment['Timestamp'] <= forecast_limit]
            if not limited_segment.empty:
                forecast_segment = limited_segment
        forecast_segment['phase'] = 'forecast'

        base_train = full_df[
            (full_df['bhe_type'] == bhe_type) & (full_df['Timestamp'] < prediction_start)
        ].copy()
        base_train = base_train.dropna(subset=['return_temp'])
        base_train = base_train.sort_values('Timestamp')

        if train_history_hours is not None and not base_train.empty:
            history_cutoff = prediction_start - pd.Timedelta(hours=train_history_hours)
            train_segment = base_train[base_train['Timestamp'] >= history_cutoff]
            if train_segment.empty:
                train_segment = base_train.tail(min(len(base_train), 3 * SEQ_LEN))
        else:
            train_segment = base_train

        if not train_segment.empty:
            train_segment = train_segment.copy()
            train_segment['actual_temp'] = train_segment['return_temp']
            train_segment['predicted_temp'] = np.nan
            train_segment['phase'] = 'training'
        else:
            train_segment = pd.DataFrame(columns=['Timestamp', 'actual_temp', 'predicted_temp', 'phase'])

        collector_segments[bhe_type] = {
            'train': train_segment,
            'forecast': forecast_segment
        }

        metric_mask = ~(forecast_segment['predicted_temp'].isna() | forecast_segment['actual_temp'].isna())
        if metric_mask.sum() == 0:
            continue
        mae = mean_absolute_error(
            forecast_segment.loc[metric_mask, 'actual_temp'],
            forecast_segment.loc[metric_mask, 'predicted_temp']
        )
        rmse = np.sqrt(
            mean_squared_error(
                forecast_segment.loc[metric_mask, 'actual_temp'],
                forecast_segment.loc[metric_mask, 'predicted_temp']
            )
        )
        collector_metrics[bhe_type] = {
            'mae': mae,
            'rmse': rmse,
            'count': int(metric_mask.sum()),
            'train_samples': int(len(base_train))
        }

    if not collector_segments:
        logging.warning("No collector segments available for plotting")
        return None

    num_collectors = len(collector_segments)
    fig_height = 4.2 * num_collectors + 3
    fig = plt.figure(figsize=(18, fig_height))
    gs = fig.add_gridspec(num_collectors + 1, 1,
                          height_ratios=[3.1] * num_collectors + [1.8],
                          hspace=0.35)

    for idx, bhe_type in enumerate(collector_order):
        segments = collector_segments.get(bhe_type)
        if segments is None:
            continue

        train_segment = segments['train']
        forecast_segment = segments['forecast']
        if forecast_segment.empty:
            continue

        ax = fig.add_subplot(gs[idx, 0])
        color = colors.get(bhe_type, '#333333')
        label = labels.get(bhe_type, bhe_type)

        handles = []
        legend_labels = []

        if not train_segment.empty:
            train_line, = ax.plot(
                train_segment['Timestamp'],
                train_segment['actual_temp'],
                color=color,
                linewidth=1.2,
                alpha=0.4,
                label='Training Actual'
            )
            handles.append(train_line)
            legend_labels.append('Training Actual')

        # Plot actual with thick solid line
        forecast_actual_line, = ax.plot(
            forecast_segment['Timestamp'],
            forecast_segment['actual_temp'],
            color=color,
            linewidth=2.5,
            alpha=0.9,
            label='Forecast Actual',
            zorder=3
        )
        handles.append(forecast_actual_line)
        legend_labels.append('Forecast Actual')

        # Plot prediction with black dashed line (distinct from actual)
        forecast_pred_line, = ax.plot(
            forecast_segment['Timestamp'],
            forecast_segment['predicted_temp'],
            color='black',
            linestyle='--',
            linewidth=2.0,
            alpha=0.7,
            label='Prediction',
            zorder=4
        )
        handles.append(forecast_pred_line)
        legend_labels.append('Prediction')

        forecast_end_time = forecast_segment['Timestamp'].max()
        ax.axvline(prediction_start, color='#4f4f4f', linestyle='--', linewidth=1.1, alpha=0.8)
        ax.axvspan(prediction_start, forecast_end_time, color=color, alpha=0.05)
        ax.text(
            prediction_start,
            0.97,
            'Forecast start',
            transform=ax.get_xaxis_transform(),
            fontsize=10,
            ha='left',
            va='top',
            color='#4f4f4f'
        )

        ax.set_title(label, fontsize=14, fontweight='bold', pad=12)
        ax.set_ylabel('Outlet Temperature (°C)', fontsize=12)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        if handles and idx == 0:
            ax.legend(handles, legend_labels, loc='upper left', fontsize=10, framealpha=0.95)

        metrics = collector_metrics.get(bhe_type)
        if metrics is not None:
            ax.text(
                0.98,
                0.95,
                f"MAE: {metrics['mae']:.3f}°C\nRMSE: {metrics['rmse']:.3f}°C\nTrain Samples: {metrics['train_samples']:,}\nForecast Samples: {metrics['count']:,}",
                transform=ax.transAxes,
                fontsize=11,
                ha='right',
                va='top',
                bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.4')
            )

        if not train_segment.empty:
            x_min = train_segment['Timestamp'].min()
        else:
            x_min = forecast_segment['Timestamp'].min()
        x_max = forecast_segment['Timestamp'].max()
        ax.set_xlim(x_min, x_max)

        if idx == num_collectors - 1:
            ax.set_xlabel('Time', fontsize=12, labelpad=10)
        else:
            ax.set_xlabel('')

    summary_ax = fig.add_subplot(gs[-1, 0])
    summary_ax.axis('off')
    summary_ax.text(
        0.5,
        0.9,
        'Model Prediction Metrics (Forecast Window)',
        fontsize=14,
        fontweight='bold',
        ha='center',
        va='top',
        transform=summary_ax.transAxes
    )

    table_rows = []
    for bhe_type in collector_order:
        metrics = collector_metrics.get(bhe_type)
        if metrics is None:
            continue
        table_rows.append([
            labels.get(bhe_type, bhe_type),
            f"{metrics['mae']:.3f}°C",
            f"{metrics['rmse']:.3f}°C",
            f"{metrics['train_samples']:,}",
            f"{metrics['count']:,}"
        ])

    table = summary_ax.table(
        cellText=table_rows,
        colLabels=['Collector', 'MAE', 'RMSE', 'Training Samples', 'Forecast Samples'],
        loc='center',
        cellLoc='center',
        bbox=[0.05, 0.0, 0.9, 0.75]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.2)

    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, 'comprehensive_collector_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    logging.info("Comprehensive collector analysis saved to: %s", plot_path)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE COLLECTOR ANALYSIS SUMMARY")
    print("=" * 80)

    for bhe_type in collector_order:
        metrics = collector_metrics.get(bhe_type)
        if metrics is None:
            continue
        label = labels.get(bhe_type, bhe_type)
        print(f"  {label}:")
        print(f"    MAE: {metrics['mae']:.4f}°C")
        print(f"    RMSE: {metrics['rmse']:.4f}°C")
    print(f"    Training Samples: {metrics['train_samples']:,}")
    print(f"    Forecast Samples: {metrics['count']:,}")

    print("=" * 80)

    return plot_path

def create_collector_performance_analysis(all_data_clean):
    """Create collector configuration performance plots with smoothed data and MAE/RMSE comparison."""
    
    logging.info("Creating collector configuration performance analysis...")
    
    # Prepare data with smoothed averages
    performance_data = {}
    
    window_size = 12

    for bhe_type, data in all_data_clean.items():
        if len(data) == 0:
            continue
            
        # Calculate temperature difference
        temp_diff = data['supply_temp'] - data['return_temp']

        # Apply gentle smoothing (12-point moving average ≈ one hour)
        temp_diff_smooth = temp_diff.rolling(window=window_size, center=True, min_periods=1).mean()
        supply_smooth = data['supply_temp'].rolling(window=window_size, center=True, min_periods=1).mean()
        return_smooth = data['return_temp'].rolling(window=window_size, center=True, min_periods=1).mean()
        power_smooth = data['power_kw'].rolling(window=window_size, center=True, min_periods=1).mean()
        flow_smooth = data['flow_rate'].rolling(window=window_size, center=True, min_periods=1).mean()
        
        performance_data[bhe_type] = {
            'timestamps': data['Timestamp'],
            'temp_diff_raw': temp_diff,
            'temp_diff_smooth': temp_diff_smooth,
            'supply_smooth': supply_smooth,
            'return_smooth': return_smooth,
            'power_smooth': power_smooth,
            'flow_smooth': flow_smooth
        }
    
    # Create comprehensive performance figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = {
        'single_u45mm': '#2E86AB',           # Deep blue
        'double_u45mm': '#A23B72',           # Deep magenta 
        'muovi_ellipse_63mm': '#F18F01'      # Orange
    }
    
    labels = {
        'single_u45mm': 'Single U45mm (Complete Field)',
        'double_u45mm': 'Double U45mm (Research)',
        'muovi_ellipse_63mm': 'MuoviEllipse 63mm (Research)'
    }
    
    # Plot 1: Temperature Response Time Series (with smoothed data)
    ax1 = fig.add_subplot(gs[0, :2])
    
    for bhe_type, data in performance_data.items():
        color = colors.get(bhe_type, '#333333')
        label = labels.get(bhe_type, bhe_type)
        
        # Plot smoothed temperature difference
        ax1.plot(data['timestamps'], data['temp_diff_smooth'], 
                color=color, alpha=0.9, linewidth=2.5, label=label)
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Temperature Difference (°C)', fontsize=12, fontweight='bold')
    ax1.set_title('BHE Temperature Response by Collector Configuration (Smoothed)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Performance Metrics (MAE/RMSE would go here after model training)
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Calculate relative thermal performance
    thermal_performance = []
    type_names = []
    
    for bhe_type, data in performance_data.items():
        # Calculate average absolute temperature difference
        avg_temp_diff = np.abs(data['temp_diff_smooth']).mean()
        thermal_performance.append(avg_temp_diff)
        type_names.append(labels.get(bhe_type, bhe_type))
    
    bars = ax2.bar(range(len(type_names)), thermal_performance,
                   color=[colors.get(list(performance_data.keys())[i], '#333333') for i in range(len(type_names))])
    
    ax2.set_xlabel('Collector Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average |Temperature Difference| (°C)', fontsize=12, fontweight='bold')
    ax2.set_title('Thermal Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(type_names)))
    ax2.set_xticklabels([name.replace(' (', '\n(') for name in type_names], fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, perf) in enumerate(zip(bars, thermal_performance)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{perf:.3f}°C', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 3: Power vs Temperature Response
    ax3 = fig.add_subplot(gs[1, 0])
    
    for bhe_type, data in performance_data.items():
        color = colors.get(bhe_type, '#333333')
        label = labels.get(bhe_type, bhe_type)
        
        ax3.scatter(data['power_smooth'], np.abs(data['temp_diff_smooth']), 
                   alpha=0.6, s=3, color=color, label=label)
    
    ax3.set_xlabel('Power per Well (kW)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('|Temperature Difference| (°C)', fontsize=12, fontweight='bold')
    ax3.set_title('Power vs Temperature Response', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Flow Rate vs Efficiency
    ax4 = fig.add_subplot(gs[1, 1])
    
    for bhe_type, data in performance_data.items():
        color = colors.get(bhe_type, '#333333')
        label = labels.get(bhe_type, bhe_type)
        
        # Thermal efficiency: temp change per unit flow
        efficiency = np.abs(data['temp_diff_smooth']) / (data['flow_smooth'] + 1e-6)
        efficiency = efficiency[efficiency < np.percentile(efficiency, 95)]  # drop outliers
        flow_data = data['flow_smooth'][:len(efficiency)]
        
        ax4.scatter(flow_data, efficiency, alpha=0.6, s=3, color=color, label=label)
    
    ax4.set_xlabel('Flow Rate per Well (m³/h)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Thermal Efficiency (°C·h/m³)', fontsize=12, fontweight='bold')
    ax4.set_title('Flow Rate vs Thermal Efficiency', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 5)  # realistic efficiency range
    
    # Plot 5: Relative Performance
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Calculate benefits vs worst performer
    if len(thermal_performance) > 1:
        baseline = max(thermal_performance)  # Highest temp difference = worst performance
        benefits = [(baseline - perf) for perf in thermal_performance]
    else:
        benefits = [0] * len(thermal_performance)
    
    bars2 = ax5.bar(range(len(type_names)), benefits,
                    color=[colors.get(list(performance_data.keys())[i], '#333333') for i in range(len(type_names))])
    
    ax5.set_xlabel('Collector Type', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Relative Performance Benefit (°C)', fontsize=12, fontweight='bold')
    ax5.set_title('Relative Performance Benefits', fontsize=14, fontweight='bold')
    ax5.set_xticks(range(len(type_names)))
    ax5.set_xticklabels([name.replace(' (', '\n(') for name in type_names], fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, benefit) in enumerate(zip(bars2, benefits)):
        height = bar.get_height()
        sign = '+' if benefit >= 0 else ''
        ax5.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.003),
                f'{sign}{benefit:.3f}°C', ha='center', 
                va='bottom' if height >= 0 else 'top', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, 'collector_configuration_performance_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f"Collector performance analysis saved to: {plot_path}")
    
    # Summary stats
    print("\nCOLLECTOR CONFIGURATION PERFORMANCE ANALYSIS")
    
    for i, (bhe_type, data) in enumerate(performance_data.items()):
        label = labels.get(bhe_type, bhe_type)
        avg_temp_diff = thermal_performance[i]
        benefit = benefits[i]
        
        print(f"\n{label}:")
        print(f"  Data points: {len(data['temp_diff_smooth']):,}")
        print(f"  Average |ΔT|: {avg_temp_diff:.3f}°C")
        print(f"  Relative benefit: {benefit:+.3f}°C")
        print(f"  Avg power per well: {data['power_smooth'].mean():.2f} kW")
        print(f"  Avg flow per well: {data['flow_smooth'].mean():.3f} m³/h")
    
    return plot_path

def create_model_performance_comparison(test_df, predictions, targets, config_analysis):
    """Create MAE/RMSE comparison plots by collector type."""
    
    logging.info("Creating model performance comparison by collector type...")
    logging.info(f"Input data: test_df shape={test_df.shape}, predictions length={len(predictions)}, targets length={len(targets)}")
    
    # Prepare test data with predictions
    test_data = test_df.copy()
    
    # Align predictions
    if len(predictions) < len(test_data):
        logging.info(f"Padding predictions: {len(predictions)} -> {len(test_data)}")
        padded_predictions = np.full(len(test_data), np.nan)
        padded_predictions[-len(predictions):] = predictions
        test_data['predicted_temp'] = padded_predictions
    else:
        test_data['predicted_temp'] = predictions[:len(test_data)]
    
    test_data['actual_temp'] = test_data['return_temp']
    
    # Get valid rows
    valid_data = test_data.dropna(subset=['predicted_temp'])
    logging.info(f"Valid data after filtering: {len(valid_data)} rows")
    
    # Calculate metrics per collector
    collector_metrics = {}
    
    for bhe_type in valid_data['bhe_type'].unique():
        type_data = valid_data[valid_data['bhe_type'] == bhe_type]
        valid_mask = ~(np.isnan(type_data['predicted_temp']) | np.isnan(type_data['actual_temp']))
        
        if valid_mask.sum() > 0:
            pred_vals = type_data[valid_mask]['predicted_temp']
            actual_vals = type_data[valid_mask]['actual_temp']
            
            mae = mean_absolute_error(actual_vals, pred_vals)
            rmse = np.sqrt(mean_squared_error(actual_vals, pred_vals))
            
            collector_metrics[bhe_type] = {
                'mae': mae,
                'rmse': rmse,
                'count': len(pred_vals)
            }
            logging.info(f"{bhe_type}: MAE={mae:.4f}, RMSE={rmse:.4f}, count={len(pred_vals)}")
    
    # Check if we have data
    if not collector_metrics:
        logging.error("No collector metrics calculated - no data to plot!")
        # Show error message on plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.text(0.5, 0.5, 'No data available\nfor MAE comparison\n\nCheck data alignment\nbetween predictions and test data', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax2.text(0.5, 0.5, 'No data available\nfor RMSE comparison\n\nCheck data alignment\nbetween predictions and test data', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax1.set_title('Model Accuracy: MAE Comparison', fontsize=14, fontweight='bold')
        ax2.set_title('Model Accuracy: RMSE Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(OUTPUT_DIR, 'model_performance_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return plot_path, {}
    
    logging.info(f"Successfully calculated metrics for {len(collector_metrics)} collector types")

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color scheme
    colors = {
        'single_u45mm': '#2E86AB',
        'double_u45mm': '#A23B72', 
        'muovi_ellipse_63mm': '#F18F01'
    }
    
    labels = {
        'single_u45mm': 'Single U45mm\n(Complete Field)',
        'double_u45mm': 'Double U45mm\n(Research)',
        'muovi_ellipse_63mm': 'MuoviEllipse 63mm\n(Research)'
    }
    
    types = list(collector_metrics.keys())
    mae_values = [collector_metrics[t]['mae'] for t in types]
    rmse_values = [collector_metrics[t]['rmse'] for t in types]
    type_labels = [labels.get(t, t) for t in types]
    type_colors = [colors.get(t, '#333333') for t in types]
    
    # MAE comparison
    bars1 = ax1.bar(range(len(types)), mae_values, color=type_colors, alpha=0.8)
    ax1.set_xlabel('Collector Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error (°C)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy: MAE Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(types)))
    ax1.set_xticklabels(type_labels, fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on MAE bars
    for i, (bar, mae) in enumerate(zip(bars1, mae_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{mae:.3f}°C', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # RMSE comparison
    bars2 = ax2.bar(range(len(types)), rmse_values, color=type_colors, alpha=0.8)
    ax2.set_xlabel('Collector Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Root Mean Square Error (°C)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Accuracy: RMSE Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(types)))
    ax2.set_xticklabels(type_labels, fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on RMSE bars
    for i, (bar, rmse) in enumerate(zip(bars2, rmse_values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{rmse:.3f}°C', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, 'model_performance_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f"Model performance comparison saved to: {plot_path}")
    
    # Print comparison
    print("\nMODEL PERFORMANCE BY COLLECTOR TYPE")
    
    for bhe_type in types:
        metrics = collector_metrics[bhe_type]
        label = labels.get(bhe_type, bhe_type)
        print(f"\n{label}:")
        print(f"  Test samples: {metrics['count']:,}")
        print(f"  MAE: {metrics['mae']:.4f}°C")
        print(f"  RMSE: {metrics['rmse']:.4f}°C")
    
    return plot_path, collector_metrics

def create_raw_data_collector_analysis(complete_field, double_u45, muovi_ellipse):
    """Create collector analysis using raw data before modeling."""
    
    logging.info("Creating raw data collector configuration analysis...")
    
    plt.style.use('default')
    
    # Calculate temperature differences once for efficiency
    complete_temp_diff = complete_field['supply_temp'] - complete_field['return_temp']
    double_temp_diff = double_u45['supply_temp'] - double_u45['return_temp']
    muovi_temp_diff = muovi_ellipse['supply_temp'] - muovi_ellipse['return_temp']
    
    # Create comprehensive analysis figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Temperature difference comparison with smoothed data
    ax1 = axes[0, 0]
    
    # Apply smoothing (12-point moving averages)
    window_size = 12
    complete_smooth = complete_temp_diff.rolling(window=window_size, center=True, min_periods=1).mean()
    double_smooth = double_temp_diff.rolling(window=window_size, center=True, min_periods=1).mean()
    muovi_smooth = muovi_temp_diff.rolling(window=window_size, center=True, min_periods=1).mean()
    
    ax1.plot(complete_field['Timestamp'], complete_smooth, 
             label='Single U45mm (Complete Field)', color='#2E86AB', alpha=0.8, linewidth=2)
    ax1.plot(double_u45['Timestamp'], double_smooth, 
             label='Double U45mm (Research)', color='#A23B72', alpha=0.8, linewidth=2)
    ax1.plot(muovi_ellipse['Timestamp'], muovi_smooth, 
             label='MuoviEllipse 63mm (Research)', color='#F18F01', alpha=0.8, linewidth=2)
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.set_xlabel('Time', fontweight='bold')
    ax1.set_ylabel('Temperature Difference (°C)', fontweight='bold')
    ax1.set_title('BHE Temperature Response (Smoothed)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Performance comparison bars
    ax2 = axes[0, 1]
    
    collectors = ['Single U45mm', 'Double U45mm', 'MuoviEllipse 63mm']
    avg_temp_diffs = [
        np.abs(complete_smooth).mean(),
        np.abs(double_smooth).mean(),
        np.abs(muovi_smooth).mean()
    ]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = ax2.bar(collectors, avg_temp_diffs, color=colors, alpha=0.8)
    ax2.set_ylabel('Average |Temperature Difference| (°C)', fontweight='bold')
    ax2.set_title('Thermal Performance Comparison', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, avg_temp_diffs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}°C', ha='center', va='bottom', fontweight='bold')
    
    # Power utilization analysis (per well)
    ax3 = axes[1, 0]
    
    ax3.scatter(complete_field['power_kw'], np.abs(complete_temp_diff), 
                alpha=0.4, s=2, color='#2E86AB', label='Single U45mm')
    ax3.scatter(double_u45['power_kw'], np.abs(double_temp_diff), 
                alpha=0.6, s=2, color='#A23B72', label='Double U45mm')
    ax3.scatter(muovi_ellipse['power_kw'], np.abs(muovi_temp_diff), 
                alpha=0.6, s=2, color='#F18F01', label='MuoviEllipse 63mm')
    
    ax3.set_xlabel('Power per Well (kW)', fontweight='bold')
    ax3.set_ylabel('|Temperature Difference| (°C)', fontweight='bold')
    ax3.set_title('Power vs Temperature Response (Per Well)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Relative performance benefits
    ax4 = axes[1, 1]
    
    # Calculate benefits relative to worst performer
    baseline = max(avg_temp_diffs)
    benefits = [(baseline - diff) for diff in avg_temp_diffs]
    
    bars2 = ax4.bar(collectors, benefits, color=colors, alpha=0.8)
    ax4.set_ylabel('Relative Performance Benefit (°C)', fontweight='bold')
    ax4.set_title('Relative Performance Benefits', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add benefit labels
    for bar, benefit in zip(bars2, benefits):
        height = bar.get_height()
        sign = '+' if benefit >= 0 else ''
        ax4.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.003),
                f'{sign}{benefit:.3f}°C', ha='center', 
                va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, 'raw_data_collector_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f"Raw data collector analysis saved to: {plot_path}")
    
    # Print detailed analysis results
    print("\n" + "="*80)
    print("RAW DATA COLLECTOR ANALYSIS (Per Well Normalized)")
    print("="*80)
    
    for i, collector in enumerate(collectors):
        print(f"\n{collector}:")
        print(f"   Average |ΔT|: {avg_temp_diffs[i]:.3f}°C")
        print(f"   Relative benefit: {benefits[i]:+.3f}°C")
        
        if i == 0:  # Complete field
            print(f"   Avg power per well: {complete_field['power_kw'].mean():.2f} kW")
            print(f"   Avg flow per well: {complete_field['flow_rate'].mean():.3f} m³/h")
        elif i == 1:  # Double U45
            print(f"   Avg power per well: {double_u45['power_kw'].mean():.2f} kW")
            print(f"   Avg flow per well: {double_u45['flow_rate'].mean():.3f} m³/h")
        else:  # MuoviEllipse
            print(f"   Avg power per well: {muovi_ellipse['power_kw'].mean():.2f} kW")
            print(f"   Avg flow per well: {muovi_ellipse['flow_rate'].mean():.3f} m³/h")
    
    print("="*80)
    
    return plot_path
    """Create standalone collector configuration analysis using raw data."""
    
    logging.info("Creating raw data collector configuration analysis...")
    
    plt.style.use('default')
    
    # Calculate temperature differences once for efficiency
    complete_temp_diff = complete_field['supply_temp'] - complete_field['return_temp']
    double_temp_diff = double_u45['supply_temp'] - double_u45['return_temp']
    muovi_temp_diff = muovi_ellipse['supply_temp'] - muovi_ellipse['return_temp']
    
    # Create separate figure for temperature profiles  
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 6))
    
    # Temperature profiles comparison
    ax1.plot(complete_field['Timestamp'], complete_temp_diff, 
             label='Single U45mm (Complete Field)', color='#2E86AB', alpha=0.6, linewidth=1)
    ax1.plot(double_u45['Timestamp'], double_temp_diff, 
             label='Double U45mm (Research)', color='#A23B72', alpha=0.7, linewidth=1)
    ax1.plot(muovi_ellipse['Timestamp'], muovi_temp_diff, 
             label='MuoviEllipse 63mm (Research)', color='#F18F01', alpha=0.7, linewidth=1)
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.set_xlabel('Time', fontweight='bold')
    ax1.set_ylabel('Temperature Difference (°C)', fontweight='bold')
    ax1.set_title('BHE Temperature Response by Collector Configuration', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Main analysis figure with 2x2 layout
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Flow rate comparison (per well)
    ax2 = axes[0, 0]
    
    # Complete field flow per well (120 total)
    complete_field_normalized_flow = complete_field['flow_rate'] / 120
    
    ax2.hist(complete_field_normalized_flow, bins=50, alpha=0.6, 
             label=f'Single U45mm (per well)\nMean: {complete_field_normalized_flow.mean():.2f} m³/h', 
             color='#2E86AB', density=True)
    ax2.hist(double_u45['flow_rate'], bins=50, alpha=0.6, 
             label=f'Double U45mm\nMean: {double_u45["flow_rate"].mean():.2f} m³/h', 
             color='#A23B72', density=True)
    ax2.hist(muovi_ellipse['flow_rate'], bins=50, alpha=0.6, 
             label=f'MuoviEllipse 63mm\nMean: {muovi_ellipse["flow_rate"].mean():.2f} m³/h', 
             color='#F18F01', density=True)
    
    ax2.set_xlabel('Flow Rate (m³/h per well)', fontweight='bold')
    ax2.set_ylabel('Probability Density', fontweight='bold')
    ax2.set_title('Flow Rate Distribution (Per Well)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Heat transfer efficiency analysis
    ax3 = axes[0, 1]
    
    # Calculate thermal efficiency: ΔT / flow_rate (temperature change per unit flow)
    complete_efficiency = np.abs(complete_temp_diff) / complete_field_normalized_flow
    double_efficiency = np.abs(double_temp_diff) / double_u45['flow_rate']
    muovi_efficiency = np.abs(muovi_temp_diff) / muovi_ellipse['flow_rate']
    
    # Remove outliers for cleaner visualization
    complete_efficiency = complete_efficiency[complete_efficiency < np.percentile(complete_efficiency, 95)]
    double_efficiency = double_efficiency[double_efficiency < np.percentile(double_efficiency, 95)]
    muovi_efficiency = muovi_efficiency[muovi_efficiency < np.percentile(muovi_efficiency, 95)]
    
    ax3.scatter(complete_field_normalized_flow[:len(complete_efficiency)], complete_efficiency, 
                alpha=0.3, s=1, color='#2E86AB', label='Single U45mm')
    ax3.scatter(double_u45['flow_rate'][:len(double_efficiency)], double_efficiency, 
                alpha=0.5, s=1, color='#A23B72', label='Double U45mm')
    ax3.scatter(muovi_ellipse['flow_rate'][:len(muovi_efficiency)], muovi_efficiency, 
                alpha=0.5, s=1, color='#F18F01', label='MuoviEllipse 63mm')
    
    ax3.set_xlabel('Flow Rate (m³/h)', fontweight='bold')
    ax3.set_ylabel('Thermal Efficiency (°C·h/m³)', fontweight='bold')
    ax3.set_title('Heat Transfer Efficiency vs Flow Rate', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 5)  # Focus on realistic efficiency range
    
    # Power utilization analysis
    ax4 = axes[1, 0]
    
    ax4.scatter(complete_field['power_kw'], np.abs(complete_temp_diff), 
                alpha=0.3, s=1, color='#2E86AB', label='Single U45mm')
    ax4.scatter(double_u45['power_kw'], np.abs(double_temp_diff), 
                alpha=0.5, s=1, color='#A23B72', label='Double U45mm')
    ax4.scatter(muovi_ellipse['power_kw'], np.abs(muovi_temp_diff), 
                alpha=0.5, s=1, color='#F18F01', label='MuoviEllipse 63mm')
    
    ax4.set_xlabel('Power (kW)', fontweight='bold')
    ax4.set_ylabel('|Temperature Difference| (°C)', fontweight='bold')
    ax4.set_title('Power vs Temperature Response', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Cross-sectional geometry analysis
    ax5 = axes[1, 1]
    
    # BHE configurations
    collectors = ['Single U45mm', 'Double U45mm', 'MuoviEllipse 63mm']
    
    # Pipe specifications (outer diameter accounting for wall thickness)
    # U45: 45mm nominal + 2×2.6mm wall = 50.2mm outer diameter
    # MuoviEllipse: elliptical cross-section 51mm × 73mm (from technical drawing)
    borehole_diameter = 140  # mm (actual borehole)
    
    # Heat transfer surface area per meter depth
    u45_outer_diameter = 50.2  # mm (45mm + 2×2.6mm wall)
    surface_areas_per_m = [
        2 * np.pi * u45_outer_diameter * 1000,      # Single U45mm: 2 legs
        4 * np.pi * u45_outer_diameter * 1000,      # Double U45mm: 4 legs (2 U-tubes)
        np.pi * np.sqrt(51 * 73) * 1000             # MuoviEllipse: approximate with equivalent circle
    ]
    
    # Pipe cross-sectional area (flow area occupied in borehole)
    pipe_cross_sections = [
        2 * np.pi * (u45_outer_diameter/2)**2,      # Single U45mm: 2 pipes
        4 * np.pi * (u45_outer_diameter/2)**2,      # Double U45mm: 4 pipes
        np.pi * (51/2) * (73/2)                     # MuoviEllipse: ellipse area
    ]
    
    borehole_cross_section = np.pi * (borehole_diameter/2)**2
    
    # Grout volume ratio (thermal coupling efficiency indicator)
    grout_ratios = [(borehole_cross_section - pipe_area) / borehole_cross_section 
                    for pipe_area in pipe_cross_sections]
    
    # Average temperature differences
    avg_temp_diffs = [
        np.abs(complete_temp_diff).mean(),
        np.abs(double_temp_diff).mean(),
        np.abs(muovi_temp_diff).mean()
    ]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax5.bar(collectors, avg_temp_diffs, color=colors, alpha=0.7)
    
    # Add flow area ratio as text
    for i, (bar, ratio) in enumerate(zip(bars, flow_ratios)):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'Flow Ratio: {ratio:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax5.set_ylabel('Average |Temperature Difference| (°C)', fontweight='bold')
    ax5.set_title('BHE Performance vs Cross-Sectional Configuration', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save both figures
    fig1.savefig('output/temperature_profiles_analysis.png', dpi=300, bbox_inches='tight')
    fig2.savefig('output/raw_data_collector_analysis.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print detailed analysis results
    print("\n" + "="*80)
    print("COMPREHENSIVE BHE COLLECTOR ANALYSIS (RAW DATA)")
    print("="*80)
    
    print(f"\n1. FLOW RATE ANALYSIS (Per well):")
    print(f"   Single U45mm:      {complete_field_normalized_flow.mean():.2f} ± {complete_field_normalized_flow.std():.2f} m³/h")
    print(f"   Double U45mm:      {double_u45['flow_rate'].mean():.2f} ± {double_u45['flow_rate'].std():.2f} m³/h")
    print(f"   MuoviEllipse 63mm: {muovi_ellipse['flow_rate'].mean():.2f} ± {muovi_ellipse['flow_rate'].std():.2f} m³/h")
    
    print(f"\n2. TEMPERATURE RESPONSE:")
    print(f"   Single U45mm:      {complete_temp_diff.mean():.3f} ± {complete_temp_diff.std():.3f} °C")
    print(f"   Double U45mm:      {double_temp_diff.mean():.3f} ± {double_temp_diff.std():.3f} °C")
    print(f"   MuoviEllipse 63mm: {muovi_temp_diff.mean():.3f} ± {muovi_temp_diff.std():.3f} °C")
    
    print(f"\n3. BHE GEOMETRY AND THERMAL PERFORMANCE:")
    print(f"   Borehole diameter: {borehole_diameter} mm")
    print(f"   U45 pipe outer diameter: {u45_outer_diameter} mm (45mm + 2×2.6mm wall)\n")
    
    for i, collector in enumerate(collectors):
        print(f"   {collector}:")
        print(f"      Heat transfer surface: {surface_areas_per_m[i]:,.0f} mm²/m")
        print(f"      Pipe cross-section: {pipe_cross_sections[i]:,.0f} mm²")
        print(f"      Grout fill ratio: {grout_ratios[i]:.1%}")
        print(f"      Avg |ΔT|: {avg_temp_diffs[i]:.3f} °C")
    
    print(f"\n4. OPERATIONAL MODES:")
    # Heat extraction: supply > return (positive temp_diff), negative power
    complete_extraction = (complete_temp_diff > 0).mean() * 100
    double_extraction = (double_temp_diff > 0).mean() * 100
    muovi_extraction = (muovi_temp_diff > 0).mean() * 100
    
    print(f"   Single U45mm:      {complete_extraction:.1f}% heat extraction mode")
    print(f"   Double U45mm:      {double_extraction:.1f}% heat extraction mode")
    print(f"   MuoviEllipse 63mm: {muovi_extraction:.1f}% heat extraction mode")
    
    print("="*80)

def train_collector_specific_model(data, collector_type, device):
    """Train a model specific to one collector type."""
    
    logging.info(f"Training {collector_type} model with {len(data)} records")
    
    # Define features and target (without bhe_type_encoded since it's single type)
    feature_cols = ['supply_temp', 'flow_rate', 'power_kw']
    target_col = 'return_temp'
    
    # Prepare data
    features = data[feature_cols].values
    targets = data[target_col].values
    
    # Split data
    train_size = int(0.6 * len(data))
    val_size = int(0.2 * len(data))
    
    X_train = features[:train_size]
    y_train = targets[:train_size]
    X_val = features[train_size:train_size+val_size]
    y_val = targets[train_size:train_size+val_size]
    X_test = features[train_size+val_size:]
    y_test = targets[train_size+val_size:]
    
    logging.info(f"Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Create datasets
    train_dataset = BHEDataset(X_train, y_train)
    val_dataset = BHEDataset(X_val, y_val)
    test_dataset = BHEDataset(X_test, y_test)
    
    # Create data loaders
    batch_size = min(512, len(train_dataset) // 10)  # Adaptive batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Initialize simple neural network model
    model = SimpleNeuralNetwork(
        input_features=3,
        hidden_size=128,
        dropout=0.3
    ).to(device)
    
    # Train model
    model, training_history = train_model(model, train_loader, val_loader, 
                                        epochs=10, lr=0.001, device=device, patience=8)
    
    # Evaluate model
    predictions, targets_eval, mae, rmse = evaluate_model(model, test_loader, device)
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, f'{collector_type}_model.pth')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")
    
    results = {
        'collector_type': collector_type,
        'mae': mae,
        'rmse': rmse,
        'training_history': training_history,
        'test_predictions': predictions.tolist(),
        'test_targets': targets_eval.tolist(),
        'model_path': model_path
    }
    
    logging.info(f"{collector_type} model - MAE: {mae:.4f}°C, RMSE: {rmse:.4f}°C")
    
    return model, results

def test_gpu_utilization(model, data_loader, device):
    """Test GPU utilization with the current model and batch size."""
    
    if device != 'cuda':
        logging.info("GPU utilization test skipped - not using CUDA")
        return
    
    logging.info("Testing GPU utilization...")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model.eval()
    with torch.no_grad():
        # Run a few batches to test GPU usage
        for i, (batch_data, batch_targets) in enumerate(data_loader):
            if i >= 5:  # Test with 5 batches
                break
                
            batch_data = batch_data.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(batch_data)
            
            # Check memory usage after each batch
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            if i == 0:
                logging.info(f"Batch {i+1}: GPU memory allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
    
    # Final memory stats
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    utilization_percent = (peak_allocated / total_memory) * 100
    
    logging.info(f"GPU Utilization Test Results:")
    logging.info(f"  Peak memory used: {peak_allocated:.2f}GB / {total_memory:.1f}GB ({utilization_percent:.1f}%)")
    logging.info(f"  Batch size: {data_loader.batch_size}")
    logging.info(f"  Recommended: {'Good utilization' if utilization_percent > 50 else 'Consider increasing batch size'}")
    
    return peak_allocated, utilization_percent

def optimize_batch_size_for_gpu(model, sample_batch, device, max_batch_size=1024):
    """Find optimal batch size for GPU memory."""
    
    if device != 'cuda':
        return 64  # Default for CPU
    
    logging.info("Finding optimal batch size for GPU...")
    
    optimal_batch_size = 64
    seq_len, features = sample_batch.shape[1], sample_batch.shape[2]
    
    for batch_size in [64, 128, 256, 384, 512, 768, 1024]:
        if batch_size > max_batch_size:
            break
            
        try:
            torch.cuda.empty_cache()
            
            # Create test batch
            test_batch = torch.randn(batch_size, seq_len, features).to(device)
            test_targets = torch.randn(batch_size, 1).to(device)
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                outputs = model(test_batch)
            
            # Test backward pass
            model.train()
            outputs = model(test_batch)
            loss = nn.MSELoss()(outputs, test_targets)
            loss.backward()
            
            # If we reach here, this batch size works
            optimal_batch_size = batch_size
            allocated = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"  Batch size {batch_size}: OK ({allocated:.2f}GB)")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.info(f"  Batch size {batch_size}: Out of memory")
                break
            else:
                raise e
    
    logging.info(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size

def analyze_bhe_configurations(combined_data):
    """Analyze the different BHE configurations."""
    
    logging.info("Analyzing BHE configurations...")
    
    analysis = {}
    for bhe_type in combined_data['bhe_type'].unique():
        subset = combined_data[combined_data['bhe_type'] == bhe_type]
        
        analysis[bhe_type] = {
            'count': len(subset),
            'avg_return_temp': subset['return_temp'].mean(),
            'avg_supply_temp': subset['supply_temp'].mean(),
            'avg_temp_diff': (subset['return_temp'] - subset['supply_temp']).mean(),
            'avg_flow_rate': subset['flow_rate'].mean(),
            'avg_power': subset['power_kw'].mean()
        }
        
        logging.info(f"{bhe_type}: {len(subset)} records, "
                    f"avg return temp: {analysis[bhe_type]['avg_return_temp']:.2f}°C, "
                    f"avg temp diff: {analysis[bhe_type]['avg_temp_diff']:.2f}°C")
    
    return analysis

def main():
    """Main execution function."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting comprehensive CNN-LSTM analysis on device: {device}")
    logging.info(f"Starting comprehensive analysis with OE401 correction and DST handling on device: {device}")
    
    # Load all data sources with corrections
    print("Loading data with OE401 correction and DST handling...")
    logging.info("Loading data from all sources...")
    
    # Load complete field data (Single U45mm) - with OE401 correction
    print("  Loading complete field data...")
    complete_field_df = load_complete_field_data()
    logging.info(f"Complete field data loaded: {len(complete_field_df)} records (corrected, 112 wells)")
    
    # Load Double U45mm research data - use raw data
    print("  Loading Double U45mm research wells data...")
    double_u45mm_df = load_double_u45mm_research_data()
    if len(double_u45mm_df) > 0:
        logging.info(f"Double U45mm data loaded: {len(double_u45mm_df)} records (raw data)")
    
    # Load MuoviEllipse data - use raw data
    print("  Loading MuoviEllipse 63mm research data...")
    muovi_ellipse_df = load_muovi_ellipse_research_data()
    if len(muovi_ellipse_df) > 0:
        logging.info(f"MuoviEllipse data loaded: {len(muovi_ellipse_df)} records (raw data)")
    
    # Find common timestamp range across all datasets
    datasets = [df for df in [complete_field_df, double_u45mm_df, muovi_ellipse_df] if len(df) > 0]
    
    if not datasets:
        raise ValueError("No valid datasets loaded")
    
    # Find overlapping timestamp range
    start_times = [df['Timestamp'].min() for df in datasets]
    end_times = [df['Timestamp'].max() for df in datasets]
    
    common_start = max(start_times)
    common_end = min(end_times)
    
    logging.info(f"Common timestamp range: {common_start} to {common_end}")
    
    # Filter all datasets to common timestamp range
    filtered_datasets = []
    for i, df in enumerate(datasets):
        original_len = len(df)
        filtered_df = df[(df['Timestamp'] >= common_start) & (df['Timestamp'] <= common_end)].copy()
        filtered_datasets.append(filtered_df)
        
        dataset_name = ['complete_field', 'double_u45mm', 'muovi_ellipse'][i] if i < 3 else f'dataset_{i}'
        logging.info(f"Filtered {dataset_name}: {original_len} -> {len(filtered_df)} records")
    
    # Store individual datasets for standalone analysis
    complete_field_clean = filtered_datasets[0] if len(filtered_datasets) > 0 else pd.DataFrame()
    double_u45_clean = filtered_datasets[1] if len(filtered_datasets) > 1 else pd.DataFrame()
    muovi_ellipse_clean = filtered_datasets[2] if len(filtered_datasets) > 2 else pd.DataFrame()
    
    # Combine filtered datasets
    combined_data = pd.concat(filtered_datasets, ignore_index=True)
    combined_data = combined_data.sort_values('Timestamp').reset_index(drop=True)
    
    logging.info(f"Combined dataset: {len(combined_data)} records from {len(filtered_datasets)} sources")
    logging.info(f"Time range: {combined_data['Timestamp'].min()} to {combined_data['Timestamp'].max()}")
    
    # Create raw data collector analysis (before model training)
    print("Creating raw data collector analysis...")
    if len(complete_field_clean) > 0 and len(double_u45_clean) > 0 and len(muovi_ellipse_clean) > 0:
        create_raw_data_collector_analysis(complete_field_clean, double_u45_clean, muovi_ellipse_clean)
        
        # Also create collector performance analysis with moving averages
        create_collector_performance_analysis({
            'single_u45mm': complete_field_clean,
            'double_u45mm': double_u45_clean, 
            'muovi_ellipse_63mm': muovi_ellipse_clean
        })
    else:
        logging.warning("Insufficient data for raw collector analysis")
    
    # Analyze BHE configurations
    config_analysis = analyze_bhe_configurations(combined_data)
    
    # Define features and target
    feature_cols = ['supply_temp', 'flow_rate', 'power_kw', 'bhe_type_encoded']
    target_col = 'return_temp'
    
    # Prepare data for training
    model_data = combined_data[feature_cols + [target_col, 'Timestamp', 'bhe_type']].dropna()
    
    logging.info(f"Model data: {len(model_data)} records with {len(feature_cols)} features")
    
    # Time-based train/validation/test split
    latest_timestamp = model_data['Timestamp'].max()
    test_start = latest_timestamp - pd.Timedelta(days=FORECAST_WINDOW_DAYS)
    val_start = test_start - pd.Timedelta(days=VALIDATION_WINDOW_DAYS)

    train_df = model_data[model_data['Timestamp'] < val_start].copy()
    val_df = model_data[(model_data['Timestamp'] >= val_start) & (model_data['Timestamp'] < test_start)].copy()
    test_df = model_data[model_data['Timestamp'] >= test_start].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Time-based split produced an empty subset. Check that the dataset covers the requested windows."
        )

    logging.info(
        "Data splits (time-based): Train=%d (up to %s), Val=%d (%s to %s), Test=%d (from %s)",
        len(train_df),
        val_start,
        len(val_df),
        val_start,
        test_start,
        len(test_df),
        test_start
    )
    
    # Create datasets
    train_dataset = ComprehensiveDataset(train_df, SEQ_LEN, PRED_HORIZON, feature_cols, target_col)
    val_dataset = ComprehensiveDataset(val_df, SEQ_LEN, PRED_HORIZON, feature_cols, target_col, 
                                     train_dataset.mean, train_dataset.std)
    test_dataset = ComprehensiveDataset(test_df, SEQ_LEN, PRED_HORIZON, feature_cols, target_col,
                                      train_dataset.mean, train_dataset.std)
    
    # Initialize model for batch size optimization
    temp_model = ComprehensiveCNNLSTM(
        input_features=len(feature_cols),
        conv_channels=CONV_CHANNELS,
        kernel_size=KERNEL_SIZE,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Optimize batch size for GPU if using CUDA
    if device == 'cuda':
        sample_batch, _ = next(iter(DataLoader(train_dataset, batch_size=1)))
        optimal_batch_size = optimize_batch_size_for_gpu(temp_model, sample_batch, device)
        actual_batch_size = min(BATCH_SIZE, optimal_batch_size)
        logging.info(f"Using batch size: {actual_batch_size} (configured: {BATCH_SIZE}, optimal: {optimal_batch_size})")
    else:
        actual_batch_size = BATCH_SIZE
    
    # Create data loaders with optimized batch size
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, 
                             shuffle=True, pin_memory=device=='cuda', num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=actual_batch_size, 
                           shuffle=False, pin_memory=device=='cuda', num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=actual_batch_size, 
                            shuffle=False, pin_memory=device=='cuda', num_workers=0)
    
    # Run GPU utilization test if using CUDA
    if device == 'cuda':
        print("Testing GPU utilization...")
        test_gpu_utilization(temp_model, train_loader, device)
    
    # Initialize model with optimized architecture
    model = ComprehensiveCNNLSTM(
        input_features=len(feature_cols),
        conv_channels=CONV_CHANNELS,
        kernel_size=KERNEL_SIZE,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model initialized: {trainable_params:,} parameters")
    logging.info(f"Model initialized with {total_params:,} parameters ({trainable_params:,} trainable)")
    logging.info(f"Features used: {feature_cols}")
    
    # Train model
    print("Starting model training...")
    model, training_history = train_model(model, train_loader, val_loader, 
                                         epochs=EPOCHS, lr=LR, 
                                         device=device, patience=PATIENCE)
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'comprehensive_model.pth')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")
    
    # Evaluate model
    print("Evaluating model performance...")
    predictions, targets, mae, rmse = evaluate_model(model, test_loader, device)
    
    logging.info(f"Test Performance: MAE={mae:.4f}°C, RMSE={rmse:.4f}°C")
    print(f"Test Performance: MAE={mae:.4f}°C, RMSE={rmse:.4f}°C")
    
    # Create comprehensive collector analysis
    print("Creating comprehensive collector analysis...")
    comprehensive_plot_path = create_comprehensive_collector_analysis(
        combined_data,
        test_df,
        predictions,
        targets,
        feature_cols,
        config_analysis,
        train_history_hours=TRAIN_HISTORY_WINDOW_HOURS,
        forecast_hours=FORECAST_WINDOW_HOURS)
    
    # Save results
    results = {
        'model_performance': {
            'mae': float(mae),
            'rmse': float(rmse),
            'training_history': training_history
        },
        'config_analysis': config_analysis,
        'feature_columns': feature_cols,
        'model_path': model_path,
        'plot_path': comprehensive_plot_path
    }
    
    results_path = os.path.join(OUTPUT_DIR, 'comprehensive_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final summary
    print("\n" + "="*70)
    print("COMPREHENSIVE CNN-LSTM ANALYSIS SUMMARY")
    print("="*70)
    print(f"Model Performance:")
    print(f"  MAE: {mae:.4f}°C")
    print(f"  RMSE: {rmse:.4f}°C")
    print(f"  Parameters: {trainable_params:,}")
    
    print(f"\nBHE Configurations Analyzed:")
    for config, analysis in config_analysis.items():
        print(f"  {config}:")
        print(f"    Records: {analysis['count']}")
        print(f"    Mean return temp: {analysis['avg_return_temp']:.2f}°C")
        print(f"    Mean temp difference: {analysis['avg_temp_diff']:.3f}°C")
    
    print(f"\nFiles generated in {OUTPUT_DIR}:")
    print(f"  - comprehensive_model.pth (trained CNN-LSTM model)")
    print(f"  - comprehensive_results.json (detailed results)")
    print(f"  - comprehensive_collector_analysis.png (multi-panel visualization)")
    print(f"  - collector_configuration_performance.png (focused collector comparison)")
    print(f"  - comprehensive_analysis.log (execution log)")
    
    # Define features and target
    feature_cols = ['supply_temp', 'flow_rate', 'power_kw', 'bhe_type_encoded']
    target_col = 'return_temp'
    
    # Prepare data for training
    model_data = combined_data[feature_cols + [target_col, 'Timestamp', 'bhe_type']].dropna()
    
    logging.info(f"Model data: {len(model_data)} records with {len(feature_cols)} features")
    
    # Time-based train/validation/test split
    latest_timestamp = model_data['Timestamp'].max()
    test_start = latest_timestamp - pd.Timedelta(days=FORECAST_WINDOW_DAYS)
    val_start = test_start - pd.Timedelta(days=VALIDATION_WINDOW_DAYS)

    train_df = model_data[model_data['Timestamp'] < val_start].copy()
    val_df = model_data[(model_data['Timestamp'] >= val_start) & (model_data['Timestamp'] < test_start)].copy()
    test_df = model_data[model_data['Timestamp'] >= test_start].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Time-based split produced an empty subset. Check that the dataset covers the requested windows."
        )

    logging.info(
        "Data splits (time-based): Train=%d (up to %s), Val=%d (%s to %s), Test=%d (from %s)",
        len(train_df),
        val_start,
        len(val_df),
        val_start,
        test_start,
        len(test_df),
        test_start
    )
    
    # Create datasets
    train_dataset = ComprehensiveDataset(train_df, SEQ_LEN, PRED_HORIZON, feature_cols, target_col)
    val_dataset = ComprehensiveDataset(val_df, SEQ_LEN, PRED_HORIZON, feature_cols, target_col, 
                                     train_dataset.mean, train_dataset.std)
    test_dataset = ComprehensiveDataset(test_df, SEQ_LEN, PRED_HORIZON, feature_cols, target_col,
                                      train_dataset.mean, train_dataset.std)
    
    # Initialize model for batch size optimization
    temp_model = ComprehensiveCNNLSTM(
        input_features=len(feature_cols),
        conv_channels=CONV_CHANNELS,
        kernel_size=KERNEL_SIZE,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Optimize batch size for GPU if using CUDA
    if device == 'cuda':
        sample_batch, _ = next(iter(DataLoader(train_dataset, batch_size=1)))
        optimal_batch_size = optimize_batch_size_for_gpu(temp_model, sample_batch, device)
        # Use the optimal batch size or stick with configured one if it's smaller
        actual_batch_size = min(BATCH_SIZE, optimal_batch_size)
        logging.info(f"Using batch size: {actual_batch_size} (configured: {BATCH_SIZE}, optimal: {optimal_batch_size})")
    else:
        actual_batch_size = BATCH_SIZE
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=actual_batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True, persistent_workers=True)
    
    # Initialize final model
    model = ComprehensiveCNNLSTM(
        input_features=len(feature_cols),
        conv_channels=CONV_CHANNELS,
        kernel_size=KERNEL_SIZE,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Test GPU before training
    if device == 'cuda':
        print("Testing GPU utilization...")
        peak_memory, utilization = test_gpu_utilization(model, train_loader, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model initialized: {total_params:,} parameters")
    logging.info(f"Model initialized with {total_params:,} parameters ({trainable_params:,} trainable)")
    logging.info(f"Features used: {feature_cols}")
    
    # Train model
    print("Starting model training...")
    logging.info("Starting model training...")
    model, history = train_model(model, train_loader, val_loader, EPOCHS, LR, device, PATIENCE)
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, "comprehensive_model.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")
    
    # Evaluate model
    print("Evaluating model performance...")
    logging.info("Evaluating model performance...")
    predictions, targets, mae, rmse = evaluate_model(model, test_loader, device)
    
    print(f"Test Performance: MAE={mae:.4f}°C, RMSE={rmse:.4f}°C")
    logging.info(f"Test Performance: MAE={mae:.4f}°C, RMSE={rmse:.4f}°C")

    # Create model performance comparison by collector type
    print("Creating model performance comparison...")
    model_plot_path, collector_metrics = create_model_performance_comparison(
        test_df, predictions, targets, config_analysis)

    # Create comprehensive visualization (includes all model performance metrics)
    print("Creating comprehensive collector analysis...")
    comprehensive_plot_path = create_comprehensive_collector_analysis(
        combined_data,
        test_df,
        predictions,
        targets,
        feature_cols,
        config_analysis,
        train_history_hours=TRAIN_HISTORY_WINDOW_HOURS,
        forecast_hours=FORECAST_WINDOW_HOURS)
    results = {
        'model_performance': {
            'overall_mae': float(mae),
            'overall_rmse': float(rmse),
            'collector_metrics': collector_metrics,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        },
        'training_history': {
            'train_losses': [float(x) for x in history['train_losses']],
            'val_losses': [float(x) for x in history['val_losses']]
        },
        'data_summary': {
            'total_records': len(combined_data),
            'train_records': len(train_df),
            'val_records': len(val_df), 
            'test_records': len(test_df),
            'features': feature_cols
        },
        'bhe_analysis': config_analysis,
        'plot_paths': {
            'comprehensive_analysis': comprehensive_plot_path,
            'model_performance': model_plot_path
        }
    }
    
    results_path = os.path.join(OUTPUT_DIR, "comprehensive_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\nCNN-LSTM ANALYSIS SUMMARY")
    print(f"Overall Model Performance:")
    print(f"  MAE: {mae:.4f}°C")
    print(f"  RMSE: {rmse:.4f}°C")
    print(f"  Parameters: {total_params:,}")
    
    print(f"\nCollector-Specific Model Performance:")
    for collector, metrics in collector_metrics.items():
        collector_name = collector.replace('_', ' ').title()
        print(f"  {collector_name}:")
        print(f"    MAE: {metrics['mae']:.4f}°C")
        print(f"    RMSE: {metrics['rmse']:.4f}°C")
        print(f"    Test samples: {metrics['count']:,}")
    
    print(f"\nBHE Configurations Analyzed:")
    
    for config, analysis in config_analysis.items():
        print(f"  {config}:")
        print(f"    Records: {analysis['count']}")
        print(f"    Mean return temp: {analysis['avg_return_temp']:.2f}°C")
        print(f"    Mean temp difference: {analysis['avg_temp_diff']:.3f}°C")
    
    print(f"\nFiles generated in {OUTPUT_DIR}:")
    print(f"  - comprehensive_model.pth (trained CNN-LSTM model)")
    print(f"  - comprehensive_results.json (detailed results)")
    print(f"  - comprehensive_collector_analysis.png (multi-panel visualization)")
    print(f"  - collector_configuration_performance_analysis.png (performance with moving averages)")
    print(f"  - model_performance_comparison.png (MAE/RMSE by collector type)")
    print(f"  - raw_data_collector_analysis.png (pre-model data analysis)")
    print(f"  - comprehensive_analysis.log (execution log)")

if __name__ == "__main__":
    main()