import os
import json
import logging
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional advanced analysis imports
try:
    from statsmodels.tsa.stattools import acf
except ImportError:
    print("statsmodels not available - using simplified autocorrelation")
    def acf(x, nlags=40, fft=True):
        return np.correlate(x, x, mode='full')[len(x)-1:len(x)+nlags]

#==============================================================================
# GEOTHERMAL BHE DEPTH SIGNAL ANALYSIS
#==============================================================================
"""
CNN-LSTM Model for BHE Depth Signal Analysis

Main Question: Can adding BHE depth as a signal improve outlet temperature 
prediction reliability, and how does predicted outlet temperature change with depth?

INPUT PARAMETERS (controlled and visible):
1. Inlet temperature, outdoor temperature 
2.  Flow rate (now using actual from CSV instead of calculated)
3. Well thermal resistance (0.09 mK/W)
4. Bore hole depth signal
5. Geothermal gradient (variable)

Validation: 650m well data import framework for future validation
"""

#------------------------------------------------------------------------------
# CONFIGURATION PARAMETERS
#------------------------------------------------------------------------------
CSV_PATH = os.environ.get(
    "CSV_PATH",
    os.path.join(os.path.dirname(__file__), 
                 "input/Borehole heat extraction complete field.csv"),
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", 
                           os.path.join(os.path.dirname(__file__), "output"))

# Model hyperparameters
SEQ_LEN = int(os.environ.get("SEQ_LEN", "48"))
PRED_HORIZON = int(os.environ.get("PRED_HORIZON", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "5")) # Model training epochs (steps), less is faster
LR = float(os.environ.get("LR", "1e-3"))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", "0.1"))
TEST_SPLIT = float(os.environ.get("TEST_SPLIT", "0.2"))

# CNN-LSTM architecture
CONV_CHANNELS = [int(x) for x in 
                 os.environ.get("CONV_CHANNELS", "32,32").split(",") 
                 if x.strip()]
KERNEL_SIZE = int(os.environ.get("KERNEL_SIZE", "3"))
LSTM_HIDDEN = int(os.environ.get("LSTM_HIDDEN", "64"))
LSTM_LAYERS = int(os.environ.get("LSTM_LAYERS", "2"))
DROPOUT = float(os.environ.get("DROPOUT", "0.1"))
PATIENCE = int(os.environ.get("PATIENCE", "16"))
USE_SCHEDULER = os.environ.get("USE_SCHEDULER", 
                              "false").lower() in {"1", "true", "yes"}

# Column names mapped to Borehole heat extraction complete field.csv
TIME_COL = "Timestamp"
# Updated column mappings for the new CSV structure
ACTUAL_FLOW_COL = "Flow rate to 120 boreholes [m³/h]"  # Actual volumetric flow rate from CSV
INLET_COL = "Supply temperature measure at energy meter [°C]"  # Supply temperature measure at energy meter
OUTLET_COL = "Return temperature measure at energy meter [°C]"  # Return temperature measure at energy meter
HEAT_EXTRACTION_COL = "Negative Heat extracion [kW] / Positive Heat rejection [kW]"  # Heat extraction/rejection
DEPTH_COL = "bore_depth_km"  # Borehole depth in km
# HX24 (Water-24% ethanol) properties for flow calculation
HX24_SPECIFIC_HEAT = 3600.0  # J/(kg·K)
HX24_DENSITY = 970.0  # kg/m³
HEAT_LOSS_EFF = 0.90 # Heat loss efficiency factor (assumed)
#OUTDOOR_TEMP_COL = "outside_t"  # Outdoor temperature
DELIVERY_POWER_COL = HEAT_EXTRACTION_COL  # Using heat extraction as power'
SUPPLY_TEMP_COL = "Supply temperature measured at external temperature sensor [°C]"  # well inlet temperature
RETURN_TEMP_COL = "Return temperature measured at external temperature sensor [°C]"  # well outlet temperature

# Well thermal properties
WELL_THERMAL_RESISTANCE = 0.09  # mK/W - well thermal resistance

# Geothermal parameters
GEOTHERMAL_GRADIENT_C_PER_KM = float(
    os.environ.get("GEOTHERMAL_GRADIENT_C_PER_KM", "27.0"))
SURFACE_BASELINE_C = float(os.environ.get("SURFACE_BASELINE_C", "10.0"))
REAL_WELL_DEPTH_KM = float(os.environ.get("REAL_WELL_DEPTH_KM", "0.30"))

#------------------------------------------------------------------------------
# VOLUMETRIC FLOW RATE CALCULATION - APPLICABLE FOR HX24 WORKING FLUID
#------------------------------------------------------------------------------
def calculate_flow_rate_hx24(df):
    """Calculate volumetric flow rate for HX24 working fluid - kept for comparison."""
    logging.info("Calculating volumetric flow rate for HX24 working fluid (for comparison)")
    
    supply_temp = pd.to_numeric(df[SUPPLY_TEMP_COL], errors='coerce')
    return_temp = pd.to_numeric(df[RETURN_TEMP_COL], errors='coerce')
    power_kw = pd.to_numeric(df[DELIVERY_POWER_COL], errors='coerce')
    
    delta_T = return_temp - supply_temp
    power_w = power_kw * 1000.0
    
    vol_flow_rate = np.where(
        np.abs(delta_T) > 0.5,
        np.abs(power_w) / (HX24_DENSITY * HX24_SPECIFIC_HEAT * np.abs(delta_T)) * HEAT_LOSS_EFF * 3600 , # from m³/s to m³/h
        np.nan
    )
    
    vol_flow_rate = pd.Series(vol_flow_rate).rolling(window=3, center=True, min_periods=1).mean().values  # Ensure non-negative flow rates and smoothing applied
    df['vol_flow_rate_calculated'] = vol_flow_rate  # Renamed to indicate calculated
    df['well_thermal_resistance_mK_per_W'] = WELL_THERMAL_RESISTANCE
    
    return df

#------------------------------------------------------------------------------
# LOGGING SETUP
#------------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "depth_analysis.log"), 
                           mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("DepthAnalysis")

#------------------------------------------------------------------------------
# DATASET CLASS
#------------------------------------------------------------------------------
class DepthAwareSequenceDataset(Dataset):
    """Dataset class that preserves depth signals during standardization."""
    
    def __init__(self, df, time_col, target, features, seq_len, horizon, 
                 mean=None, std=None, preserve_depth_signal=True):
        self.time = df[time_col].to_numpy()
        self.y = df[target].to_numpy(dtype=np.float32)
        self.X = df[features].to_numpy(dtype=np.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.features = features
        
        if mean is None or std is None:
            self.mean = self.X.mean(axis=0)
            self.std = self.X.std(axis=0) + 1e-8
        else:
            self.mean = mean
            self.std = std.copy()
        
        # Preserve depth signal variation
        if preserve_depth_signal:
            depth_related_indices = []
            for i, feat in enumerate(features):
                if any(keyword in feat.lower() for keyword in 
                      ['depth', 'geo_baseline', 'geo_gradient', 'flow']):
                    depth_related_indices.append(i)
            
            for idx in depth_related_indices:
                if 'depth' in features[idx].lower():
                    self.std[idx] = 0.5  # Inflate depth signal importance
                    logging.info(f"Enhanced depth signal preservation: std={self.std[idx]:.3f}")
                elif self.std[idx] < 0.05:
                    self.std[idx] = 0.05
                    logging.info(f"Enhanced signal preservation for '{features[idx]}'")
        
        self.X = (self.X - self.mean) / self.std
        
        # Calculate valid sequence indices
        self.valid_idx = []
        max_start = len(self.X) - (seq_len + horizon)
        for i in range(max(0, max_start) + 1):
            self.valid_idx.append(i)

    def __len__(self): 
        return len(self.valid_idx)

    def __getitem__(self, idx):
        i = self.valid_idx[idx]
        seq = self.X[i : i + self.seq_len]
        target = self.y[i + self.seq_len + self.horizon - 1]
        seq_ch_first = torch.from_numpy(seq).float().transpose(0, 1)
        return seq_ch_first, torch.tensor(target, dtype=torch.float32)

#------------------------------------------------------------------------------
# CNN-LSTM MODEL
#------------------------------------------------------------------------------
class DepthAwareHybridCNNLSTM(nn.Module):
    """CNN-LSTM hybrid model with depth-aware processing."""
    
    def __init__(self, in_channels, conv_channels=(32,32), kernel_size=3,
                 lstm_hidden=64, lstm_layers=2, dropout=0.1, 
                 depth_feature_indices=None):
        super().__init__()
        channels = [in_channels] + list(conv_channels)
        
        # CNN layers
        convs = []
        for i in range(len(channels) - 1):
            convs += [
                nn.Conv1d(channels[i], channels[i+1], kernel_size, 
                         padding=kernel_size//2),
                nn.ReLU(),
            ]
        self.conv = nn.Sequential(*convs)
        
        # Optional depth attention
        self.depth_attention = None
        if depth_feature_indices and len(depth_feature_indices) > 0:
            self.depth_attention = nn.MultiheadAttention(
                embed_dim=channels[-1], num_heads=4, dropout=dropout, 
                batch_first=False
            )
            logging.info(f"Depth attention enabled for {len(depth_feature_indices)} features")
        
        # LSTM layers
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=channels[-1], hidden_size=lstm_hidden,
                           num_layers=lstm_layers, batch_first=False)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        x = self.conv(x)
        
        if self.depth_attention is not None:
            x_permuted = x.permute(2, 0, 1)
            x_attended, _ = self.depth_attention(x_permuted, x_permuted, x_permuted)
            x = x_attended.permute(1, 2, 0)
        
        x = self.dropout(x)
        x = x.permute(2, 0, 1)
        out, _ = self.lstm(x)
        last = out[-1]
        y = self.fc(last).squeeze(-1)
        return y

#------------------------------------------------------------------------------
# ANALYSIS FUNCTIONS
#------------------------------------------------------------------------------
def depth_sensitivity_analysis(model, test_df, features_with, tr_ds, device, target):
    """Analyze how outlet temperature changes with depth."""
    
    print("\nDEPTH SENSITIVITY ANALYSIS")
    print("="*50)
    
    # Check depth signal preservation
    depth_idx = features_with.index(DEPTH_COL)
    print(f"Depth range: [{test_df[DEPTH_COL].min():.3f}, {test_df[DEPTH_COL].max():.3f}] km")
    print(f"Standardization - mean: {tr_ds.mean[depth_idx]:.6f}, std: {tr_ds.std[depth_idx]:.6f}")
    
    # Test depth response
    model.eval()
    depths_test = np.linspace(0.2, 1.5, 8)  # 200m to 1500m
    responses = []
    
    print(f"Testing depth range: {depths_test[0]:.1f}km to {depths_test[-1]:.1f}km")
    
    for depth in depths_test:
        cf = test_df.head(50).copy()
        cf[DEPTH_COL] = depth
        cf["geo_baseline_T_at_depth"] = (SURFACE_BASELINE_C + 
                                        GEOTHERMAL_GRADIENT_C_PER_KM * depth)
        
        ds = DepthAwareSequenceDataset(cf, TIME_COL, target, features_with, 
                                     SEQ_LEN, PRED_HORIZON, 
                                     mean=tr_ds.mean, std=tr_ds.std)
        dl = DataLoader(ds, batch_size=len(ds), shuffle=False)
        
        with torch.no_grad():
            _, preds, _, _ = evaluate_model(model, dl, device)
        
        avg_pred = preds.mean()
        responses.append(avg_pred)
        print(f"  Depth {depth:.2f}km -> Outlet temp: {avg_pred:.4f}C")
    
    # Calculate sensitivity
    sensitivity = ((responses[-1] - responses[0]) / (depths_test[-1] - depths_test[0]))
    print(f"\nDepth sensitivity: {sensitivity:.4f} C/km")
    
    if sensitivity > 0:
        print("Positive: deeper wells show higher outlet temperatures")
    else:
        print("Negative: deeper wells show lower outlet temperatures")
    
    return depths_test, responses, sensitivity

def setup_650m_validation_framework():
    """Framework for 650m validation data using actual Well 1 from research boreholes."""
    
    validation_csv_path = os.path.join(os.path.dirname(__file__), 
                                      "input/Energi meters research boreholes.csv")
    
    if os.path.exists(validation_csv_path):
        print(f"\nLoading 650m validation data from {validation_csv_path}")
        try:
            real_650m_df = pd.read_csv(validation_csv_path, sep=';', decimal=',')
            real_650m_df[TIME_COL] = pd.to_datetime(real_650m_df[TIME_COL], format='%d.%m.%Y %H:%M', errors="coerce")
            real_650m_df = real_650m_df.dropna(subset=[TIME_COL]).reset_index(drop=True)
            # Use only Well 2 columns
            well2_cols = ['Return temperature2 [°C]', 'supply temperature2 [°C]', 'Heat extracion / rejection2 [kW]']

            # Check if Well 2 columns exist
            if all(col in real_650m_df.columns for col in well2_cols):
                # Keep only timestamp and Well 2 data
                well2_data = real_650m_df[[TIME_COL] + well2_cols].copy()
                well2_data = well2_data.dropna()

                print(f"Loaded {len(well2_data)} Well 2 measurements from 650m")
                return well2_data
            else:
                print(f"Missing Well 2 columns in CSV")
                return None
        except Exception as e:
            print(f"Error loading 650m data: {e}")
            return None
    else:
        print(f"\nCreating 650m validation framework with proper extrapolation")
        
        depth_change = 0.65 - REAL_WELL_DEPTH_KM  # 650m - 300m = 350m
        temp_increase = GEOTHERMAL_GRADIENT_C_PER_KM * depth_change
        
        print(f"Extrapolating from {REAL_WELL_DEPTH_KM}km to 0.65km")
        print(f"Depth increase: {depth_change:.2f}km")
        print(f"Expected temperature increase: {temp_increase:.2f}°C")
        
        return None

#------------------------------------------------------------------------------
# TRAINING & EVALUATION
#------------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs, lr, device, 
               patience, use_scheduler, log_prefix=""):
    """Training pipeline."""
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(1,patience//2)
    ) if use_scheduler else None
    
    best_val = float("inf")
    best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    bad_epochs = 0
    hist = {"train_loss":[], "val_loss":[]}
    
    for ep in range(1, epochs+1):
        # Training
        model.train()
        tr_loss = 0.0
        for Xb, yb in tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [train]", leave=False):
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            yh = model(Xb)
            loss = crit(yh, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * Xb.size(0)
        tr_loss /= max(1, len(train_loader.dataset))
        
        # Validation
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for Xb, yb in tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [val]", leave=False):
                Xb, yb = Xb.to(device), yb.to(device)
                yh = model(Xb)
                va_loss += crit(yh, yb).item() * Xb.size(0)
        va_loss /= max(1, len(val_loader.dataset))
        
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        logging.info(f"{log_prefix}Epoch {ep}: train_loss={tr_loss:.5f}, val_loss={va_loss:.5f}")
        
        if scheduler:
            scheduler.step(va_loss)
        
        if va_loss + 1e-9 < best_val:
            best_val = va_loss
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logging.info(f"Early stopping at epoch {ep}")
                break
    
    model.load_state_dict(best_state)
    return model, hist

def evaluate_model(model, data_loader, device="cpu"):
    """Model evaluation."""
    model.eval()
    preds, trues = [], []
    
    with torch.no_grad():
        for Xb, yb in tqdm(data_loader, desc="Evaluating", leave=False):
            Xb, yb = Xb.to(device), yb.to(device)
            yh = model(Xb)
            preds.append(yh.cpu().numpy())
            trues.append(yb.cpu().numpy())
    
    preds = np.concatenate(preds) if preds else np.array([])
    trues = np.concatenate(trues) if trues else np.array([])
    
    mae = float(np.mean(np.abs(preds - trues))) if len(preds) else float("nan")
    rmse = float(np.sqrt(np.mean((preds - trues)**2))) if len(preds) else float("nan")
    
    return trues, preds, mae, rmse

def make_loaders_enhanced(features, tr_df, va_df, test_df, target):
    """Create data loaders."""
    tr_ds = DepthAwareSequenceDataset(tr_df, TIME_COL, target, features, 
                                     SEQ_LEN, PRED_HORIZON)
    va_ds = DepthAwareSequenceDataset(va_df, TIME_COL, target, features, 
                                     SEQ_LEN, PRED_HORIZON, 
                                     mean=tr_ds.mean, std=tr_ds.std)
    te_ds = DepthAwareSequenceDataset(test_df, TIME_COL, target, features, 
                                     SEQ_LEN, PRED_HORIZON, 
                                     mean=tr_ds.mean, std=tr_ds.std)
    return (DataLoader(tr_ds, BATCH_SIZE, shuffle=True),
            DataLoader(va_ds, BATCH_SIZE),
            DataLoader(te_ds, BATCH_SIZE),
            tr_ds, te_ds)

#==============================================================================
# MAIN EXECUTION
#==============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Starting depth analysis on device: {device}")
    
    # Load data
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Data file not found: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH, sep=';', decimal=',')
    logging.info(f"Loaded {len(df)} records with {len(df.columns)} features")
    
    # Process timestamp
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], format='%d.%m.%Y %H:%M', errors="coerce")
    df = df.sort_values(TIME_COL).dropna(subset=[TIME_COL]).reset_index(drop=True)

    # Calculate volumetric flow rate using HX24 properties and add well thermal resistance
    df = calculate_flow_rate_hx24(df)
    df['well_thermal_resistance_mK_per_W'] = WELL_THERMAL_RESISTANCE

    # Compare actual vs calculated volumetric flow rates
    actual_fr = df[ACTUAL_FLOW_COL].values
    calculated_fr = df['vol_flow_rate_calculated'].values
    
    print(f"\nFlow Rate (FR) Comparison:")
    print(f"Actual FR - Mean: {np.nanmean(actual_fr):.2f}, Std: {np.nanstd(actual_fr):.2f}")
    print(f"Calculated FR - Mean: {np.nanmean(calculated_fr):.2f}, Std: {np.nanstd(calculated_fr):.2f}")

    # Plot volumetric flow rate comparison
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    valid_indices = ~(np.isnan(actual_fr) | np.isnan(calculated_fr))
    plt.plot(actual_fr[valid_indices][:1000], label='Actual Flow Rate', alpha=0.7)
    plt.plot(calculated_fr[valid_indices][:1000], label='Calculated Flow Rate', alpha=0.7)
    plt.title('Flow Rate Comparison (First 1000 valid points)')
    plt.xlabel('Time Steps')
    plt.ylabel('Flow Rate [m3/h]')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    valid_data = ~np.isnan(actual_fr) & ~np.isnan(calculated_fr)
    if np.sum(valid_data) > 0:
        plt.scatter(actual_fr[valid_data], calculated_fr[valid_data], alpha=0.5)
        min_val = min(np.nanmin(actual_fr), np.nanmin(calculated_fr))
        max_val = max(np.nanmax(actual_fr), np.nanmax(calculated_fr))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.title('Actual vs Calculated Flow Rate')
    plt.xlabel('Actual Flow Rate [m3/h]')
    plt.ylabel('Calculated Flow Rate [m3/h]')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "flow_rate_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Add depth and geothermal features
    if DEPTH_COL not in df.columns:
        df[DEPTH_COL] = REAL_WELL_DEPTH_KM
        logging.info(f"Added constant depth: {REAL_WELL_DEPTH_KM} km")
    
    df["geo_baseline_T_at_depth"] = (SURFACE_BASELINE_C + 
                                    GEOTHERMAL_GRADIENT_C_PER_KM * df[DEPTH_COL])
    if "geo_gradient_C_per_km" not in df.columns:
        df["geo_gradient_C_per_km"] = GEOTHERMAL_GRADIENT_C_PER_KM
    
    # # Add outdoor temperature (dummy if not available)
    # if OUTDOOR_TEMP_COL not in df.columns:
    #     df[OUTDOOR_TEMP_COL] = 10.0  # Default outdoor temperature
    #     logging.info("Added default outdoor temperature: 10.0°C")

    # # Feature selection - CONTROLLED PARAMETERS ONLY
    target = OUTLET_COL
    
    # CONTROLLED FEATURES ONLY - exactly 6 parameters
    controlled_features = [
        INLET_COL,                          # Inlet temperature
        # OUTDOOR_TEMP_COL,                   # Outdoor temperature  
        ACTUAL_FLOW_COL,             # Vol flow rate (actual from CSV)
        'well_thermal_resistance_mK_per_W', # Well thermal resistance (0.09 mK/W)
    ]
    
    # Geothermal parameters for depth analysis
    geo_features = [
        DEPTH_COL,                         # Bore hole depth (300m baseline)
        "geo_gradient_C_per_km"            # Geothermal gradient (variable)
    ]
    
    # Features WITHOUT depth (300m baseline model)
    features_without_depth = controlled_features.copy()
    
    # Features WITH depth (for extrapolation to 650m)
    features_with_depth = controlled_features + geo_features

    # Features WITHOUT flow rate
    features_without_fr = [f for f in controlled_features if f != ACTUAL_FLOW_COL] + geo_features
    
    # Validate features exist
    missing_features = [f for f in features_with_depth if f not in df.columns]
    if missing_features:
        logging.error(f"Missing controlled features: {missing_features}")
        raise RuntimeError(f"Missing controlled features: {missing_features}")
    
    logging.info("CONTROLLED PARAMETERS:")
    logging.info(f"  Core features (4): {controlled_features}")
    logging.info(f"  Geo features (2): {geo_features}")
    logging.info(f"  Total controlled: {len(features_with_depth)} parameters")
    logging.info(f"  Features without FR: {len(features_without_fr)} parameters")
    
    # Clean data - only keep records with all controlled parameters
    df = df.dropna(subset=features_with_depth + [target]).reset_index(drop=True)
    logging.info(f"Clean dataset: {len(df)} records with controlled parameters")
    
    # Data splitting
    N = len(df)
    test_start = int(N * (1.0 - TEST_SPLIT))
    test_start = max(test_start, SEQ_LEN + PRED_HORIZON)
    
    train_df = df.iloc[:test_start].copy()
    test_df = df.iloc[test_start:].copy()
    
    val_size = max(1, int(len(train_df) * VAL_SPLIT))
    tr_df = train_df.iloc[:-val_size].copy()
    va_df = train_df.iloc[-val_size:].copy()
    
    logging.info(f"Data split - Train: {len(tr_df)}, Val: {len(va_df)}, Test: {len(test_df)}")
    
    # Model configuration
    depth_feature_indices = [i for i, f in enumerate(features_with_depth) 
                            if f in [DEPTH_COL, "geo_gradient_C_per_km"]]
    
    logging.info(f"Depth feature indices: {depth_feature_indices}")
    
    # Train model WITH depth (300m baseline + depth signal for extrapolation)
    logging.info("Training model WITH depth features (for 300m->650m extrapolation)")
    tr_loader_with, va_loader_with, te_loader_with, tr_ds, te_ds = make_loaders_enhanced(
        features_with_depth, tr_df, va_df, test_df, target)
    model_with = DepthAwareHybridCNNLSTM(len(features_with_depth), CONV_CHANNELS, 
                                        KERNEL_SIZE, LSTM_HIDDEN, LSTM_LAYERS, 
                                        DROPOUT, depth_feature_indices).to(device)
    model_with, hist_with = train_model(model_with, tr_loader_with, 
                                       va_loader_with, EPOCHS, LR, device, 
                                       PATIENCE, USE_SCHEDULER, "with_depth|")
    y_true_with, y_pred_with, mae_with, rmse_with = evaluate_model(
        model_with, te_loader_with, device)
    logging.info(f"Model WITH depth (300m baseline) - MAE: {mae_with:.4f}, RMSE: {rmse_with:.4f}")

    # Train model WITHOUT depth (300m only, no extrapolation capability)
    logging.info("Training model WITHOUT depth features (300m only)")
    tr_loader_no, va_loader_no, te_loader_no, tr_ds_no, te_ds_no = make_loaders_enhanced(
        features_without_depth, tr_df, va_df, test_df, target)
    model_no = DepthAwareHybridCNNLSTM(len(features_without_depth), CONV_CHANNELS, 
                                      KERNEL_SIZE, LSTM_HIDDEN, LSTM_LAYERS, 
                                      DROPOUT).to(device)
    model_no, hist_no = train_model(model_no, tr_loader_no, va_loader_no, 
                                   EPOCHS, LR, device, PATIENCE, USE_SCHEDULER, 
                                   "no_depth|")
    y_true_no, y_pred_no, mae_no, rmse_no = evaluate_model(
        model_no, te_loader_no, device)
    logging.info(f"Model WITHOUT depth (300m only) - MAE: {mae_no:.4f}, RMSE: {rmse_no:.4f}")

    # Train model WITHOUT flow rate
    logging.info("Training model WITHOUT flow rate")
    tr_loader_no_fr, va_loader_no_fr, te_loader_no_fr, tr_ds_no_fr, te_ds_no_fr = make_loaders_enhanced(
        features_without_fr, tr_df, va_df, test_df, target)
    model_no_fr = DepthAwareHybridCNNLSTM(len(features_without_fr), CONV_CHANNELS, 
                                          KERNEL_SIZE, LSTM_HIDDEN, LSTM_LAYERS, 
                                          DROPOUT).to(device)
    model_no_fr, hist_no_fr = train_model(model_no_fr, tr_loader_no_fr, va_loader_no_fr, 
                                          EPOCHS, LR, device, PATIENCE, USE_SCHEDULER, 
                                          "no_fr")
    y_true_no_fr, y_pred_no_fr, mae_no_fr, rmse_no_fr = evaluate_model(
        model_no_fr, te_loader_no_fr, device)
    logging.info(f"Model WITHOUT flow rate - MAE: {mae_no_fr:.4f}, RMSE: {rmse_no_fr:.4f}")

    # Depth sensitivity analysis
    depths, responses, sensitivity = depth_sensitivity_analysis(
        model_with, test_df, features_with_depth, tr_ds, device, target)

    # 650m extrapolation analysis
    logging.info("Performing 650m extrapolation analysis")
    cf_650m = test_df.copy()
    cf_650m[DEPTH_COL] = 0.65  # 650m depth
    cf_650m["geo_baseline_T_at_depth"] = (SURFACE_BASELINE_C + 
                                         GEOTHERMAL_GRADIENT_C_PER_KM * 0.65)
    
    ds_650m = DepthAwareSequenceDataset(cf_650m, TIME_COL, target, features_with_depth, 
                                       SEQ_LEN, PRED_HORIZON, 
                                       mean=tr_ds.mean, std=tr_ds.std)
    dl_650m = DataLoader(ds_650m, BATCH_SIZE)
    _, y_pred_650m, mae_650m, rmse_650m = evaluate_model(model_with, dl_650m, device)
    
    logging.info(f"650m extrapolation - MAE: {mae_650m:.4f}, RMSE: {rmse_650m:.4f}")
    
    # Temperature increase from 300m to 650m
    temp_increase_predicted = np.mean(y_pred_650m - y_pred_with)
    logging.info(f"Predicted temperature increase (300m->650m): {temp_increase_predicted:.3f}°C")

    # Controlled parameter importance analysis
    def controlled_parameter_importance():
        """Analyze importance of controlled parameters."""
        importance_scores = {}
        
        # Feature names for importance analysis
        param_names = {
            INLET_COL: 'Inlet Temperature',
            # OUTDOOR_TEMP_COL: 'Outdoor Temperature', 
            ACTUAL_FLOW_COL: 'Flow Rate',
            'well_thermal_resistance_mK_per_W': 'Thermal Resistance',
            DEPTH_COL: 'Bore Depth',
            'geo_gradient_C_per_km': 'Geothermal Gradient'
        }
        
        # Calculate feature importance based on standard deviation impact
        for i, feature in enumerate(features_with_depth):
            std_val = tr_ds.std[i]
            importance_scores[param_names.get(feature, feature)] = float(std_val)
        
        return importance_scores

    param_importance = controlled_parameter_importance()

    # Comprehensive metrics
    metrics = {
        "controlled_parameters": {
            "inlet_temperature": INLET_COL,
            # "outdoor_temperature": OUTDOOR_TEMP_COL,
            "vol_flow_rate_actual": ACTUAL_FLOW_COL,
            "well_thermal_resistance_mK_per_W": WELL_THERMAL_RESISTANCE,
            "bore_depth_km": REAL_WELL_DEPTH_KM,
            "geothermal_gradient_C_per_km": GEOTHERMAL_GRADIENT_C_PER_KM
        },
        "model_performance_300m": {
            "with_depth": {"MAE": float(mae_with), "RMSE": float(rmse_with)},
            "without_depth": {"MAE": float(mae_no), "RMSE": float(rmse_no)},
            "without_flow": {"MAE": float(mae_no_fr), "RMSE": float(rmse_no_fr)},
            "improvement_MAE_depth": float(mae_no - mae_with),
            "improvement_RMSE_depth": float(rmse_no - rmse_with),
            "improvement_MAE_fr": float(mae_no_fr - mae_with),
            "improvement_RMSE_fr": float(rmse_no_fr - rmse_with)
        },
        "model_performance_650m": {
            "extrapolated": {"MAE": float(mae_650m), "RMSE": float(rmse_650m)},
            "temperature_increase_300m_to_650m": float(temp_increase_predicted)
        },
        "depth_analysis": {
            "sensitivity_C_per_km": float(sensitivity),
            "baseline_depth_km": REAL_WELL_DEPTH_KM,
            "extrapolation_depth_km": 0.65
        },
        "parameter_importance": param_importance,
        "feature_counts": {
            "controlled_features": len(controlled_features),
            "geo_features": len(geo_features),
            "total": len(features_with_depth),
            "without_fr": len(features_without_fr)
        }
    }

    with open(os.path.join(OUTPUT_DIR, "controlled_analysis_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Calculate test times for plotting
    test_times = test_df[TIME_COL].iloc[SEQ_LEN+PRED_HORIZON-1:].reset_index(drop=True)
    
    # Load actual 650m Well 1 data for validation
    real_650m_well1 = setup_650m_validation_framework()
    
    # SEPARATE PLOT 1: 650m Counterfactual Analysis with Well 1 Validation
    plt.figure(figsize=(15, 8))
    plt.plot(test_times, y_true_with, label="Actual (300m)", linewidth=2, color='blue')
    plt.plot(test_times, y_pred_650m, label="Predicted @ 650m", linewidth=2, color='red')
    plt.plot(test_times, y_pred_with, label="Predicted @ 300m", linewidth=2, color='green')
    
    # Add actual 650m Well 1 data if available
    if real_650m_well1 is not None and len(real_650m_well1) > 0:
        well1_data = real_650m_well1[real_650m_well1[TIME_COL].between(test_times.min(), test_times.max())]
        if len(well1_data) > 0:
            plt.plot(well1_data[TIME_COL], well1_data['Return temperature2 [°C]'].rolling(window=20, center=True).mean(), 
                label="Actual Well 2 (650m)", linewidth=2, color='purple', alpha=0.5, linestyle=':', zorder=1)

    plt.ylabel("Outlet Temperature (°C)", fontsize=12)
    plt.xlabel("Time", fontsize=12)
    plt.title("650m Depth Analysis: Extrapolated vs Actual Well 2", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add temperature difference annotation
    temp_diff = np.mean(y_pred_650m - y_pred_with)
    plt.text(0.02, 0.98, f'Avg. Temperature Increase: {temp_diff:.3f}°C', 
             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "650m_counterfactual_analysis.png"), 
                dpi=200, bbox_inches='tight')
    plt.close()
    
    # SEPARATE PLOT 2: Model Comparison Including Flow Rate Analysis
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Depth Response
    plt.subplot(2, 3, 1)
    plt.plot(depths, responses, marker='o', linewidth=2, markersize=8, color='purple')
    plt.axhline(y=np.mean(y_pred_with), color='green', linestyle='--', alpha=0.7, label='300m baseline')
    plt.axhline(y=np.mean(y_pred_650m), color='red', linestyle='--', alpha=0.7, label='650m extrapolation')
    plt.xlabel('Depth (km)', fontsize=11)
    plt.ylabel('Outlet Temperature (°C)', fontsize=11)
    plt.title(f'Depth Response: {sensitivity:.3f} °C/km', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2. Controlled Parameter Importance
    plt.subplot(2, 3, 2)
    param_names = list(param_importance.keys())
    param_values = list(param_importance.values())
    
    colors = ['lightcoral', 'skyblue', 'lightgreen', 'gold', 'plum', 'orange']
    bars = plt.barh(param_names, param_values, color=colors[:len(param_names)])
    plt.xlabel('Relative Importance (Std Dev)', fontsize=11)
    plt.title('Controlled Parameter Importance', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # 3. Performance Comparison (MAE/RMSE)
    plt.subplot(2, 3, 3)
    scenarios = ['300m\n(no depth)', '300m\n(with depth)', '300m\n(no FR)', '650m\n(extrapolated)']
    mae_values = [mae_no, mae_with, mae_no_fr, mae_650m]
    rmse_values = [rmse_no, rmse_with, rmse_no_fr, rmse_650m]

    x = np.arange(len(scenarios))
    width = 0.35
    
    plt.bar(x - width/2, mae_values, width, label='MAE', color='steelblue')
    plt.bar(x + width/2, rmse_values, width, label='RMSE', color='orange')
    
    plt.xlabel('Model Scenario', fontsize=11)
    plt.ylabel('Error', fontsize=11)
    plt.title('Performance Comparison', fontsize=12, fontweight='bold')
    plt.xticks(x, scenarios, fontsize=9)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 4. Temperature Increase Analysis
    plt.subplot(2, 3, 4)
    depth_comparison = ['300m Baseline', '650m Extrapolated']
    temp_means = [np.mean(y_pred_with), np.mean(y_pred_650m)]
    temp_stds = [np.std(y_pred_with), np.std(y_pred_650m)]
    
    plt.bar(depth_comparison, temp_means, yerr=temp_stds, capsize=5, 
            color=['green', 'red'], alpha=0.7)
    plt.ylabel('Mean Outlet Temperature (°C)', fontsize=11)
    plt.title(f'Temperature Increase: {temp_increase_predicted:.3f}°C', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add temperature increase annotation
    plt.annotate(f'+{temp_increase_predicted:.3f}°C', 
                xy=(1, temp_means[1]), xytext=(0.5, temp_means[1] + 0.1),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=12, fontweight='bold', ha='center')
    
    # 5. Flow Rate Impact Analysis
    plt.subplot(2, 3, 5)
    fr_comparison = ['With FR', 'Without FR']
    fr_mae_values = [mae_with, mae_no_fr]
    fr_rmse_values = [rmse_with, rmse_no_fr]

    x_fr = np.arange(len(fr_comparison))
    plt.bar(x_fr - width/2, fr_mae_values, width, label='MAE', color='lightblue')
    plt.bar(x_fr + width/2, fr_rmse_values, width, label='RMSE', color='lightsalmon')

    plt.xlabel('Flow Rate Usage', fontsize=11)
    plt.ylabel('Error', fontsize=11)
    plt.title('Flow Rate Impact', fontsize=12, fontweight='bold')
    plt.xticks(x_fr, fr_comparison, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 6. Model Performance Summary
    plt.subplot(2, 3, 6)
    improvement_fr = mae_no_fr - mae_with
    improvement_depth = mae_no - mae_with

    improvements = ['Depth Signal', 'Flow Rate']
    improvement_values = [improvement_depth, improvement_fr]
    colors_imp = ['purple', 'cyan']
    
    bars = plt.bar(improvements, improvement_values, color=colors_imp, alpha=0.7)
    plt.ylabel('MAE Improvement', fontsize=11)
    plt.title('Feature Impact Summary', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, improvement_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comprehensive_analysis_with_fr.png"), 
                dpi=200, bbox_inches='tight')
    plt.close()

    # Save model
    torch.save(model_with.state_dict(), 
               os.path.join(OUTPUT_DIR, "controlled_depth_model.pth"))
    
    logging.info("Controlled parameter analysis complete")
    logging.info(f"Controlled parameters: {len(features_with_depth)}")
    logging.info(f"300m MAE: {mae_with:.4f}, 650m MAE: {mae_650m:.4f}")
    logging.info(f"Temperature increase (300m->650m): {temp_increase_predicted:.3f}°C")
    logging.info(f"Depth sensitivity: {sensitivity:.4f} C/km")

    if device == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("CONTROLLED PARAMETER DEPTH ANALYSIS COMPLETE")
    print("="*80)
    print(f"CONTROLLED PARAMETERS (6 total):")
    print(f"   1. Inlet Temperature: {INLET_COL}")
    # print(f"   2. Outdoor Temperature: {OUTDOOR_TEMP_COL}")
    print(f"   3. Flow Rate (Actual): {ACTUAL_FLOW_COL}")
    print(f"   4. Thermal Resistance: {WELL_THERMAL_RESISTANCE} mK/W")
    print(f"   5. Bore Depth: {DEPTH_COL}")
    print(f"   6. Geothermal Gradient: {GEOTHERMAL_GRADIENT_C_PER_KM} C/km")
    print(f"")
    print(f"RESULTS:")
    print(f"   300m baseline MAE: {mae_with:.4f}")
    print(f"   650m extrapolated MAE: {mae_650m:.4f}")
    print(f"   Without Flow Rate MAE: {mae_no_fr:.4f}")
    print(f"   Temperature increase (300m→650m): {temp_increase_predicted:.3f}°C")
    print(f"   Depth sensitivity: {sensitivity:.4f} °C/km")
    print(f"   Model improvement with depth: {metrics['model_performance_300m']['improvement_MAE_depth']:.4f} MAE")
    print(f"   Model improvement with FR: {metrics['model_performance_300m']['improvement_MAE_fr']:.4f} MAE")
    print(f"")
    print(f"VALIDATION FRAMEWORK:")
    print(f"   - 650m predictions ready for validation with real data")
    print(f"   - Controlled parameters ensure reproducible results")
    print(f"   - Flow rate comparison available")
    print(f"   - All outputs saved to: {OUTPUT_DIR}")
    print("="*80)

    # Depth feature debug code:
    depth_idx = features_with_depth.index(DEPTH_COL)
    print(f"Depth feature index: {depth_idx}")
    print(f"Training depth stats - mean: {tr_ds.mean[depth_idx]:.6f}, std: {tr_ds.std[depth_idx]:.6f}")
    print(f"300m normalized: {(0.30 - tr_ds.mean[depth_idx]) / tr_ds.std[depth_idx]:.6f}")
    print(f"650m normalized: {(0.65 - tr_ds.mean[depth_idx]) / tr_ds.std[depth_idx]:.6f}")