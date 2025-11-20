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
2. Mass flow rate (calculated from power and deltaT)
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
                 "datapunkter_expanded_en.csv"),
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", 
                           os.path.join(os.path.dirname(__file__), "output"))

# Model hyperparameters
SEQ_LEN = int(os.environ.get("SEQ_LEN", "48"))
PRED_HORIZON = int(os.environ.get("PRED_HORIZON", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
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

# Column names mapped to combined_ede_mapped_v3.csv
TIME_COL = "Timestamp"
INLET_COL = "Borehole field energy meter — Supply temperature |°C|"  # Supply temperature as inlet
OUTLET_COL = "Borehole field energy meter — Return temperature |°C|"  # Production return temperature as outlet
DEPTH_COL = "bore_depth_km"

# HX24 (Water-24% ethanol) properties for mass flow calculation
HX24_SPECIFIC_HEAT = 3800.0  # J/(kg·K)
OUTDOOR_TEMP_COL = "outside_t"  # Outdoor temperature
DELIVERY_POWER_COL = "Borehole field energy meter — Power |kW|"  # Power obtained from well
SUPPLY_TEMP_COL = "Borehole field energy meter — Supply temperature |°C|"  # well inlet temperature
RETURN_TEMP_COL = "Borehole field energy meter — Return temperature |°C|"  # well outlet temperature

# Well thermal properties
WELL_THERMAL_RESISTANCE = 0.09  # mK/W - well thermal resistance

# Geothermal parameters
GEOTHERMAL_GRADIENT_C_PER_KM = float(
    os.environ.get("GEOTHERMAL_GRADIENT_C_PER_KM", "27.0"))
SURFACE_BASELINE_C = float(os.environ.get("SURFACE_BASELINE_C", "8.0"))
REAL_WELL_DEPTH_KM = float(os.environ.get("REAL_WELL_DEPTH_KM", "0.30"))

#------------------------------------------------------------------------------
# MASS FLOW RATE CALCULATION - SIMPLIFIED FOR HX24 WORKING FLUID
#------------------------------------------------------------------------------
def calculate_mass_flow_rate_hx24(df):
    """Calculate mass flow rate for HX24 working fluid."""
    logging.info("Calculating mass flow rate for HX24 working fluid")
    
    supply_temp = df[SUPPLY_TEMP_COL].fillna(df[INLET_COL])
    return_temp = df[RETURN_TEMP_COL].fillna(df[OUTLET_COL])
    power_kw = df[DELIVERY_POWER_COL]
    
    delta_T = return_temp - supply_temp
    power_w = power_kw * 1000.0
    
    mass_flow_rate = np.where(
        np.abs(delta_T) > 0.5,
        power_w / (HX24_SPECIFIC_HEAT * delta_T),
        np.nan
    )
    
    mass_flow_rate = np.abs(mass_flow_rate)
    df['mass_flow_hx24_kg_s'] = mass_flow_rate
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
                      ['depth', 'geo_baseline', 'geo_gradient', 'mass_flow']):
                    depth_related_indices.append(i)
            
            for idx in depth_related_indices:
                if 'depth' in features[idx].lower():
                    self.std[idx] = 0.3  # Changed from 0.05 to 0.3
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
    """Framework for future 650m validation data with proper extrapolation."""
    
    validation_csv_path = os.path.join(os.path.dirname(__file__), 
                                      "real_650m_validation_data.csv")
    
    if os.path.exists(validation_csv_path):
        print(f"\nLoading 650m validation data from {validation_csv_path}")
        try:
            real_650m_df = pd.read_csv(validation_csv_path)
            real_650m_df[TIME_COL] = pd.to_datetime(real_650m_df[TIME_COL])
            real_650m_df = real_650m_df.dropna(subset=[TIME_COL]).reset_index(drop=True)
            print(f"Loaded {len(real_650m_df)} 650m measurements")
            return real_650m_df
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
    
    df = pd.read_csv(CSV_PATH)
    logging.info(f"Loaded {len(df)} records with {len(df.columns)} features")
    
    # Process timestamp
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.sort_values(TIME_COL).dropna(subset=[TIME_COL]).reset_index(drop=True)
    
    # Calculate mass flow rate using HX24 properties and add well thermal resistance
    # df = calculate_mass_flow_rate_hx24(df)
    df['well_thermal_resistance_mK_per_W'] = WELL_THERMAL_RESISTANCE
    # Add depth and geothermal features
    if DEPTH_COL not in df.columns:
        df[DEPTH_COL] = REAL_WELL_DEPTH_KM
        logging.info(f"Added constant depth: {REAL_WELL_DEPTH_KM} km")
    
    df["geo_baseline_T_at_depth"] = (SURFACE_BASELINE_C + 
                                    GEOTHERMAL_GRADIENT_C_PER_KM * df[DEPTH_COL])
    if "geo_gradient_C_per_km" not in df.columns:
        df["geo_gradient_C_per_km"] = GEOTHERMAL_GRADIENT_C_PER_KM
    

    # Feature selection - CONTROLLED PARAMETERS ONLY
    target = OUTLET_COL
    
    # CONTROLLED FEATURES ONLY - exactly 6 parameters
    controlled_features = [
        INLET_COL,                          # Inlet temperature
        OUTDOOR_TEMP_COL,                   # Outdoor temperature  
        # 'mass_flow_hx24_kg_s',             # Mass flow rate (calculated)
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
    
    # Validate features exist
    missing_features = [f for f in features_with_depth if f not in df.columns]
    if missing_features:
        logging.error(f"Missing controlled features: {missing_features}")
        raise RuntimeError(f"Missing controlled features: {missing_features}")
    
    logging.info("CONTROLLED PARAMETERS:")
    logging.info(f"  Core features (4): {controlled_features}")
    logging.info(f"  Geo features (2): {geo_features}")
    logging.info(f"  Total controlled: {len(features_with_depth)} parameters")
    
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
            OUTDOOR_TEMP_COL: 'Outdoor Temperature', 
            'mass_flow_hx24_kg_s': 'Mass Flow Rate',
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
            "outdoor_temperature": OUTDOOR_TEMP_COL,
            # "mass_flow_rate_hx24": "mass_flow_hx24_kg_s",
            "well_thermal_resistance_mK_per_W": WELL_THERMAL_RESISTANCE,
            "bore_depth_km": REAL_WELL_DEPTH_KM,
            "geothermal_gradient_C_per_km": GEOTHERMAL_GRADIENT_C_PER_KM
        },
        "model_performance_300m": {
            "with_depth": {"MAE": float(mae_with), "RMSE": float(rmse_with)},
            "without_depth": {"MAE": float(mae_no), "RMSE": float(rmse_no)},
            "improvement_MAE": float(mae_no - mae_with),
            "improvement_RMSE": float(rmse_no - rmse_with)
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
            "total": len(features_with_depth)
        }
    }

    with open(os.path.join(OUTPUT_DIR, "controlled_analysis_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Calculate test times for plotting
    test_times = test_df[TIME_COL].iloc[SEQ_LEN+PRED_HORIZON-1:].reset_index(drop=True)
    
    # SEPARATE PLOT 1: 650m Counterfactual Analysis
    plt.figure(figsize=(15, 8))
    plt.plot(test_times, y_true_with, label="Actual (300m)", linewidth=2, color='blue')
    plt.plot(test_times, y_pred_650m, label="Predicted @ 650m", linewidth=2, color='red')
    plt.plot(test_times, y_pred_with, label="Predicted @ 300m", linewidth=2, color='green')
    
    plt.ylabel("Outlet Temperature (°C)", fontsize=12)
    plt.xlabel("Time", fontsize=12)
    plt.title("650m Depth Counterfactual Analysis", fontsize=14, fontweight='bold')
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
    
    # SEPARATE PLOT 2: Comprehensive Analysis (4 panels, excluding model comparison)
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Depth Response
    plt.subplot(2, 2, 1)
    plt.plot(depths, responses, marker='o', linewidth=2, markersize=8, color='purple')
    plt.axhline(y=np.mean(y_pred_with), color='green', linestyle='--', alpha=0.7, label='300m baseline')
    plt.axhline(y=np.mean(y_pred_650m), color='red', linestyle='--', alpha=0.7, label='650m extrapolation')
    plt.xlabel('Depth (km)', fontsize=11)
    plt.ylabel('Outlet Temperature (°C)', fontsize=11)
    plt.title(f'Depth Response: {sensitivity:.3f} °C/km', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2. Controlled Parameter Importance
    plt.subplot(2, 2, 2)
    param_names = list(param_importance.keys())
    param_values = list(param_importance.values())
    
    colors = ['lightcoral', 'skyblue', 'lightgreen', 'gold', 'plum', 'orange']
    bars = plt.barh(param_names, param_values, color=colors[:len(param_names)])
    plt.xlabel('Relative Importance (Std Dev)', fontsize=11)
    plt.title('Controlled Parameter Importance', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # 3. Performance Comparison (MAE/RMSE)
    plt.subplot(2, 2, 3)
    scenarios = ['300m\n(no depth)', '300m\n(with depth)', '650m\n(extrapolated)']
    mae_values = [mae_no, mae_with, mae_650m]
    rmse_values = [rmse_no, rmse_with, rmse_650m]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    plt.bar(x - width/2, mae_values, width, label='MAE', color='steelblue')
    plt.bar(x + width/2, rmse_values, width, label='RMSE', color='orange')
    
    plt.xlabel('Model Scenario', fontsize=11)
    plt.ylabel('Error', fontsize=11)
    plt.title('Performance Comparison: 300m vs 650m', fontsize=12, fontweight='bold')
    plt.xticks(x, scenarios, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 4. Temperature Increase Analysis
    plt.subplot(2, 2, 4)
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
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comprehensive_analysis.png"), 
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
    print(f"   2. Outdoor Temperature: {OUTDOOR_TEMP_COL}")
    # print(f"   3. Mass Flow Rate (HX24): calculated")
    print(f"   4. Thermal Resistance: {WELL_THERMAL_RESISTANCE} mK/W")
    print(f"   5. Bore Depth: {DEPTH_COL}")
    print(f"   6. Geothermal Gradient: {GEOTHERMAL_GRADIENT_C_PER_KM} C/km")
    print(f"")
    print(f"RESULTS:")
    print(f"   300m baseline MAE: {mae_with:.4f}")
    print(f"   650m extrapolated MAE: {mae_650m:.4f}")
    print(f"   Temperature increase (300m→650m): {temp_increase_predicted:.3f}°C")
    print(f"   Depth sensitivity: {sensitivity:.4f} °C/km")
    print(f"   Model improvement with depth: {metrics['model_performance_300m']['improvement_MAE']:.4f} MAE")
    print(f"")
    print(f"VALIDATION FRAMEWORK:")
    print(f"   - 650m predictions ready for validation with real data")
    print(f"   - Controlled parameters ensure reproducible results")
    print(f"   - All outputs saved to: {OUTPUT_DIR}")
    print("="*80)

    # Depth feature debug code:
    depth_idx = features_with_depth.index(DEPTH_COL)
    print(f"Depth feature index: {depth_idx}")
    print(f"Training depth stats - mean: {tr_ds.mean[depth_idx]:.6f}, std: {tr_ds.std[depth_idx]:.6f}")
    print(f"300m normalized: {(0.30 - tr_ds.mean[depth_idx]) / tr_ds.std[depth_idx]:.6f}")
    print(f"650m normalized: {(0.65 - tr_ds.mean[depth_idx]) / tr_ds.std[depth_idx]:.6f}")
