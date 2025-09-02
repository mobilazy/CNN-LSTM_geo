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

Validation: 650m well data import framework for future validation
"""

#------------------------------------------------------------------------------
# CONFIGURATION PARAMETERS
#------------------------------------------------------------------------------
CSV_PATH = os.environ.get(
    "CSV_PATH",
    os.path.join(os.path.dirname(__file__), 
                 "EDE_with_geothermal_features_eng.csv"),
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

# Column names
TIME_COL = "timestamp"
INLET_COL = "Energy_meter_energy_wells_inlet_temperature_C"
OUTLET_COL = "Energy_meter_energy_wells_return_temperature_C"
DEPTH_COL = "bore_depth_km"

# Geothermal parameters
GEOTHERMAL_GRADIENT_C_PER_KM = float(
    os.environ.get("GEOTHERMAL_GRADIENT_C_PER_KM", "27.0"))
SURFACE_BASELINE_C = float(os.environ.get("SURFACE_BASELINE_C", "8.0"))
REAL_WELL_DEPTH_KM = float(os.environ.get("REAL_WELL_DEPTH_KM", "0.30"))

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
                      ['depth', 'geo_baseline', 'geo_gradient']):
                    depth_related_indices.append(i)
            
            for idx in depth_related_indices:
                if self.std[idx] < 0.05:
                    self.std[idx] = 0.05
                    logging.info(f"Enhanced depth signal for '{features[idx]}'")
        
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
    """Framework for future 650m validation data."""
    
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
        print(f"\nCreating 650m validation framework at {validation_csv_path}")
        
        # Create placeholder structure
        sample_data = {
            TIME_COL: pd.date_range('2024-01-01', periods=100, freq='H'),
            INLET_COL: np.random.normal(12.0, 2.0, 100),
            OUTLET_COL: np.random.normal(15.0, 1.5, 100),
            DEPTH_COL: [0.65] * 100,
            "flow_rate_m3_h": np.random.normal(50.0, 5.0, 100),
            "outdoor_temperature_C": np.random.normal(8.0, 5.0, 100),
        }
        
        placeholder_df = pd.DataFrame(sample_data)
        placeholder_df.to_csv(validation_csv_path, index=False)
        print("Framework created - replace with real 650m data when available")
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
    
    # Validate required columns
    required_columns = [TIME_COL, INLET_COL, OUTLET_COL]
    for col in required_columns:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column: {col}")
    
    # Process timestamp
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.sort_values(TIME_COL).dropna(subset=[TIME_COL]).reset_index(drop=True)
    
    # Add depth and geothermal features
    if DEPTH_COL not in df.columns:
        df[DEPTH_COL] = REAL_WELL_DEPTH_KM
    
    df["geo_baseline_T_at_depth"] = (SURFACE_BASELINE_C + 
                                    GEOTHERMAL_GRADIENT_C_PER_KM * df[DEPTH_COL])
    if "geo_gradient_C_per_km" not in df.columns:
        df["geo_gradient_C_per_km"] = GEOTHERMAL_GRADIENT_C_PER_KM
    
    # Feature selection
    target = OUTLET_COL
    inlet = INLET_COL
    
    # Core features
    core_feats = [inlet]
    if "outdoor_temperature_C" in df.columns:
        core_feats.append("outdoor_temperature_C")
    
    # System features
    effect_cols = [c for c in df.columns 
                  if "power" in c.lower() or c.lower().endswith("_kw")][:6]
    flow_cols = [c for c in df.columns 
                if "flow" in c.lower()][:3]
    pressure_cols = [c for c in df.columns if "pressure" in c.lower()][:3]
    temp_aux_cols = [c for c in df.columns 
                    if "temperature" in c.lower() and 
                       c not in {target, inlet, "outdoor_temperature_C"}][:10]
    
    # Geothermal features
    geo_cols = [c for c in ["geo_gradient_C_per_km", DEPTH_COL, "geo_baseline_T_at_depth"] 
                if c in df.columns]
    
    # Derived features
    df["delta_T_in_out"] = df[inlet] - df[target]
    
    # Feature sets
    base_features = (core_feats + effect_cols + flow_cols + 
                    pressure_cols + temp_aux_cols)
    if "delta_T_in_out" not in base_features:
        base_features.append("delta_T_in_out")
    
    geo_depth_features = geo_cols.copy()
    if DEPTH_COL in df.columns:
        df[f"{DEPTH_COL}__d1"] = df[DEPTH_COL].diff()
        geo_depth_features.append(f"{DEPTH_COL}__d1")
    
    df = df.dropna().reset_index(drop=True)
    logging.info(f"Features: {len(base_features)} base + {len(geo_depth_features)} depth")
    
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
    features_with = base_features + geo_depth_features
    depth_feature_indices = [i for i, f in enumerate(features_with) 
                            if any(kw in f.lower() for kw in 
                                  ['depth', 'geo_baseline', 'geo_gradient'])]
    
    # Train model WITH depth
    logging.info("Training model WITH depth features")
    tr_loader_with, va_loader_with, te_loader_with, tr_ds, te_ds = make_loaders_enhanced(
        features_with, tr_df, va_df, test_df, target)
    model_with = DepthAwareHybridCNNLSTM(len(features_with), CONV_CHANNELS, 
                                        KERNEL_SIZE, LSTM_HIDDEN, LSTM_LAYERS, 
                                        DROPOUT, depth_feature_indices).to(device)
    model_with, hist_with = train_model(model_with, tr_loader_with, 
                                       va_loader_with, EPOCHS, LR, device, 
                                       PATIENCE, USE_SCHEDULER, "with_depth|")
    y_true_with, y_pred_with, mae_with, rmse_with = evaluate_model(
        model_with, te_loader_with, device)
    logging.info(f"Model WITH depth - MAE: {mae_with:.4f}, RMSE: {rmse_with:.4f}")

    # Train model WITHOUT depth
    logging.info("Training model WITHOUT depth features")
    tr_loader_no, va_loader_no, te_loader_no, tr_ds_no, te_ds_no = make_loaders_enhanced(
        base_features, tr_df, va_df, test_df, target)
    model_no = DepthAwareHybridCNNLSTM(len(base_features), CONV_CHANNELS, 
                                      KERNEL_SIZE, LSTM_HIDDEN, LSTM_LAYERS, 
                                      DROPOUT).to(device)
    model_no, hist_no = train_model(model_no, tr_loader_no, va_loader_no, 
                                   EPOCHS, LR, device, PATIENCE, USE_SCHEDULER, 
                                   "no_depth|")
    y_true_no, y_pred_no, mae_no, rmse_no = evaluate_model(
        model_no, te_loader_no, device)
    logging.info(f"Model WITHOUT depth - MAE: {mae_no:.4f}, RMSE: {rmse_no:.4f}")

    # Depth sensitivity analysis
    depths, responses, sensitivity = depth_sensitivity_analysis(
        model_with, test_df, features_with, tr_ds, device, target)

    # 650m validation framework
    real_650m_data = setup_650m_validation_framework()

    # Metrics
    metrics = {
        "with_depth": {"MAE": float(mae_with), "RMSE": float(rmse_with)},
        "no_depth": {"MAE": float(mae_no), "RMSE": float(rmse_no)},
        "improvement_MAE": float(mae_no - mae_with),
        "improvement_RMSE": float(rmse_no - rmse_with),
        "depth_sensitivity_C_per_km": float(sensitivity),
        "feature_counts": {
            "with_depth": len(features_with),
            "no_depth": len(base_features),
            "depth_features": len(depth_feature_indices)
        }
    }

    with open(os.path.join(OUTPUT_DIR, "metrics_depth_analysis.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Visualization
    test_times = test_df[TIME_COL].iloc[SEQ_LEN+PRED_HORIZON-1:].reset_index(drop=True)
    
    # Main analysis plots
    plt.figure(figsize=(15,10))

    # Model comparison
    plt.subplot(2,2,1)
    plt.plot(train_df[TIME_COL], train_df[target], label="Training data", alpha=0.7)
    plt.plot(test_times, y_true_with, label="Test actual", linewidth=2)
    plt.plot(test_times, y_pred_with, label="With depth", linewidth=2)
    plt.plot(test_times, y_pred_no, label="No depth", alpha=0.8)
    plt.axvline(test_df[TIME_COL].iloc[0], ls="--", color='red', alpha=0.5)
    plt.legend()
    plt.ylabel("Outlet Temperature (°C)")
    plt.title("Model Comparison")
    plt.grid(True, alpha=0.3)

    # Depth response analysis
    plt.subplot(2,2,2)
    plt.plot(depths, responses, marker='o', linewidth=2, markersize=6, color='green')
    plt.xlabel('Depth (km)')
    plt.ylabel('Outlet Temperature (°C)')
    plt.title(f'Depth Response: {sensitivity:.3f} °C/km')
    plt.grid(True, alpha=0.3)
    
    # Performance comparison
    plt.subplot(2,2,3)
    models = ['Without Depth', 'With Depth']
    mae_values = [mae_no, mae_with]
    rmse_values = [rmse_no, rmse_with]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, mae_values, width, label='MAE', color='steelblue')
    plt.bar(x + width/2, rmse_values, width, label='RMSE', color='orange')
    
    plt.xlabel('Model Type')
    plt.ylabel('Error')
    plt.title('Performance Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comprehensive_analysis.png"), 
                dpi=200, bbox_inches='tight')
    plt.close()

    # 650m counterfactual analysis
    cf = test_df.copy()
    cf[DEPTH_COL] = 0.65
    cf["geo_baseline_T_at_depth"] = (SURFACE_BASELINE_C + 
                                    GEOTHERMAL_GRADIENT_C_PER_KM * cf[DEPTH_COL])
    ds = DepthAwareSequenceDataset(cf, TIME_COL, target, features_with, 
                                SEQ_LEN, PRED_HORIZON, 
                                mean=tr_ds.mean, std=tr_ds.std)
    dl = DataLoader(ds, BATCH_SIZE)
    _, ycf, _, _ = evaluate_model(model_with, dl, device)

    plt.figure(figsize=(12,6))
    plt.plot(test_times, y_true_with, label="Actual", linewidth=2, color='blue')
    plt.plot(test_times, ycf, label="Predicted @ 650m", linewidth=2, color='red')
    plt.plot(test_times, y_pred_with, label="Predicted @ original depth", alpha=0.7, color='green')
    plt.legend()
    plt.ylabel("Outlet Temperature (°C)")
    plt.title("650m Depth Counterfactual Analysis")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "650m_counterfactual_analysis.png"), 
                dpi=200, bbox_inches='tight')
    plt.close()

    # Save model
    torch.save(model_with.state_dict(), 
               os.path.join(OUTPUT_DIR, "depth_aware_model.pth"))
    
    logging.info("Analysis complete")
    logging.info(f"Depth sensitivity: {sensitivity:.4f} C/km")
    logging.info(f"MAE improvement: {metrics['improvement_MAE']:.4f}")
    logging.info(f"Outputs saved to: {OUTPUT_DIR}")

    if device == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("DEPTH ANALYSIS COMPLETE")
    print("="*60)
    print(f"Main findings:")
    print(f"   Can depth signal improve prediction? MAE change: {metrics['improvement_MAE']:.4f}")
    print(f"   How does outlet temp change with depth? {sensitivity:.4f} °C/km")
    print(f"   Depth features used: {len(depth_feature_indices)}")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("="*60)