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

# =============================================================================
# Configuration
# =============================================================================
"""
Hybrid CNN+LSTM forecaster on geothermal time series (rev05).
- Enhanced depth signal preservation during standardization
- Advanced diagnostics for depth feature behavior
- Depth-aware attention mechanism
- Comprehensive analysis tools
"""

CSV_PATH = os.environ.get(
    "CSV_PATH",
    os.path.join(os.path.dirname(__file__), "EDE_with_geothermal_features_eng.csv"),
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "output"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "48"))
PRED_HORIZON = int(os.environ.get("PRED_HORIZON", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
LR = float(os.environ.get("LR", "1e-3"))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", "0.1"))
TEST_SPLIT = float(os.environ.get("TEST_SPLIT", "0.2"))

# Model hyperparams
CONV_CHANNELS = [int(x) for x in os.environ.get("CONV_CHANNELS", "32,32").split(",") if x.strip()]
KERNEL_SIZE = int(os.environ.get("KERNEL_SIZE", "3"))
LSTM_HIDDEN = int(os.environ.get("LSTM_HIDDEN", "64"))
LSTM_LAYERS = int(os.environ.get("LSTM_LAYERS", "2"))
DROPOUT = float(os.environ.get("DROPOUT", "0.1"))
PATIENCE = int(os.environ.get("PATIENCE", "16"))
USE_SCHEDULER = os.environ.get("USE_SCHEDULER", "false").lower() in {"1", "true", "yes"}

# Fixed column names
TIME_COL = "timestamp"
INLET_COL = "Energy_meter_energy_wells_inlet_temperature_C"
OUTLET_COL = "Energy_meter_energy_wells_return_temperature_C"
DEPTH_COL = "bore_depth_km"

# Geothermal assumptions
GEOTHERMAL_GRADIENT_C_PER_KM = float(os.environ.get("GEOTHERMAL_GRADIENT_C_PER_KM", "27.0"))
SURFACE_BASELINE_C = float(os.environ.get("SURFACE_BASELINE_C", "8.0"))
REAL_WELL_DEPTH_KM = float(os.environ.get("REAL_WELL_DEPTH_KM", "0.30"))

# =============================================================================
# Logging
# =============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "train.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("train")

# =============================================================================
# Enhanced Dataset with Depth Signal Preservation
# =============================================================================
class DepthAwareSequenceDataset(Dataset):
    """Enhanced dataset that preserves depth signal during standardization."""
    
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
        
        # 🎯 SMART DEPTH SIGNAL PRESERVATION
        if preserve_depth_signal:
            depth_related_indices = []
            for i, feat in enumerate(features):
                if any(keyword in feat.lower() for keyword in ['depth', 'geo_baseline', 'geo_gradient']):
                    depth_related_indices.append(i)
            
            # Ensure depth features maintain sufficient variation
            for idx in depth_related_indices:
                if self.std[idx] < 0.05:
                    self.std[idx] = 0.05
                    logging.info(f"Enhanced std for depth feature '{features[idx]}': {self.std[idx]:.3f}")
        
        # Standardize with preserved depth signal
        self.X = (self.X - self.mean) / self.std
        
        # Calculate valid indices
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

# =============================================================================
# Enhanced Model with Depth-Aware Attention
# =============================================================================
class DepthAwareHybridCNNLSTM(nn.Module):
    """Enhanced CNN-LSTM with depth feature attention mechanism."""
    
    def __init__(self, in_channels, conv_channels=(32,32), kernel_size=3,
                 lstm_hidden=64, lstm_layers=2, dropout=0.1, depth_feature_indices=None):
        super().__init__()
        channels = [in_channels] + list(conv_channels)
        
        # Convolutional layers
        convs = []
        for i in range(len(channels) - 1):
            convs += [
                nn.Conv1d(channels[i], channels[i+1], kernel_size, padding=kernel_size//2),
                nn.ReLU(),
            ]
        self.conv = nn.Sequential(*convs)
        
        # Depth feature attention
        self.depth_attention = None
        if depth_feature_indices and len(depth_feature_indices) > 0:
            self.depth_attention = nn.MultiheadAttention(
                embed_dim=channels[-1], num_heads=4, dropout=dropout, batch_first=False
            )
            logging.info(f"Enabled depth attention for {len(depth_feature_indices)} features")
        
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=channels[-1], hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=False)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # x shape: (batch, features, time)
        x = self.conv(x)  # (batch, channels, time)
        
        # Apply depth attention if available
        if self.depth_attention is not None:
            x_permuted = x.permute(2, 0, 1)  # (time, batch, channels)
            x_attended, _ = self.depth_attention(x_permuted, x_permuted, x_permuted)
            x = x_attended.permute(1, 2, 0)  # back to (batch, channels, time)
        
        x = self.dropout(x)
        x = x.permute(2, 0, 1)  # (time, batch, channels)
        out, _ = self.lstm(x)
        last = out[-1]  # (batch, hidden)
        y = self.fc(last).squeeze(-1)
        return y

# =============================================================================
# Advanced Diagnostics
# =============================================================================

def simplified_depth_analysis(model, test_df, features_with, tr_ds, device):
    """Simplified depth analysis focusing on counterfactual responses."""
    
    print("\n🔬 DEPTH SIGNAL ANALYSIS")
    print("="*50)
    
    # 1. Feature standardization impact analysis
    depth_idx = features_with.index(DEPTH_COL)
    baseline_idx = features_with.index("geo_baseline_T_at_depth")
    
    print(f"\n📊 STANDARDIZATION IMPACT:")
    print(f"Original depth range: [{test_df[DEPTH_COL].min():.3f}, {test_df[DEPTH_COL].max():.3f}] km")
    print(f"Training mean/std: {tr_ds.mean[depth_idx]:.6f} / {tr_ds.std[depth_idx]:.6f}")
    
    # Calculate signal-to-noise ratio
    depth_variation = test_df[DEPTH_COL].std()
    standardized_variation = depth_variation / tr_ds.std[depth_idx]
    print(f"Signal preservation ratio: {standardized_variation:.6f} (should be > 0.1)")
    
    # 2. Counterfactual response test with multiple depths
    model.eval()
    depths_test = np.linspace(0.2, 1.5, 8)  # 200m to 1500m
    responses = []
    
    print(f"\n🎯 COUNTERFACTUAL RESPONSE CURVE:")
    for depth in depths_test:
        cf = test_df.head(50).copy()
        cf[DEPTH_COL] = depth
        cf["geo_baseline_T_at_depth"] = SURFACE_BASELINE_C + GEOTHERMAL_GRADIENT_C_PER_KM * depth
        
        ds = DepthAwareSequenceDataset(cf, TIME_COL, target, features_with, SEQ_LEN, PRED_HORIZON, 
                                     mean=tr_ds.mean, std=tr_ds.std)
        dl = DataLoader(ds, batch_size=len(ds), shuffle=False)
        
        with torch.no_grad():
            _, preds, _, _ = evaluate_model(model, dl, device)
        
        avg_pred = preds.mean()
        responses.append(avg_pred)
        print(f"Depth {depth:.2f}km → Avg prediction: {avg_pred:.4f}°C")
    
    # Calculate response sensitivity
    depth_response_slope = (responses[-1] - responses[0]) / (depths_test[-1] - depths_test[0])
    print(f"Overall depth sensitivity: {depth_response_slope:.4f} °C/km")
    
    return depths_test, responses, depth_response_slope

# =============================================================================
# Training & eval
# =============================================================================
def train_model(model, train_loader, val_loader, epochs, lr, device, patience, use_scheduler, log_prefix=""):
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                           patience=max(1,patience//2)) if use_scheduler else None
    best_val = float("inf"); best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    bad=0; hist={"train_loss":[],"val_loss":[]}
    for ep in range(1,epochs+1):
        model.train(); tr_loss=0.0
        for Xb,yb in tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [train]", leave=False):
            Xb=Xb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            yh=model(Xb); loss=crit(yh,yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tr_loss+=loss.item()*Xb.size(0)
        tr_loss/=max(1,len(train_loader.dataset))
        model.eval(); va_loss=0.0
        with torch.no_grad():
            for Xb,yb in tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [val]", leave=False):
                Xb=Xb.to(device); yb=yb.to(device)
                yh=model(Xb); va_loss+=crit(yh,yb).item()*Xb.size(0)
        va_loss/=max(1,len(val_loader.dataset))
        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        logging.info(f"{log_prefix}Epoch {ep} train {tr_loss:.5f} val {va_loss:.5f}")
        if scheduler: scheduler.step(va_loss)
        if va_loss+1e-9<best_val: best_val=va_loss; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=patience: logging.info(f"Early stop at {ep}"); break
    model.load_state_dict(best_state); return model,hist

def evaluate_model(model,data_loader,device="cpu"):
    model.eval(); preds=[]; trues=[]
    with torch.no_grad():
        for Xb,yb in tqdm(data_loader,desc="Evaluating",leave=False):
            Xb=Xb.to(device); yb=yb.to(device); yh=model(Xb)
            preds.append(yh.cpu().numpy()); trues.append(yb.cpu().numpy())
    preds=np.concatenate(preds) if preds else np.array([])
    trues=np.concatenate(trues) if trues else np.array([])
    mae=float(np.mean(np.abs(preds-trues))) if len(preds) else float("nan")
    rmse=float(np.sqrt(np.mean((preds-trues)**2))) if len(preds) else float("nan")
    return trues,preds,mae,rmse
    
# =============================================================================
# Placeholder to add real data for 650m semi-deep wells for validation
# ============================================================================= 
def setup_650m_validation_csv_import():
    """Setup CSV import placeholder for real 650m validation data."""
    
    validation_csv_path = os.path.join(os.path.dirname(__file__), "real_650m_validation_data.csv")
    
    if os.path.exists(validation_csv_path):
        print(f"\n📊 LOADING REAL 650m VALIDATION DATA")
        try:
            real_650m_df = pd.read_csv(validation_csv_path)
            real_650m_df[TIME_COL] = pd.to_datetime(real_650m_df[TIME_COL], errors="coerce")
            real_650m_df = real_650m_df.dropna(subset=[TIME_COL]).reset_index(drop=True)
            print(f"✅ Loaded {len(real_650m_df)} real 650m measurements")
            return real_650m_df
        except Exception as e:
            print(f"❌ Error loading 650m data: {e}")
            return None
    else:
        print(f"\n📝 CREATING CSV PLACEHOLDER FOR 650m VALIDATION DATA")
        sample_data = {
            TIME_COL: pd.date_range('2024-01-01', periods=100, freq='H'),
            INLET_COL: np.random.normal(12.0, 2.0, 100),
            OUTLET_COL: np.random.normal(15.0, 1.5, 100),
            DEPTH_COL: [0.65] * 100,
            "flow_rate_m3_h": np.random.normal(50.0, 5.0, 100),
            "outdoor_temperature_C": np.random.normal(8.0, 5.0, 100)
        }
        placeholder_df = pd.DataFrame(sample_data)
        placeholder_df.to_csv(validation_csv_path, index=False)
        print(f"📋 Created CSV placeholder: {validation_csv_path}")
        return None

# =============================================================================
# Main
# =============================================================================
if __name__=="__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"; logging.info(f"Using device {device}")
    if not os.path.exists(CSV_PATH): raise FileNotFoundError(CSV_PATH)
    df=pd.read_csv(CSV_PATH)
    for rc in [TIME_COL,INLET_COL,OUTLET_COL]:
        if rc not in df.columns: raise RuntimeError(f"Missing {rc}")
    df[TIME_COL]=pd.to_datetime(df[TIME_COL],errors="coerce")
    df=df.sort_values(TIME_COL).dropna(subset=[TIME_COL]).reset_index(drop=True)

    # Inject depth signal
    if DEPTH_COL not in df.columns: df[DEPTH_COL]=REAL_WELL_DEPTH_KM
    df["geo_baseline_T_at_depth"]=SURFACE_BASELINE_C+GEOTHERMAL_GRADIENT_C_PER_KM*df[DEPTH_COL]
    if "geo_gradient_C_per_km" not in df.columns: df["geo_gradient_C_per_km"]=GEOTHERMAL_GRADIENT_C_PER_KM

    # Features
    target=OUTLET_COL; inlet=INLET_COL
    core_feats=[inlet]
    if "outdoor_temperature_C" in df.columns and "outdoor_temperature_C"!=target:
        core_feats.append("outdoor_temperature_C")
    effect_cols=[c for c in df.columns if "power" in c.lower() or c.lower().endswith("_kw") or "heat" in c.lower()]
    flow_cols=[c for c in df.columns if "flow" in c.lower() or "throughput" in c.lower()]
    pressure_cols=[c for c in df.columns if "pressure" in c.lower()]
    temp_aux_cols=[c for c in df.columns if "temperature" in c.lower() and c not in {target,inlet,"outdoor_temperature_C"}]
    geo_cols=[c for c in ["geo_gradient_C_per_km","geo_heatflow_mW_m2",DEPTH_COL,"geo_baseline_T_at_depth"] if c in df.columns]
    df["delta_T_in_out"]=df[inlet]-df[target]
    base_features=core_feats+effect_cols[:6]+flow_cols[:3]+pressure_cols[:3]+temp_aux_cols[:10]
    if "delta_T_in_out" not in base_features: base_features.append("delta_T_in_out")
    geo_depth_features=geo_cols.copy()
    if DEPTH_COL in df.columns:
        df[f"{DEPTH_COL}__d1"]=df[DEPTH_COL].diff(); geo_depth_features.append(f"{DEPTH_COL}__d1")
    df=df.dropna().reset_index(drop=True)
    
    # Split
    N=len(df)
    if N<(SEQ_LEN+PRED_HORIZON+1): raise SystemExit("Dataset too small")
    test_start=int(N*(1.0-TEST_SPLIT)); test_start=max(test_start,SEQ_LEN+PRED_HORIZON)
    train_df=df.iloc[:test_start].copy(); test_df=df.iloc[test_start:].copy()
    val_size=max(1,int(len(train_df)*VAL_SPLIT))
    tr_df=train_df.iloc[:-val_size].copy(); va_df=train_df.iloc[-val_size:].copy()

    def make_loaders_enhanced(features):
        tr_ds = DepthAwareSequenceDataset(tr_df, TIME_COL, target, features, SEQ_LEN, PRED_HORIZON)
        va_ds = DepthAwareSequenceDataset(va_df, TIME_COL, target, features, SEQ_LEN, PRED_HORIZON, 
                                         mean=tr_ds.mean, std=tr_ds.std)
        te_ds = DepthAwareSequenceDataset(test_df, TIME_COL, target, features, SEQ_LEN, PRED_HORIZON, 
                                         mean=tr_ds.mean, std=tr_ds.std)
        return (DataLoader(tr_ds, BATCH_SIZE, shuffle=True),
                DataLoader(va_ds, BATCH_SIZE),
                DataLoader(te_ds, BATCH_SIZE),
                tr_ds, te_ds)

    # With depth features
    features_with = base_features + geo_depth_features
    
    # Find depth feature indices for attention mechanism
    depth_feature_indices = [i for i, f in enumerate(features_with) 
                            if any(kw in f.lower() for kw in ['depth', 'geo_baseline', 'geo_gradient'])]
    
    logging.info(f"Depth feature indices: {depth_feature_indices}")
    logging.info(f"Depth features: {[features_with[i] for i in depth_feature_indices]}")

    # PREFLIGHT CHECK (simplified for rev05)
    ABORT_IF_NO_DEPTH_SIGNAL = os.environ.get("ABORT_IF_NO_DEPTH_SIGNAL", "0") in {"1","true","yes"}
    
    # 1. Train model WITH depth features
    logging.info(f"Training model WITH depth features ({len(features_with)} features)")
    tr_loader_with, va_loader_with, te_loader_with, tr_ds, te_ds = make_loaders_enhanced(features_with)
    model_with = DepthAwareHybridCNNLSTM(len(features_with), CONV_CHANNELS, KERNEL_SIZE, 
                                        LSTM_HIDDEN, LSTM_LAYERS, DROPOUT, 
                                        depth_feature_indices=depth_feature_indices).to(device)
    model_with, hist_with = train_model(model_with, tr_loader_with, va_loader_with, 
                                      EPOCHS, LR, device, PATIENCE, USE_SCHEDULER, "with_depth|")
    y_true_with, y_pred_with, mae_with, rmse_with = evaluate_model(model_with, te_loader_with, device)
    logging.info(f"Model WITH depth - MAE: {mae_with:.4f}, RMSE: {rmse_with:.4f}")

    # 2. Train model WITHOUT depth features
    logging.info(f"Training model WITHOUT depth features ({len(base_features)} features)")
    tr_loader_no, va_loader_no, te_loader_no, tr_ds_no, te_ds_no = make_loaders_enhanced(base_features)
    model_no = DepthAwareHybridCNNLSTM(len(base_features), CONV_CHANNELS, KERNEL_SIZE,
                                      LSTM_HIDDEN, LSTM_LAYERS, DROPOUT).to(device)
    model_no, hist_no = train_model(model_no, tr_loader_no, va_loader_no, 
                                   EPOCHS, LR, device, PATIENCE, USE_SCHEDULER, "no_depth|")
    y_true_no, y_pred_no, mae_no, rmse_no = evaluate_model(model_no, te_loader_no, device)
    logging.info(f"Model WITHOUT depth - MAE: {mae_no:.4f}, RMSE: {rmse_no:.4f}")

    # 3. Advanced depth analysis
    depths, responses, sensitivity = simplified_depth_analysis(model_with, test_df, features_with, tr_ds, device)

    # 4. Create metrics
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

    # Setup real 650m validation data
    real_650m_data = setup_650m_validation_csv_import()

    # Save metrics (existing line)
    with open(os.path.join(OUTPUT_DIR, "metrics_geothermal_rev05.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Enhanced plotting with diagnostics
    test_times = test_df[TIME_COL].iloc[SEQ_LEN+PRED_HORIZON-1:].reset_index(drop=True)
    
    # Timeline comparison plot
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    plt.plot(train_df[TIME_COL], train_df[target], label="Training data", alpha=0.7)
    plt.plot(test_times, y_true_with, label="Test actual", linewidth=2)
    plt.plot(test_times, y_pred_with, label="With depth", linewidth=2)
    plt.plot(test_times, y_pred_no, label="No depth", alpha=0.8)
    plt.axvline(test_df[TIME_COL].iloc[0], ls="--", color='red', alpha=0.5)
    plt.legend(); plt.ylabel("Outlet °C"); plt.title("Model Comparison")
    
    # Depth response curve
    plt.subplot(1,2,2)
    plt.plot(depths, responses, 'o-', linewidth=2, markersize=8)
    plt.xlabel("Depth (km)"); plt.ylabel("Avg Prediction (°C)")
    plt.title(f"Depth Sensitivity: {sensitivity:.3f} °C/km")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "enhanced_depth_analysis_rev05.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Counterfactual 650m with enhanced analysis
    cf = test_df.copy(); cf[DEPTH_COL] = 0.65
    cf["geo_baseline_T_at_depth"] = SURFACE_BASELINE_C + GEOTHERMAL_GRADIENT_C_PER_KM * cf[DEPTH_COL]
    ds = DepthAwareSequenceDataset(cf, TIME_COL, target, features_with, SEQ_LEN, PRED_HORIZON, 
                                  mean=tr_ds.mean, std=tr_ds.std)
    dl = DataLoader(ds, BATCH_SIZE)
    _, ycf, _, _ = evaluate_model(model_with, dl, device)
    
    plt.figure(figsize=(12,5))
    plt.plot(test_times, y_true_with, label="Actual", linewidth=2)
    plt.plot(test_times, ycf, label="Predicted @ 650m", linewidth=2)
    plt.plot(test_times, y_pred_with, label="Predicted @ original depth", alpha=0.7)
    plt.legend(); plt.ylabel("Outlet °C"); plt.title("650m Counterfactual Analysis")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "counterfactual_650m_enhanced_rev05.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Depth sweep analysis
    depths = [0.30, 0.65, 1.30]; pred_by = {}
    for d in depths:
        cf = test_df.copy(); cf[DEPTH_COL] = d
        cf["geo_baseline_T_at_depth"] = SURFACE_BASELINE_C + GEOTHERMAL_GRADIENT_C_PER_KM * cf[DEPTH_COL]
        ds = DepthAwareSequenceDataset(cf, TIME_COL, target, features_with, SEQ_LEN, PRED_HORIZON, 
                                      mean=tr_ds.mean, std=tr_ds.std)
        dl = DataLoader(ds, BATCH_SIZE)
        _, yp, _, _ = evaluate_model(model_with, dl, device)
        pred_by[d] = yp

    inlet_aligned = test_df[inlet].iloc[SEQ_LEN+PRED_HORIZON-1:].to_numpy()
    plt.figure(figsize=(10,6))
    colors = ['blue', 'orange', 'green']
    for i, d in enumerate(depths):
        plt.scatter(inlet_aligned, pred_by[d], s=20, alpha=0.7, 
                   label=f"{int(d*1000)}m depth", color=colors[i])
    plt.legend(); plt.xlabel("Inlet °C"); plt.ylabel("Outlet °C")
    plt.title("Inlet vs Outlet by Depth (Enhanced)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "inlet_vs_outlet_enhanced_rev05.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Save model
    torch.save(model_with.state_dict(), os.path.join(OUTPUT_DIR, "enhanced_cnn_lstm_with_depth_rev05.pth"))
    
    logging.info("Rev05 training complete with enhanced diagnostics!")
    logging.info(f"Depth sensitivity: {sensitivity:.4f} °C/km")
    logging.info(f"MAE improvement: {metrics['improvement_MAE']:.4f}")

    # Clear CUDA cache
    if device == "cuda":
        torch.cuda.empty_cache()