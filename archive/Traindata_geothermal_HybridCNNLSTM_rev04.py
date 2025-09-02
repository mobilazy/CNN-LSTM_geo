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
Hybrid CNN+LSTM forecaster on geothermal time series (rev03).
- Fixed English headers (no heuristics).
- Physics-informed depth features so counterfactual depth changes affect predictions.
- Counterfactual 650 m timeline, inlet vs outlet by depth (300/650/1300 m),
  and a combined supplementary figure.
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
DEPTH_COL = "bore_depth_km"  # created if missing

# Geothermal assumptions (adjust via env)
GEOTHERMAL_GRADIENT_C_PER_KM = float(os.environ.get("GEOTHERMAL_GRADIENT_C_PER_KM", "27.0"))
SURFACE_BASELINE_C = float(os.environ.get("SURFACE_BASELINE_C", "8.0"))
REAL_WELL_DEPTH_KM = float(os.environ.get("REAL_WELL_DEPTH_KM", "0.30"))  # used if depth is missing in CSV

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
# Dataset
# =============================================================================
class SequenceDataset(Dataset):
    def __init__(self, df, time_col, target, features, seq_len, horizon, mean=None, std=None):
        self.time = df[time_col].to_numpy()
        self.y = df[target].to_numpy(dtype=np.float32)
        self.X = df[features].to_numpy(dtype=np.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        if mean is None or std is None:
            self.mean = self.X.mean(axis=0)
            self.std = self.X.std(axis=0) + 1e-8
        else:
            self.mean = mean
            self.std = std
        self.X = (self.X - self.mean) / self.std

        self.valid_idx = []
        max_start = len(self.X) - (seq_len + horizon)
        for i in range(max(0, max_start) + 1):
            self.valid_idx.append(i)

    def __len__(self): return len(self.valid_idx)

    def __getitem__(self, idx):
        i = self.valid_idx[idx]
        seq = self.X[i : i + self.seq_len]
        target = self.y[i + self.seq_len + self.horizon - 1]
        seq_ch_first = torch.from_numpy(seq).float().transpose(0, 1)
        return seq_ch_first, torch.tensor(target, dtype=torch.float32)

# =============================================================================
# Model
# =============================================================================
class HybridCNNLSTM(nn.Module):
    def __init__(self, in_channels, conv_channels=(32,32), kernel_size=3,
                 lstm_hidden=64, lstm_layers=2, dropout=0.1):
        super().__init__()
        channels = [in_channels] + list(conv_channels)
        convs = []
        for i in range(len(channels) - 1):
            convs += [
                nn.Conv1d(channels[i], channels[i+1], kernel_size, padding=kernel_size//2),
                nn.ReLU(),
            ]
        self.conv = nn.Sequential(*convs)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=channels[-1], hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=False)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = x.permute(2,0,1)
        out,_ = self.lstm(x)
        last = out[-1]
        y = self.fc(last).squeeze(-1)
        return y
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

    def make_loaders(features):
        tr_ds=SequenceDataset(tr_df,TIME_COL,target,features,SEQ_LEN,PRED_HORIZON)
        va_ds=SequenceDataset(va_df,TIME_COL,target,features,SEQ_LEN,PRED_HORIZON,mean=tr_ds.mean,std=tr_ds.std)
        te_ds=SequenceDataset(test_df,TIME_COL,target,features,SEQ_LEN,PRED_HORIZON,mean=tr_ds.mean,std=tr_ds.std)
        return (DataLoader(tr_ds,BATCH_SIZE,shuffle=True),
                DataLoader(va_ds,BATCH_SIZE),
                DataLoader(te_ds,BATCH_SIZE),
                tr_ds,te_ds)

    # With depth
    features_with=base_features+geo_depth_features
   # -------------------------------------------------------------------------
   # PREFLIGHT: Does changing depth actually change the feature tensor?
ABORT_IF_NO_DEPTH_SIGNAL = os.environ.get("ABORT_IF_NO_DEPTH_SIGNAL", "0") in {"1","true","yes"}
SHALLOW_KM, DEEP_KM = 0.30, 1.30
PROBE_N = min(500, len(df))

# Ensure we have enough rows to form at least one sequence
min_needed = SEQ_LEN + PRED_HORIZON + 1
if PROBE_N < min_needed:
    logging.warning(
        f"[Preflight] Skipping depth-signal check: only {PROBE_N} rows "
        f"(< {min_needed} needed for one sequence)."
    )
else:
    # Create two sample datasets with different depths
    df_shallow = df.head(PROBE_N).copy()
    df_shallow[DEPTH_COL] = SHALLOW_KM
    df_shallow["geo_baseline_T_at_depth"] = (
        SURFACE_BASELINE_C + GEOTHERMAL_GRADIENT_C_PER_KM * df_shallow[DEPTH_COL]
    )

    df_deep = df.head(PROBE_N).copy()
    df_deep[DEPTH_COL] = DEEP_KM
    df_deep["geo_baseline_T_at_depth"] = (
        SURFACE_BASELINE_C + GEOTHERMAL_GRADIENT_C_PER_KM * df_deep[DEPTH_COL]
    )

    # Build probe datasets (standardize deep with shallow's stats)
    ds_shallow = SequenceDataset(df_shallow, TIME_COL, target, features_with, SEQ_LEN, PRED_HORIZON)
    ds_deep = SequenceDataset(
        df_deep, TIME_COL, target, features_with, SEQ_LEN, PRED_HORIZON,
        mean=ds_shallow.mean, std=ds_shallow.std
    )

    if len(ds_shallow) == 0 or len(ds_deep) == 0:
        logging.warning("[Preflight] Skipping depth-signal check: probe datasets are empty.")
    else:
        probe_loader_shallow = DataLoader(ds_shallow, batch_size=8, shuffle=False)
        probe_loader_deep = DataLoader(ds_deep, batch_size=8, shuffle=False)

        tensor_shallow = next(iter(probe_loader_shallow))[0]
        tensor_deep = next(iter(probe_loader_deep))[0]

        mean_abs_diff = float(torch.abs(tensor_shallow - tensor_deep).mean())
        logging.info(f"[Preflight] mean_abs_diff={mean_abs_diff:.3e} (threshold=1e-6)")
        if mean_abs_diff < 1e-6:
            msg = "Depth changes do not affect feature tensor (no signal in standardized features)"
            if ABORT_IF_NO_DEPTH_SIGNAL:
                raise SystemExit(msg)  # or RuntimeError(msg) if you prefer
            else:
                logging.warning(msg)
        else:
            logging.info(f"Depth signal confirmed (mean abs diff: {mean_abs_diff:.2e})")

# -------------------------------------------------------------------------
# After preflight check and cache clearing, train models in correct order:

# 1. First train and evaluate model WITH depth features
logging.info(f"Training model WITH depth features ({len(features_with)} features)")
tr_loader_with, va_loader_with, te_loader_with, tr_ds, te_ds = make_loaders(features_with)
model_with = HybridCNNLSTM(len(features_with), CONV_CHANNELS, KERNEL_SIZE, 
                          LSTM_HIDDEN, LSTM_LAYERS, DROPOUT).to(device)
model_with, hist_with = train_model(model_with, tr_loader_with, va_loader_with, 
                                  EPOCHS, LR, device, PATIENCE, USE_SCHEDULER, "with_depth|")
# Store ALL evaluation results including predictions for plotting
y_true_with, y_pred_with, mae_with, rmse_with = evaluate_model(model_with, te_loader_with, device)
logging.info(f"Model WITH depth - MAE: {mae_with:.4f}, RMSE: {rmse_with:.4f}")

# 2. Train and evaluate model WITHOUT depth features
logging.info(f"Training model WITHOUT depth features ({len(base_features)} features)")
tr_loader_no, va_loader_no, te_loader_no, tr_ds_no, te_ds_no = make_loaders(base_features)
model_no = HybridCNNLSTM(len(base_features), CONV_CHANNELS, KERNEL_SIZE,
                         LSTM_HIDDEN, LSTM_LAYERS, DROPOUT).to(device)
model_no, hist_no = train_model(model_no, tr_loader_no, va_loader_no, 
                               EPOCHS, LR, device, PATIENCE, USE_SCHEDULER, "no_depth|")
# Store ALL evaluation results including predictions for plotting
y_true_no, y_pred_no, mae_no, rmse_no = evaluate_model(model_no, te_loader_no, device)
logging.info(f"Model WITHOUT depth - MAE: {mae_no:.4f}, RMSE: {rmse_no:.4f}")

# 3. Create metrics dictionary after both evaluations are complete
metrics = {
    "with_depth": {"MAE": float(mae_with), "RMSE": float(rmse_with)},
    "no_depth": {"MAE": float(mae_no), "RMSE": float(rmse_no)},
    "improvement_MAE": float(mae_no - mae_with),
    "improvement_RMSE": float(rmse_no - rmse_with)
}

# 4. Save metrics
json.dump(metrics, open(os.path.join(OUTPUT_DIR, "metrics_geothermal.json"), "w"), indent=2)

# Timeline plot
test_times=test_df[TIME_COL].iloc[SEQ_LEN+PRED_HORIZON-1:].reset_index(drop=True)
plt.figure(figsize=(12,4))
plt.plot(train_df[TIME_COL],train_df[target],label="Previous training")
plt.plot(test_times,y_true_with,label="Test actual")
plt.plot(test_times,y_pred_with,label="With depth")
plt.plot(test_times,y_pred_no,label="No depth")
plt.axvline(test_df[TIME_COL].iloc[0],ls="--"); plt.legend(); plt.ylabel("Outlet °C")
plt.savefig(os.path.join(OUTPUT_DIR,"cnn_lstm_depth_comparison_eng.png"),dpi=200)

# Counterfactual 650m
cf=test_df.copy(); cf[DEPTH_COL]=0.65
cf["geo_baseline_T_at_depth"]=SURFACE_BASELINE_C+GEOTHERMAL_GRADIENT_C_PER_KM*cf[DEPTH_COL]
ds=SequenceDataset(cf,TIME_COL,target,features_with,SEQ_LEN,PRED_HORIZON,mean=tr_ds.mean,std=tr_ds.std)
dl=DataLoader(ds,BATCH_SIZE); _,ycf,_,_=evaluate_model(model_with,dl,device)
plt.figure(); plt.plot(test_times,y_true_with,label="Actual"); plt.plot(test_times,ycf,label="Pred 650m"); plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR,"counterfactual_650m_timeline_eng.png"),dpi=200)

# Depth sweep
depths=[0.30,0.65,1.30]; pred_by={}
for d in depths:
    cf=test_df.copy(); cf[DEPTH_COL]=d
    cf["geo_baseline_T_at_depth"]=SURFACE_BASELINE_C+GEOTHERMAL_GRADIENT_C_PER_KM*cf[DEPTH_COL]
    ds=SequenceDataset(cf,TIME_COL,target,features_with,SEQ_LEN,PRED_HORIZON,mean=tr_ds.mean,std=tr_ds.std)
    dl=DataLoader(ds,BATCH_SIZE); _,yp,_,_=evaluate_model(model_with,dl,device); pred_by[d]=yp
inlet_aligned=test_df[inlet].iloc[SEQ_LEN+PRED_HORIZON-1:].to_numpy()
plt.figure(); 
for d in depths: plt.scatter(inlet_aligned,pred_by[d],s=8,label=f"{int(d*1000)}m")
plt.legend(); plt.xlabel("Inlet °C"); plt.ylabel("Outlet °C")
plt.savefig(os.path.join(OUTPUT_DIR,"inlet_vs_outlet_by_depth_eng.png"),dpi=200)

# Clear CUDA cache after all operations are complete
if device == "cuda":
    torch.cuda.empty_cache()