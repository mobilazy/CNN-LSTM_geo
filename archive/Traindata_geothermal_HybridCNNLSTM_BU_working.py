import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ================================
# Configuration
# ================================
CSV_PATH = os.environ.get("CSV_PATH", os.path.join(os.path.dirname(__file__), "EDE_with_geothermal_features.csv"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "output"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "48"))           # sequence length (timesteps per sample)
PRED_HORIZON = int(os.environ.get("PRED_HORIZON", "1"))  # forecast horizon (steps ahead)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
LR = float(os.environ.get("LR", "1e-3"))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", "0.1"))    # portion of training set used for validation
TEST_SPLIT = float(os.environ.get("TEST_SPLIT", "0.2"))  # last portion of time used for test

# ================================
# Utility functions
# ================================
def detect_time_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        s = c.lower()
        if s.startswith("tid") or "time" in s or "timestamp" in s:
            return c
    return df.columns[0]

def is_depth_col(name: str) -> bool:
    s = name.lower()
    return ("nivå" in s or "vannstand" in s) and any(k in s for k in ["grunn", "brønn", "brønner", "brønnpark", "grunnvann"])

def pick_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Pick target and feature groups from the dataframe header."""
    cols = [c for c in df.columns if c != time_col]

    # Target: outlet / return temperature
    target_col = None
    for k in ["retur temperatur", "returtemperatur", "outlet", "utløpstemperatur"]:
        for c in cols:
            if k in c.lower():
                target_col = c
                break
        if target_col:
            break
    if target_col is None:
        # fallback -> any temperature column
        for c in cols:
            if "temperatur" in c.lower():
                target_col = c
                break
    if target_col is None:
        raise RuntimeError("Could not find a target (return/outlet temperature) column.")

    # Inlet / supply temperature
    inlet_col = None
    for k in ["tur temperatur", "innløp", "frem", "inlet", "supply"]:
        if inlet_col: break
        for c in cols:
            if k in c.lower():
                inlet_col = c
                break

    # Outdoor
    outdoor_col = None
    for c in cols:
        if "utetemperatur" in c.lower() or "outdoor" in c.lower():
            outdoor_col = c
            break

    # Power/effect, flow, pressures, aux temps
    effect_cols   = [c for c in cols if "avgitt effekt" in c.lower() or c.lower().endswith(" kw")]
    flow_cols     = [c for c in cols if "flow" in c.lower() or "gjennomstrøm" in c.lower()]
    pressure_cols = [c for c in cols if "trykk" in c.lower()]
    temp_aux_cols = [c for c in cols if "temperatur" in c.lower() and c not in {target_col, inlet_col, outdoor_col}]

    # Geothermal features from earlier augmentation
    geo_cols = [c for c in ["geo_gradient_C_per_km","geo_heatflow_mW_m2","bore_depth_km","geo_baseline_T_at_depth"] if c in df.columns]

    # Depth / groundwater columns (levels)
    depth_cols = [c for c in cols if is_depth_col(c)]

    groups = {
        "target": [target_col],
        "core": [c for c in [inlet_col, outdoor_col] if c is not None],
        "effect": effect_cols[:6],
        "flow": flow_cols[:3],
        "pressure": pressure_cols[:3],
        "temp_aux": temp_aux_cols[:10],
        "geo": geo_cols,
        "depth": depth_cols[:3],
    }
    return groups

# ================================
# Dataset
# ================================
class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, time_col: str, target: str, features: List[str],
                 seq_len: int, horizon: int, mean: np.ndarray = None, std: np.ndarray = None):
        self.time = df[time_col].to_numpy()
        self.y = df[target].to_numpy(dtype=np.float32)
        self.X = df[features].to_numpy(dtype=np.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        # Standardize features
        if mean is None or std is None:
            self.mean = self.X.mean(axis=0)
            self.std = self.X.std(axis=0) + 1e-8
        else:
            self.mean = mean
            self.std = std
        self.X = (self.X - self.mean) / self.std

        self.valid_idx = []
        max_start = len(self.X) - (seq_len + horizon)
        for i in range(max_start + 1):
            self.valid_idx.append(i)

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        i = self.valid_idx[idx]
        seq = self.X[i:i+self.seq_len]            # (seq_len, features)
        target = self.y[i+self.seq_len+self.horizon-1]  # horizon-ahead target
        # Return shape for CNN1d: (channels=in_features, seq_len)
        seq_ch_first = torch.from_numpy(seq).float().transpose(0,1)  # (features, seq_len)
        return seq_ch_first, torch.tensor(target, dtype=torch.float32)

# ================================
# Model: Hybrid CNN + LSTM
# ================================
class HybridCNNLSTM(nn.Module):
    def __init__(self, in_channels: int, conv_channels: int = 32, lstm_hidden: int = 64, lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        # Conv over time: input (B, C=in_channels, T)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        # LSTM expects (T, B, F); we'll permute to (T, B, C_conv)
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=False)
        self.fc = nn.Linear(lstm_hidden, 1)  # regression

    def forward(self, x):
        # x: (B, C, T)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)            # (B, Cc, T)
        x = x.permute(2, 0, 1)         # (T, B, Cc)
        out, _ = self.lstm(x)          # (T, B, H)
        last = out[-1]                 # (B, H)
        y = self.fc(last).squeeze(-1)  # (B,)
        return y

# ================================
# Training helpers
# ================================
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu"):
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            yh = model(Xb)
            loss = crit(yh, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss += loss.item() * Xb.size(0)
        tr_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                yh = model(Xb)
                loss = crit(yh, yb)
                va_loss += loss.item() * Xb.size(0)
        va_loss /= len(val_loader.dataset)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        if (ep+1) % max(1, epochs//5) == 0:
            print(f"Epoch {ep+1}/{epochs} - train: {tr_loss:.4f}  val: {va_loss:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history

def evaluate_model(model, data_loader, device="cpu") -> Tuple[np.ndarray, np.ndarray, float, float]:
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for Xb, yb in data_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            yh = model(Xb)
            preds.append(yh.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues)**2))
    return trues, preds, mae, rmse

# ================================
# Main
# ================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load data
    df = pd.read_csv(CSV_PATH)
    time_col = detect_time_col(df)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)

    # 2) Select columns
    groups = pick_columns(df)
    target = groups["target"][0]

    # Build feature lists
    base_features = groups["core"] + groups["effect"] + groups["flow"] + groups["pressure"] + groups["temp_aux"]
    geo_depth_features = groups["geo"] + groups["depth"]

    # Depth derivatives (if depth cols present)
    for dc in groups["depth"]:
        df[f"{dc}__d1"] = df[dc].diff()
        geo_depth_features.append(f"{dc}__d1")

    # Delta T between inlet and outlet if both exist
    inlet = None
    for c in groups["core"]:
        if "tur" in c.lower() or "inlet" in c.lower() or "innløp" in c.lower() or "frem" in c.lower():
            inlet = c
            break
    if inlet is not None:
        df["delta_T_in_out"] = df[inlet] - df[target]
        base_features.append("delta_T_in_out")

    # Remove rows with NaNs introduced by diff
    df = df.dropna().reset_index(drop=True)

    # 3) Split by time
    N = len(df)
    test_start = int(N * (1.0 - TEST_SPLIT))
    train_df = df.iloc[:test_start].copy()
    test_df  = df.iloc[test_start:].copy()

    # Train/Val split
    val_size = int(len(train_df) * VAL_SPLIT)
    tr_df = train_df.iloc[:-val_size].copy() if val_size > 0 else train_df.copy()
    va_df = train_df.iloc[-val_size:].copy() if val_size > 0 else train_df.iloc[-1:].copy()

    # Helper to build datasets/loaders
    def make_loaders(feature_list: List[str]):
        tr_ds = SequenceDataset(tr_df, time_col, target, feature_list, SEQ_LEN, PRED_HORIZON)
        va_ds = SequenceDataset(va_df, time_col, target, feature_list, SEQ_LEN, PRED_HORIZON,
                                mean=tr_ds.mean, std=tr_ds.std)
        te_ds = SequenceDataset(test_df, time_col, target, feature_list, SEQ_LEN, PRED_HORIZON,
                                mean=tr_ds.mean, std=tr_ds.std)
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        return tr_loader, va_loader, te_loader, tr_ds, te_ds

    # 4) Model WITH depth/geothermal features
    features_with_depth = base_features + geo_depth_features
    tr_loader, va_loader, te_loader, tr_ds, te_ds = make_loaders(features_with_depth)
    model_with = HybridCNNLSTM(in_channels=len(features_with_depth), conv_channels=32, lstm_hidden=64, lstm_layers=2, dropout=0.1).to(device)
    model_with, hist_with = train_model(model_with, tr_loader, va_loader, epochs=EPOCHS, lr=LR, device=device)
    y_true_with, y_pred_with, mae_with, rmse_with = evaluate_model(model_with, te_loader, device=device)
    torch.save({
        "state_dict": model_with.state_dict(),
        "features": features_with_depth,
        "seq_len": SEQ_LEN,
        "horizon": PRED_HORIZON,
    }, os.path.join(OUTPUT_DIR, "cnn_lstm_with_depth.pth"))

    # 5) Model WITHOUT depth/geothermal features (ablation)
    features_no_depth = base_features
    tr_loader2, va_loader2, te_loader2, tr_ds2, te_ds2 = make_loaders(features_no_depth)
    model_no = HybridCNNLSTM(in_channels=len(features_no_depth), conv_channels=32, lstm_hidden=64, lstm_layers=2, dropout=0.1).to(device)
    model_no, hist_no = train_model(model_no, tr_loader2, va_loader2, epochs=EPOCHS, lr=LR, device=device)
    y_true_no, y_pred_no, mae_no, rmse_no = evaluate_model(model_no, te_loader2, device=device)
    torch.save({
        "state_dict": model_no.state_dict(),
        "features": features_no_depth,
        "seq_len": SEQ_LEN,
        "horizon": PRED_HORIZON,
    }, os.path.join(OUTPUT_DIR, "cnn_lstm_no_depth.pth"))

    # 6) Save metrics
    metrics = {
        "with_depth": {"MAE": float(mae_with), "RMSE": float(rmse_with), "features": features_with_depth},
        "no_depth":   {"MAE": float(mae_no),   "RMSE": float(rmse_no),   "features": features_no_depth},
        "improvement_MAE": float(mae_no - mae_with),
        "improvement_RMSE": float(rmse_no - rmse_with),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics_geothermal.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:", json.dumps(metrics, indent=2))

    # 7) Build a plot: training values + test actual + forecasts (with/without depth)
    # Reconstruct test timestamps
    test_times = test_df[time_col].iloc[SEQ_LEN + PRED_HORIZON - 1:].reset_index(drop=True)
    plt.figure(figsize=(12,4))
    # training history curve (previous training values)
    plt.plot(train_df[time_col], train_df[target], label="Previous training values", linewidth=2)
    # test actual
    plt.plot(test_times, y_true_with, label="Test actual values")
    # forecasts
    plt.plot(test_times, y_pred_with, label="CNN-LSTM (with depth) forecasted values")
    plt.plot(test_times, y_pred_no, label="CNN-LSTM (no depth) forecasted values")
    # split line
    plt.axvline(test_df[time_col].iloc[0], linestyle="--")
    plt.xlabel("Timeline")
    plt.ylabel("Outlet temperature (°C)")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "cnn_lstm_utes_depth_comparison.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Saved plot to: {fig_path}")

    # 8) Optional: Save predictions CSV for inspection
    out_csv = os.path.join(OUTPUT_DIR, "cnn_lstm_predictions_test.csv")
    pd.DataFrame({
        "time": test_times,
        "y_true": y_true_with,
        "y_pred_with_depth": y_pred_with,
        "y_pred_no_depth": y_pred_no
    }).to_csv(out_csv, index=False)
    print(f"Saved predictions to: {out_csv}")
