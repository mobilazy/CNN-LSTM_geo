
import os
import json
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# Configuration
# =============================================================================
"""
Cleaned-up training script (rev02) for Hybrid CNN+LSTM forecaster on geothermal time series.
- Assumes CSV headers are in English (already overwritten in-place).
- Uses fixed column names for time, inlet, outlet (no heuristics).
- Includes:
  * Counterfactual 650 m prediction placeholder & plot
  * Inlet vs Outlet plot for 300 m / 650 m / 1300 m
  * Supplementary combined figure (timeline, inlet–outlet, efficiency vs depth)
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
PATIENCE = int(os.environ.get("PATIENCE", "8"))
USE_SCHEDULER = os.environ.get("USE_SCHEDULER", "false").lower() in {"1", "true", "yes"}

# Fixed column names (as overwritten in your CSVs)
TIME_COL = "timestamp"
INLET_COL = "Energy_meter_energy_wells_inlet_temperature_C"
OUTLET_COL = "Energy_meter_energy_wells_return_temperature_C"

# Optional depth column name expected for counterfactuals (set to your dataset's actual depth feature if present)
DEPTH_COL = "bore_depth_km"  # keep as-is if available; counterfactuals will silently skip if absent

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
    def __init__(
        self,
        df: pd.DataFrame,
        time_col: str,
        target: str,
        features: List[str],
        seq_len: int,
        horizon: int,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
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

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int):
        i = self.valid_idx[idx]
        seq = self.X[i : i + self.seq_len]
        target = self.y[i + self.seq_len + self.horizon - 1]
        seq_ch_first = torch.from_numpy(seq).float().transpose(0, 1)  # (features, seq_len)
        return seq_ch_first, torch.tensor(target, dtype=torch.float32)

# =============================================================================
# Model
# =============================================================================
class HybridCNNLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_channels: List[int] = (32, 32),
        kernel_size: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = [in_channels] + list(conv_channels)
        convs = []
        for i in range(len(channels) - 1):
            convs += [
                nn.Conv1d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.ReLU(),
            ]
        self.conv = nn.Sequential(*convs)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=channels[-1], hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=False)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.dropout(x)
        x = x.permute(2, 0, 1)
        out, _ = self.lstm(x)
        last = out[-1]
        y = self.fc(last).squeeze(-1)
        return y

# =============================================================================
# Training helpers
# =============================================================================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 8,
    use_scheduler: bool = False,
    log_prefix: str = "",
) -> tuple[nn.Module, Dict[str, list]]:
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=max(1, patience // 2))
        if use_scheduler else None
    )

    best_val = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": []}

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [train]", leave=False):
            Xb = Xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            yh = model(Xb)
            loss = crit(yh, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss += loss.item() * Xb.size(0)
        tr_loss /= max(1, len(train_loader.dataset))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for Xb, yb in tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [val]", leave=False):
                Xb = Xb.to(device); yb = yb.to(device)
                yh = model(Xb)
                loss = crit(yh, yb)
                va_loss += loss.item() * Xb.size(0)
        va_loss /= max(1, len(val_loader.dataset))

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        logging.info(f"{log_prefix}Epoch {ep}/{epochs} - train: {tr_loss:.5f}  val: {va_loss:.5f}")

        if scheduler is not None:
            scheduler.step(va_loss)

        if va_loss + 1e-9 < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logging.info(f"Early stopping at epoch {ep} (no val improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    return model, history

def evaluate_model(model: nn.Module, data_loader: DataLoader, device: str = "cpu"):
    model.eval()
    preds = []; trues = []
    with torch.no_grad():
        for Xb, yb in tqdm(data_loader, desc="Evaluating", leave=False):
            Xb = Xb.to(device); yb = yb.to(device)
            yh = model(Xb)
            preds.append(yh.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds) if preds else np.array([])
    trues = np.concatenate(trues) if trues else np.array([])
    mae = float(np.mean(np.abs(preds - trues))) if len(preds) else float("nan")
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2))) if len(preds) else float("nan")
    return trues, preds, mae, rmse

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # 1) Load data
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # Enforce required columns (no heuristics)
    required_cols = [TIME_COL, INLET_COL, OUTLET_COL]
    for rc in required_cols:
        if rc not in df.columns:
            raise RuntimeError(f"Required column missing: {rc}")

    # Parse time and sort
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.sort_values(TIME_COL).dropna(subset=[TIME_COL]).reset_index(drop=True)

    # 2) Build features
    target = OUTLET_COL
    inlet = INLET_COL

    # core features: inlet + optional outdoor temperature if present
    core_feats = [inlet]
    if "outdoor_temperature_C" in df.columns and "outdoor_temperature_C" != target:
        core_feats.append("outdoor_temperature_C")

    # effect/power, flow, pressure, aux temps (generic filters on Englishized headers)
    effect_cols = [c for c in df.columns if "power" in c.lower() or c.lower().endswith("_kw") or "heat" in c.lower()]
    flow_cols = [c for c in df.columns if "flow" in c.lower() or "throughput" in c.lower()]
    pressure_cols = [c for c in df.columns if "pressure" in c.lower()]
    temp_aux_cols = [c for c in df.columns if "temperature" in c.lower() and c not in {target, inlet, "outdoor_temperature_C"}]

    # geothermal/depth features (if present)
    geo_cols = [c for c in [
        "geo_gradient_C_per_km",
        "geo_heatflow_mW_m2",
        "bore_depth_km",
        "geo_baseline_T_at_depth",
    ] if c in df.columns]

    # delta T
    df["delta_T_in_out"] = df[inlet] - df[target]

    base_features = core_feats + effect_cols[:6] + flow_cols[:3] + pressure_cols[:3] + temp_aux_cols[:10]
    if "delta_T_in_out" not in base_features:
        base_features.append("delta_T_in_out")

    geo_depth_features = geo_cols.copy()

    # Derivative for depth-like columns if present
    if DEPTH_COL in df.columns:
        df[f"{DEPTH_COL}__d1"] = df[DEPTH_COL].diff()
        geo_depth_features.append(f"{DEPTH_COL}__d1")

    # Remove NaNs after diff
    df = df.dropna().reset_index(drop=True)

    # 3) Split by time
    N = len(df)
    if N < (SEQ_LEN + PRED_HORIZON + 1):
        raise SystemExit("Dataset too small after preprocessing.")

    test_start = int(N * (1.0 - TEST_SPLIT))
    test_start = max(test_start, SEQ_LEN + PRED_HORIZON)
    train_df = df.iloc[:test_start].copy()
    test_df = df.iloc[test_start:].copy()

    # Train/Val split within train_df
    val_size = max(1, int(len(train_df) * VAL_SPLIT))
    tr_df = train_df.iloc[:-val_size].copy()
    va_df = train_df.iloc[-val_size:].copy()

    def make_loaders(feature_list: List[str]):
        missing = [c for c in feature_list + [target] if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required columns: {missing}")
        tr_ds = SequenceDataset(tr_df, TIME_COL, target, feature_list, SEQ_LEN, PRED_HORIZON)
        va_ds = SequenceDataset(va_df, TIME_COL, target, feature_list, SEQ_LEN, PRED_HORIZON, mean=tr_ds.mean, std=tr_ds.std)
        te_ds = SequenceDataset(test_df, TIME_COL, target, feature_list, SEQ_LEN, PRED_HORIZON, mean=tr_ds.mean, std=tr_ds.std)
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        return tr_loader, va_loader, te_loader, tr_ds, te_ds

    # 4) Train WITH depth/geothermal features
    features_with_depth = base_features + geo_depth_features
    tr_loader, va_loader, te_loader, tr_ds, te_ds = make_loaders(features_with_depth)

    model_with = HybridCNNLSTM(
        in_channels=len(features_with_depth),
        conv_channels=CONV_CHANNELS,
        kernel_size=KERNEL_SIZE,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    model_with, hist_with = train_model(
        model_with, tr_loader, va_loader, epochs=EPOCHS, lr=LR, device=device,
        patience=PATIENCE, use_scheduler=USE_SCHEDULER, log_prefix="with_depth | "
    )
    y_true_with, y_pred_with, mae_with, rmse_with = evaluate_model(model_with, te_loader, device=device)

    torch.save(
        {"state_dict": model_with.state_dict(), "features": features_with_depth, "seq_len": SEQ_LEN, "horizon": PRED_HORIZON},
        os.path.join(OUTPUT_DIR, "cnn_lstm_with_depth_eng.pth"),
    )

    # 5) Train WITHOUT depth/geothermal features (ablation)
    features_no_depth = base_features
    tr_loader2, va_loader2, te_loader2, tr_ds2, te_ds2 = make_loaders(features_no_depth)

    model_no = HybridCNNLSTM(
        in_channels=len(features_no_depth),
        conv_channels=CONV_CHANNELS,
        kernel_size=KERNEL_SIZE,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    model_no, hist_no = train_model(
        model_no, tr_loader2, va_loader2, epochs=EPOCHS, lr=LR, device=device,
        patience=PATIENCE, use_scheduler=USE_SCHEDULER, log_prefix="no_depth | "
    )
    y_true_no, y_pred_no, mae_no, rmse_no = evaluate_model(model_no, te_loader2, device=device)

    # 6) Save metrics
    metrics = {
        "with_depth": {"MAE": float(mae_with), "RMSE": float(rmse_with), "features": features_with_depth},
        "no_depth": {"MAE": float(mae_no), "RMSE": float(rmse_no), "features": features_no_depth},
        "improvement_MAE": float(mae_no - mae_with),
        "improvement_RMSE": float(rmse_no - rmse_with),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics_geothermal_eng.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Metrics saved.")

    # 7) Plot comparison (timeline)
    test_times = test_df[TIME_COL].iloc[SEQ_LEN + PRED_HORIZON - 1 :].reset_index(drop=True)
    plt.figure(figsize=(12, 4))
    plt.plot(train_df[TIME_COL], train_df[target], label="Previous training values", linewidth=2)
    plt.plot(test_times, y_true_with, label="Test actual values")
    plt.plot(test_times, y_pred_with, label="CNN-LSTM (with depth) forecasted values")
    plt.plot(test_times, y_pred_no, label="CNN-LSTM (no depth) forecasted values")
    plt.axvline(test_df[TIME_COL].iloc[0], linestyle="--")
    plt.xlabel("Timeline"); plt.ylabel("Outlet temperature (°C)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cnn_lstm_depth_comparison_eng.png"), dpi=200)

    # 8) Save predictions CSV
    out_csv = os.path.join(OUTPUT_DIR, "cnn_lstm_predictions_test_eng.csv")
    pd.DataFrame({
        "time": test_times, "y_true": y_true_with,
        "y_pred_with_depth": y_pred_with, "y_pred_no_depth": y_pred_no
    }).to_csv(out_csv, index=False)

    # -------------------------------------------------------------------------
    # PLACEHOLDER: Real 650 m measurements (enable later)
    # CSV_PATH_650 = os.environ.get("CSV_PATH_650", os.path.join(os.path.dirname(__file__), "EDE_650m_measurements_eng.csv"))
    # if os.path.exists(CSV_PATH_650):
    #     df_650 = pd.read_csv(CSV_PATH_650)
    #     df_650[TIME_COL] = pd.to_datetime(df_650[TIME_COL], errors="coerce")
    #     df_650 = df_650.sort_values(TIME_COL).dropna(subset=[TIME_COL]).reset_index(drop=True)
    # -------------------------------------------------------------------------

    # 9) Counterfactual/interpolated 650 m prediction
    try:
        if DEPTH_COL in test_df.columns:
            test_df_cf_065 = test_df.copy()
            test_df_cf_065[DEPTH_COL] = 0.65  # 650 m
        else:
            test_df_cf_065 = test_df.copy()

        te_cf_ds_065 = SequenceDataset(test_df_cf_065, TIME_COL, target, features_with_depth, SEQ_LEN, PRED_HORIZON, mean=tr_ds.mean, std=tr_ds.std)
        te_cf_loader_065 = DataLoader(te_cf_ds_065, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        y_true_cf_065, y_pred_cf_065, mae_cf_065, rmse_cf_065 = evaluate_model(model_with, te_cf_loader_065, device=device)

        cf_065_csv = os.path.join(OUTPUT_DIR, "counterfactual_650m_predictions_eng.csv")
        pd.DataFrame({
            "time": test_times,
            "y_true_reference": y_true_with,
            "y_pred_650m_counterfactual": y_pred_cf_065
        }).to_csv(cf_065_csv, index=False)

        plt.figure(figsize=(12, 4))
        plt.plot(test_times, y_true_with, label="Test actual values (reference depth)")
        plt.plot(test_times, y_pred_cf_065, label="Predicted outlet (counterfactual 650 m)", linewidth=2)
        plt.xlabel("Timeline"); plt.ylabel("Outlet temperature (°C)")
        plt.title("Counterfactual prediction at 650 m (to be validated with future data)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "counterfactual_650m_timeline_eng.png"), dpi=200)
    except Exception as e:
        logging.exception(f"Counterfactual 650 m failed: {e}")

    # 10) Inlet vs. Outlet curves for 300 m, 650 m, 1300 m
    try:
        depths_km = [0.30, 0.65, 1.30]
        pred_by_depth: Dict[float, np.ndarray] = {}

        for dkm in depths_km:
            cf_df = test_df.copy()
            if DEPTH_COL in cf_df.columns:
                cf_df[DEPTH_COL] = dkm
            ds = SequenceDataset(cf_df, TIME_COL, target, features_with_depth, SEQ_LEN, PRED_HORIZON, mean=tr_ds.mean, std=tr_ds.std)
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
            _, ypred, _, _ = evaluate_model(model_with, dl, device=device)
            pred_by_depth[dkm] = ypred

        inlet_aligned = test_df[inlet].iloc[SEQ_LEN + PRED_HORIZON - 1 :].to_numpy()

        plt.figure(figsize=(8, 6))
        for dkm in depths_km:
            plt.scatter(inlet_aligned, pred_by_depth[dkm], s=10, alpha=0.6, label=f"{int(dkm*1000)} m")
        # Optional smoothing
        try:
            for dkm in depths_km:
                x = inlet_aligned; y = pred_by_depth[dkm]
                if len(x) >= 10:
                    coeff = np.polyfit(x, y, deg=2)
                    xx = np.linspace(float(np.min(x)), float(np.max(x)), 200)
                    yy = np.polyval(coeff, xx)
                    plt.plot(xx, yy, linewidth=2)
        except Exception as _fit_err:
            logging.warning(f"Polyfit smoothing skipped: {_fit_err}")
        plt.xlabel(f"Inlet temperature ({inlet}) (°C)")
        plt.ylabel("Predicted outlet temperature (°C)")
        plt.title("Inlet vs Outlet — effect of BHE depth (300 m / 650 m / 1300 m)")
        plt.legend(title="BHE depth"); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "inlet_vs_outlet_by_depth_eng.png"), dpi=200)

        io_csv = os.path.join(OUTPUT_DIR, "inlet_vs_outlet_by_depth_eng.csv")
        pd.DataFrame({
            "time": test_times,
            "inlet_aligned": inlet_aligned,
            "y_pred_300m": pred_by_depth.get(0.30),
            "y_pred_650m": pred_by_depth.get(0.65),
            "y_pred_1300m": pred_by_depth.get(1.30),
        }).to_csv(io_csv, index=False)
    except Exception as e:
        logging.exception(f"Inlet–Outlet depth plot failed: {e}")

    # 11) Supplementary combined figure
    try:
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)

        # (A) Timeline
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(test_times, y_true_with, label="Actual (reference depth)")
        if 'y_pred_cf_065' in locals():
            ax1.plot(test_times, y_pred_cf_065, label="Predicted (counterfactual 650 m)", linewidth=2)
        ax1.set_title("Timeline – Counterfactual 650 m")
        ax1.set_xlabel("Time"); ax1.set_ylabel("Outlet Temp (°C)"); ax1.legend()

        # (B) Inlet vs Outlet
        ax2 = fig.add_subplot(gs[1, 0])
        for dkm in [0.30, 0.65, 1.30]:
            if 'pred_by_depth' in locals() and dkm in pred_by_depth:
                ax2.scatter(inlet_aligned, pred_by_depth[dkm], s=8, alpha=0.6, label=f"{int(dkm*1000)} m")
        ax2.set_title("Inlet vs Outlet by depth")
        ax2.set_xlabel("Inlet temperature (°C)"); ax2.set_ylabel("Predicted outlet temperature (°C)"); ax2.legend()

        # (C) Efficiency vs Depth (relative to 650 m)
        ax3 = fig.add_subplot(gs[1, 1])
        if 'pred_by_depth' in locals() and 0.65 in pred_by_depth:
            rmse_baseline = float(np.sqrt(np.mean((pred_by_depth[0.65] - y_true_with) ** 2)))
            xs = []; ys = []
            for dkm, preds in pred_by_depth.items():
                rmse_d = float(np.sqrt(np.mean((preds - y_true_with) ** 2)))
                eff = 1 - rmse_d / rmse_baseline
                xs.append(dkm * 1000); ys.append(eff)
            ax3.plot(xs, ys, marker="o", linewidth=2)
            ax3.set_title("Relative Heat Retention Efficiency vs Depth (650 m baseline)")
            ax3.set_xlabel("BHE depth (m)"); ax3.set_ylabel("Relative efficiency")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "supplementary_combined_view_eng.png"), dpi=200)
    except Exception as e:
        logging.exception(f"Supplementary figure failed: {e}")
