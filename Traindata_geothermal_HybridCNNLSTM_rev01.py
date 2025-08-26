import os
import math
import json
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# Configuration (env-overridable)
# =============================================================================
"""Training script for a Hybrid CNN+LSTM forecaster on geothermal time series.

This version expects English column names / parameters (e.g., after translation)
 and defaults to the *_eng.csv files.

Key improvements:
- Docstrings for clarity
- Robust error handling for file reading and column selection
- Parameterized model architecture (kernel size, channels, dropout, etc.)
- Early stopping and optional learning-rate scheduler
- Training progress logging to a file, plus tqdm progress bars
"""

CSV_PATH = os.environ.get(
    "CSV_PATH",
    os.path.join(os.path.dirname(__file__), "EDE_with_geothermal_features_eng.csv"),
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "output")
)
SEQ_LEN = int(os.environ.get("SEQ_LEN", "48"))  # timesteps per sample
PRED_HORIZON = int(os.environ.get("PRED_HORIZON", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
LR = float(os.environ.get("LR", "1e-3"))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", "0.1"))
TEST_SPLIT = float(os.environ.get("TEST_SPLIT", "0.2"))

# Model hyperparams (env-overridable)
CONV_CHANNELS = os.environ.get("CONV_CHANNELS", "32,32")  # comma-separated
KERNEL_SIZE = int(os.environ.get("KERNEL_SIZE", "3"))
LSTM_HIDDEN = int(os.environ.get("LSTM_HIDDEN", "64"))
LSTM_LAYERS = int(os.environ.get("LSTM_LAYERS", "2"))
DROPOUT = float(os.environ.get("DROPOUT", "0.1"))
PATIENCE = int(os.environ.get("PATIENCE", "8"))  # early stopping patience (epochs)
USE_SCHEDULER = os.environ.get("USE_SCHEDULER", "false").lower() in {"1", "true", "yes"}

# =============================================================================
# Logging
# =============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_path = os.path.join(OUTPUT_DIR, "train.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("train")


# =============================================================================
# Utilities
# =============================================================================

def detect_time_col(df: pd.DataFrame) -> str:
    """Detect a reasonable time column.

    Heuristics: prefer columns containing 'time', 'timestamp', or 'date'.
    Falls back to the first column if nothing matches.
    """
    candidates = []
    for c in df.columns:
        s = c.lower()
        if any(k in s for k in ["timestamp", "time", "date"]):
            candidates.append(c)
    return candidates[0] if candidates else df.columns[0]


def is_depth_col(name: str) -> bool:
    """Return True if column name looks like a depth/water-level/groundwater signal.

    Supports both English and possible leftover Norwegian tokens.
    """
    s = name.lower()
    tokens = [
        "depth", "level", "waterlevel", "water_level", "groundwater",
        "nivå", "vannstand", "grunnvann", "brønn", "bronn",
    ]
    return any(t in s for t in tokens)


def pick_columns(df: pd.DataFrame, time_col: str) -> Dict[str, List[str]]:
    """Pick target and feature groups based on English column names.

    Expected target keywords (case-insensitive):
      - 'outlet temperature', 'return temperature', 'outlet', 'return'
    Expected inlet/supply keywords:
      - 'inlet', 'supply', 'forward'
    Also gathers effects/power, flow, pressure, auxiliary temperatures, geothermal features,
    and depth/groundwater level columns.

    Raises:
        RuntimeError: if target column cannot be determined.
    """
    cols = [c for c in df.columns if c != time_col]

    # Target: outlet / return temperature
    target_col: Optional[str] = None
    target_keys = [
        "outlet temperature", "return temperature", "outlet_temperature",
        "return_temperature", "outlet", "return",
    ]
    for k in target_keys:
        for c in cols:
            if k in c.lower():
                target_col = c
                break
        if target_col:
            break
    if target_col is None:
        # fallback -> any temperature column
        for c in cols:
            if "temperature" in c.lower():
                target_col = c
                break
    if target_col is None:
        raise RuntimeError("Could not find a target (outlet/return temperature) column.")

    # Inlet / supply temperature
    inlet_col: Optional[str] = None
    inlet_keys = ["inlet", "supply", "forward", "supply temperature", "inlet_temperature"]
    for k in inlet_keys:
        if inlet_col:
            break
        for c in cols:
            if k in c.lower():
                inlet_col = c
                break

    # Outdoor temperature
    outdoor_col: Optional[str] = None
    for c in cols:
        if "outdoor" in c.lower() or "ambient" in c.lower():
            outdoor_col = c
            break

    # Power/effect, flow, pressures, aux temps
    effect_cols = [
        c for c in cols if ("power" in c.lower() or c.lower().endswith(" kw") or "heat" in c.lower())
    ]
    flow_cols = [c for c in cols if "flow" in c.lower() or "throughput" in c.lower()]
    pressure_cols = [c for c in cols if "pressure" in c.lower()]
    temp_aux_cols = [
        c
        for c in cols
        if ("temperature" in c.lower() and c not in {target_col, inlet_col, outdoor_col})
    ]

    # Geothermal features (post-translation expected names)
    geo_cols = [
        c
        for c in [
            "geo_gradient_C_per_km",
            "geo_heatflow_mW_m2",
            "bore_depth_km",
            "geo_baseline_T_at_depth",
        ]
        if c in df.columns
    ]

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


# =============================================================================
# Dataset
# =============================================================================
class SequenceDataset(Dataset):
    """Windowed time-series dataset producing (sequence, horizon target) pairs.

    Args:
        df: Full dataframe (chronologically sorted).
        time_col: Name of the time column.
        target: Target column name (regression).
        features: List of feature column names.
        seq_len: Number of past steps per sample.
        horizon: Forecast horizon (steps ahead).
        mean/std: Optional standardization stats. If not provided, computed on df.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        time_col: str,
        target: str,
        features: List[str],
        seq_len: int,
        horizon: int,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
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
        for i in range(max(0, max_start) + 1):
            self.valid_idx.append(i)

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int):
        i = self.valid_idx[idx]
        seq = self.X[i : i + self.seq_len]  # (seq_len, features)
        target = self.y[i + self.seq_len + self.horizon - 1]
        # Return shape for CNN1d: (channels=in_features, seq_len)
        seq_ch_first = torch.from_numpy(seq).float().transpose(0, 1)  # (features, seq_len)
        return seq_ch_first, torch.tensor(target, dtype=torch.float32)


# =============================================================================
# Model: Flexible Hybrid CNN + LSTM
# =============================================================================
class HybridCNNLSTM(nn.Module):
    """Hybrid temporal model: stacks 1D conv layers over time, then an LSTM.

    Args:
        in_channels: Number of input feature channels.
        conv_channels: Sequence of conv output channels per layer.
        kernel_size: 1D conv kernel size.
        lstm_hidden: LSTM hidden units.
        lstm_layers: Number of LSTM layers.
        dropout: Dropout probability after conv stack.
    """

    def __init__(
        self,
        in_channels: int,
        conv_channels: List[int] | Tuple[int, ...] = (32, 32),
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
        self.lstm = nn.LSTM(
            input_size=channels[-1], hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=False
        )
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.conv(x)
        x = self.dropout(x)  # (B, Cc, T)
        x = x.permute(2, 0, 1)  # (T, B, Cc)
        out, _ = self.lstm(x)  # (T, B, H)
        last = out[-1]  # (B, H)
        y = self.fc(last).squeeze(-1)  # (B,)
        return y


# =============================================================================
# Training helpers
# =============================================================================

def parse_channels(spec: str) -> List[int]:
    """Parse a comma-separated channels spec (e.g., "32,64,64") into a list of ints."""
    return [int(x) for x in spec.split(",") if str(x).strip()]


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
    """Train model with early stopping and optional LR scheduler.

    Returns the best-scoring (val MSE) model state and training history.
    """
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=max(1, patience // 2))
        if use_scheduler
        else None
    )

    best_val = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": []}

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [train]", leave=False)
        for Xb, yb in pbar:
            Xb = Xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            yh = model(Xb)
            loss = crit(yh, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss += loss.item() * Xb.size(0)
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
        tr_loss /= max(1, len(train_loader.dataset))

        # Validation
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for Xb, yb in tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [val]", leave=False):
                Xb = Xb.to(device)
                yb = yb.to(device)
                yh = model(Xb)
                loss = crit(yh, yb)
                va_loss += loss.item() * Xb.size(0)
        va_loss /= max(1, len(val_loader.dataset))

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        logger.info(
            f"{log_prefix}Epoch {ep}/{epochs} - train: {tr_loss:.5f}  val: {va_loss:.5f}  lr: {opt.param_groups[0]['lr']:.2e}"
        )

        if scheduler is not None:
            scheduler.step(va_loss)

        # Early stopping
        if va_loss + 1e-9 < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info(f"Early stopping at epoch {ep} (no val improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    return model, history


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Run inference and compute MAE/RMSE."""
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for Xb, yb in tqdm(data_loader, desc="Evaluating", leave=False):
            Xb = Xb.to(device)
            yb = yb.to(device)
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
    logger.info(f"Using device: {device}")

    # 1) Load data with error handling
    try:
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        logger.exception(f"Failed to read CSV: {e}")
        raise SystemExit(1)

    # Time parsing
    time_col = detect_time_col(df)
    try:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col).reset_index(drop=True)
        df = df.dropna(subset=[time_col])
    except Exception as e:
        logger.exception(f"Failed to parse/sort time column '{time_col}': {e}")
        raise SystemExit(1)

    # 2) Select columns (robust)
    try:
        groups = pick_columns(df, time_col)
    except Exception as e:
        logger.exception(f"Column selection failed: {e}")
        raise SystemExit(1)

    target = groups["target"][0]

    # Build feature lists
    base_features = groups["core"] + groups["effect"] + groups["flow"] + groups["pressure"] + groups["temp_aux"]
    geo_depth_features = groups["geo"] + groups["depth"]

    # Depth derivatives (if depth cols present)
    for dc in groups["depth"]:
        try:
            df[f"{dc}__d1"] = df[dc].diff()
            geo_depth_features.append(f"{dc}__d1")
        except Exception as e:
            logger.warning(f"Could not build derivative for depth column '{dc}': {e}")

    # Delta T between inlet and outlet if both exist
    inlet = None
    for c in groups["core"]:
        if any(k in c.lower() for k in ["inlet", "supply", "forward"]):
            inlet = c
            break
    if inlet is not None:
        df["delta_T_in_out"] = df[inlet] - df[target]
        base_features.append("delta_T_in_out")

    # Remove rows with NaNs introduced by diff
    df = df.dropna().reset_index(drop=True)

    # 3) Split by time
    N = len(df)
    if N < (SEQ_LEN + PRED_HORIZON + 1):
        logger.error("Dataset too small after preprocessing.")
        raise SystemExit(1)

    test_start = int(N * (1.0 - TEST_SPLIT))
    test_start = max(test_start, SEQ_LEN + PRED_HORIZON)
    train_df = df.iloc[:test_start].copy()
    test_df = df.iloc[test_start:].copy()

    # Train/Val split
    val_size = int(len(train_df) * VAL_SPLIT)
    if val_size == 0:
        logger.warning("VAL_SPLIT produced 0 validation samples; using last 1% as validation.")
        val_size = max(1, int(0.01 * len(train_df)))
    tr_df = train_df.iloc[:-val_size].copy()
    va_df = train_df.iloc[-val_size:].copy()

    def make_loaders(feature_list: List[str]):
        """Construct datasets and dataloaders, reusing train stats for val/test."""
        missing = [c for c in feature_list + [target] if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required columns: {missing}")
        tr_ds = SequenceDataset(tr_df, time_col, target, feature_list, SEQ_LEN, PRED_HORIZON)
        va_ds = SequenceDataset(
            va_df, time_col, target, feature_list, SEQ_LEN, PRED_HORIZON, mean=tr_ds.mean, std=tr_ds.std
        )
        te_ds = SequenceDataset(
            test_df, time_col, target, feature_list, SEQ_LEN, PRED_HORIZON, mean=tr_ds.mean, std=tr_ds.std
        )
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        return tr_loader, va_loader, te_loader, tr_ds, te_ds

    # 4) Model WITH depth/geothermal features
    features_with_depth = base_features + geo_depth_features
    tr_loader, va_loader, te_loader, tr_ds, te_ds = make_loaders(features_with_depth)

    conv_channels = parse_channels(CONV_CHANNELS)
    model_with = HybridCNNLSTM(
        in_channels=len(features_with_depth),
        conv_channels=conv_channels,
        kernel_size=KERNEL_SIZE,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    model_with, hist_with = train_model(
        model_with,
        tr_loader,
        va_loader,
        epochs=EPOCHS,
        lr=LR,
        device=device,
        patience=PATIENCE,
        use_scheduler=USE_SCHEDULER,
        log_prefix="with_depth | ",
    )
    y_true_with, y_pred_with, mae_with, rmse_with = evaluate_model(model_with, te_loader, device=device)

    torch.save(
        {
            "state_dict": model_with.state_dict(),
            "features": features_with_depth,
            "seq_len": SEQ_LEN,
            "horizon": PRED_HORIZON,
        },
        os.path.join(OUTPUT_DIR, "cnn_lstm_with_depth_eng.pth"),
    )

    # 5) Model WITHOUT depth/geothermal features (ablation)
    features_no_depth = base_features
    tr_loader2, va_loader2, te_loader2, tr_ds2, te_ds2 = make_loaders(features_no_depth)

    model_no = HybridCNNLSTM(
        in_channels=len(features_no_depth),
        conv_channels=conv_channels,
        kernel_size=KERNEL_SIZE,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    model_no, hist_no = train_model(
        model_no,
        tr_loader2,
        va_loader2,
        epochs=EPOCHS,
        lr=LR,
        device=device,
        patience=PATIENCE,
        use_scheduler=USE_SCHEDULER,
        log_prefix="no_depth | ",
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
    logger.info("Metrics:\n" + json.dumps(metrics, indent=2))

    # 7) Plot comparison
    test_times = test_df[time_col].iloc[SEQ_LEN + PRED_HORIZON - 1 :].reset_index(drop=True)
    plt.figure(figsize=(12, 4))
    plt.plot(train_df[time_col], train_df[target], label="Previous training values", linewidth=2)
    plt.plot(test_times, y_true_with, label="Test actual values")
    plt.plot(test_times, y_pred_with, label="CNN-LSTM (with depth) forecasted values")
    plt.plot(test_times, y_pred_no, label="CNN-LSTM (no depth) forecasted values")
    plt.axvline(test_df[time_col].iloc[0], linestyle="--")
    plt.xlabel("Timeline")
    plt.ylabel("Outlet temperature (°C)")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "cnn_lstm_depth_comparison_eng.png")
    plt.savefig(fig_path, dpi=200)
    logger.info(f"Saved plot to: {fig_path}")

    # 8) Save predictions CSV for inspection
    out_csv = os.path.join(OUTPUT_DIR, "cnn_lstm_predictions_test_eng.csv")
    pd.DataFrame(
        {
            "time": test_times,
            "y_true": y_true_with,
            "y_pred_with_depth": y_pred_with,
            "y_pred_no_depth": y_pred_no,
        }
    ).to_csv(out_csv, index=False)
    logger.info(f"Saved predictions to: {out_csv}")
