"""RT514 sensor recovery using trained CNN-LSTM model.

BENCHMARK CONSISTENCY WITH REV10:
This implementation maximizes code reuse from Traindata_geothermal_HybridCNNLSTM_rev10_final.py
to ensure consistency with the main study:

From rev10:
- Uses rev10.ComprehensiveCNNLSTM architecture (6 input features instead of 4)
- Uses rev10.ComprehensiveDataset for data preparation
- Uses rev10.train_model() for training loop
- Uses rev10.load_double_u45mm_research_data() for OE403 data
- Uses rev10's hyperparameters (EPOCHS, LR, PATIENCE, BATCH_SIZE, SEQ_LEN, etc.)
- Uses rev10 model architecture parameters (CONV_CHANNELS, KERNEL_SIZE, LSTM_HIDDEN, etc.)

Key Difference:
- Input features: 6 (supply_temp, flow_rate/4, power/4, RT512, RT513, RT515)
  vs. main study's 4 (supply_temp, flow_rate, power, bhe_type)
- Target: Individual sensor RT514 vs. aggregated manifold return temp
- Dataset: Pre-degradation period only (before Sept 8, 2025) for training

EVALUATION APPROACH:
- Train on pre-degradation data (RT514 healthy)
- Evaluate predictions against actual RT514 during pre-degradation period
- This validates model accuracy during the period when RT514 was still reliable
- Predictions start after first 4-hour window (SEQ_LEN=48 timesteps)
- Post-degradation predictions cannot be validated (no ground truth)

This ensures that any performance differences between sensor recovery and 
the main model reflect the task difference (individual vs. aggregated prediction)
rather than implementation inconsistencies.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import rev10 for consistency - this is the BENCHMARK
import Traindata_geothermal_HybridCNNLSTM_rev10_final as rev10


TARGET_SENSOR = "737.003-RT514 [°C]"
ALL_SENSORS = [
    "737.003-RT512 [°C]",
    "737.003-RT513 [°C]",
    "737.003-RT514 [°C]",
    "737.003-RT515 [°C]",
]
FAULT_THRESHOLD = -10.0
# When the sensor degradation started
DEGRADATION_TIME = pd.Timestamp("2025-09-08 21:45")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover RT514 using trained CNN-LSTM model")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).resolve().parent / "input" / "DoubleU45_Treturn.csv",
        help="Path to Double U-45 return temperature CSV",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(rev10.OUTPUT_DIR) / "rt514_recovery_model.pth",
        help="Path to trained sensor recovery model",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(rev10.OUTPUT_DIR) / "rt514_sensor_recovery.png",
        help="Where to save the recovery plot",
    )
    parser.add_argument(
        "--days-before-fault",
        type=int,
        default=32,
        help="Days of data to show before fault (for context)",
    )
    parser.add_argument(
        "--days-after-fault",
        type=int,
        default=8,
        help="Days of prediction after fault",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new model instead of using existing one (uses rev10.EPOCHS)",
    )
    return parser.parse_args()


def load_sensor_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Sensor data not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip().replace("�", "°") for col in df.columns]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d.%m.%Y %H:%M", errors="coerce")
    df = df.sort_values("Timestamp").dropna(subset=["Timestamp"]).reset_index(drop=True)

    for col in ALL_SENSORS:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def detect_fault_time(df: pd.DataFrame) -> pd.Timestamp:
    fault_mask = df[TARGET_SENSOR] < FAULT_THRESHOLD
    if not fault_mask.any():
        raise ValueError(f"No malfunction detected")
    
    fault_time = df.loc[fault_mask, "Timestamp"].iloc[0]
    logging.info(f"Fault detected: {fault_time}")
    logging.info(f"RT514 value: {df.loc[fault_mask, TARGET_SENSOR].iloc[0]:.1f}°C")
    return fault_time


def load_and_prepare_cnn_lstm_model(model_path: Path, device: torch.device):
    """Load trained CNN-LSTM model for sensor recovery (3 features: physics-based)."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load checkpoint containing model and normalization parameters
    checkpoint = torch.load(model_path, map_location=device)
    
    # Model uses 3 features: supply_temp, flow_rate, power_kw
    model = rev10.ComprehensiveCNNLSTM(
        input_features=3,
        conv_channels=rev10.CONV_CHANNELS,
        kernel_size=rev10.KERNEL_SIZE,
        lstm_hidden=rev10.LSTM_HIDDEN,
        lstm_layers=rev10.LSTM_LAYERS,
        dropout=rev10.DROPOUT
    ).to(device)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        feature_mean = checkpoint['feature_mean']
        feature_std = checkpoint['feature_std']
    else:
        # Legacy format (just state dict)
        model.load_state_dict(checkpoint)
        feature_mean = None
        feature_std = None
        logging.warning("Loaded legacy model format - normalization parameters not saved")
    
    model.eval()
    logging.info(f"Loaded CNN-LSTM model from {model_path}")
    
    # Load test_start if available
    test_start = None
    if 'test_start' in checkpoint:
        test_start = pd.Timestamp(checkpoint['test_start'])
    
    return model, feature_mean, feature_std, test_start


def load_oe403_data(csv_path: Path) -> pd.DataFrame:
    """Load OE403 energy meter data using rev10's function."""
    oe403_df = rev10.load_double_u45mm_research_data()
    
    if oe403_df.empty:
        raise FileNotFoundError(f"OE403 data not found or failed to load")
    
    # Convert timezone-aware timestamps to timezone-naive for merge compatibility
    if oe403_df["Timestamp"].dt.tz is not None:
        oe403_df["Timestamp"] = oe403_df["Timestamp"].dt.tz_localize(None)
    
    oe403_df = oe403_df.rename(columns={
        "flow_rate": "Flow [m³/h]",
        "power_kw": "Power [kW]"
    })
    
    logging.info(f"Loaded OE403 data: {len(oe403_df)} records")
    return oe403_df


def reconstruct_features_from_sensors(df: pd.DataFrame, oe403_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct input features from sensor data (physics-based: operational parameters only).
    
    Args:
        df: Sensor dataframe with RT512, RT513, RT514, RT515
        oe403_df: OE403 meter data with supply_temp, flow, power
    
    Returns 3 features: supply_temp, flow_rate, power_kw
    """
    df_feat = df.copy()
    
    df_feat = df_feat.merge(
        oe403_df[["Timestamp", "supply_temp", "Flow [m³/h]", "Power [kW]"]],
        on="Timestamp",
        how="left"
    )
    
    df_feat["flow_rate"] = df_feat["Flow [m³/h]"] / 4.0
    df_feat["power_kw"] = df_feat["Power [kW]"] / 4.0
    
    df_feat["supply_temp"] = df_feat["supply_temp"].fillna(method="ffill").fillna(method="bfill")
    df_feat["flow_rate"] = df_feat["flow_rate"].fillna(method="ffill").fillna(method="bfill")
    df_feat["power_kw"] = df_feat["power_kw"].fillna(method="ffill").fillna(method="bfill")
    
    return df_feat


def predict_rt514_with_cnn_lstm(model: torch.nn.Module, df: pd.DataFrame, oe403_df: pd.DataFrame,
                                device: torch.device, feature_mean: np.ndarray = None, 
                                feature_std: np.ndarray = None) -> np.ndarray:
    """Use CNN-LSTM model to predict RT514 values from 3 physics-based features.
    
    Note: ComprehensiveDataset only normalizes features, not targets.
    Model predictions are already in actual temperature scale.
    """
    # Prepare features (3 features: supply_temp, flow_rate, power_kw)
    feature_cols = ["supply_temp", "flow_rate", "power_kw"]
    df_feat = reconstruct_features_from_sensors(df, oe403_df)
    features = df_feat[feature_cols].values.astype(np.float32)
    
    # Use provided normalization parameters if available, otherwise calculate from data
    if feature_mean is not None and feature_std is not None:
        logging.info("Using feature normalization parameters from trained model")
    else:
        logging.warning("Feature normalization parameters not provided - calculating from pre-degradation data")
        # Calculate normalization from available data (pre-fault period)
        prefault_mask = df["Timestamp"] < DEGRADATION_TIME
        valid_data = ~np.isnan(features).any(axis=1) & prefault_mask
        
        if valid_data.sum() > 100:
            feature_mean = features[valid_data].mean(axis=0)
            feature_std = features[valid_data].std(axis=0) + 1e-8
            logging.info(f"Calculated feature normalization from {valid_data.sum()} pre-degradation samples")
        else:
            # Fallback to reasonable defaults for 3 features
            feature_mean = np.array([7.0, 0.7, -0.4])
            feature_std = np.array([2.5, 0.35, 1.5]) + 1e-8
            logging.warning("Using default feature normalization parameters")
    
    # Normalize features for model input
    features_norm = (features - feature_mean) / feature_std
    
    # Predict using sequences (use rev10.SEQ_LEN)
    # Model outputs predictions in actual temperature scale (targets not normalized)
    predictions = np.full(len(df), np.nan)
    
    with torch.no_grad():
        for i in range(rev10.SEQ_LEN, len(features_norm)):
            seq = features_norm[i-rev10.SEQ_LEN:i]
            # Skip if sequence contains NaN
            if np.isnan(seq).any():
                continue
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
            pred = model(seq_tensor).cpu().numpy()[0, 0]
            predictions[i] = pred
    
    logging.info(f"Predictions range: [{np.nanmin(predictions):.2f}, {np.nanmax(predictions):.2f}]°C")
    
    return predictions


def create_simple_recovery_plot(df: pd.DataFrame, predictions: np.ndarray, fault_time: pd.Timestamp,
                                output_path: Path, days_before: int = 28, days_after: int = 7) -> Path: 
    # Filter to show requested time window
    start_time = fault_time - pd.Timedelta(days=days_before)
    end_time = fault_time + pd.Timedelta(days=days_after)
    mask = (df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)
    df_plot = df[mask].copy()
    predictions_plot = predictions[mask]
    
    fig, ax = plt.subplots(figsize=(18, 8))

    # Contrasting colors for color printing 
    colors = {
        "737.003-RT512 [°C]": "#1B9E77",   # teal
        "737.003-RT513 [°C]": "#D95F02",   # orange
        "737.003-RT514 [°C]": "#0B67A1",   # dark blue (actual)
        "737.003-RT515 [°C]": "#7570B3",   # purple
    }

    for sensor in ALL_SENSORS:
        ax.plot(df_plot["Timestamp"], df_plot[sensor],
                label=sensor.replace("737.003-", ""),
                color=colors[sensor],
                linewidth=1.6 if sensor == TARGET_SENSOR else 1.3,
                alpha=0.95)
    
    # Plot predicted RT514 as a distinct dashed line (post-fault only)
    postfault_mask = df_plot["Timestamp"] >= fault_time
    valid_mask = postfault_mask.values & (~np.isnan(predictions_plot))

    if valid_mask.any():
        ax.plot(
            df_plot.loc[valid_mask, "Timestamp"],
            predictions_plot[valid_mask],
            label="RT514 predicted (CNN-LSTM)",
            color="#DC143C",  
            linestyle=(0, (3, 3)),
            linewidth=1.5,
            zorder=12
        )
    
    # Mark fault with vertical line and shaded region 
    fault_line_color = "#CC5500"
    ax.axvline(fault_time, color=fault_line_color, linestyle="-", linewidth=2.0, alpha=0.9, label="Malfunction start")
    ax.axvspan(fault_time, end_time, color=fault_line_color, alpha=0.08)

    # Add text annotation for malfunction 
    y_top = ax.get_ylim()[1]
    ax.text(fault_time, y_top - 1.2, f' Malfunction\n {fault_time.strftime("%Y-%m-%d %H:%M")}',
        rotation=90, va='top', ha='right', fontsize=9, color=fault_line_color, fontweight="bold")

    # Mark observed degradation start
    degr = DEGRADATION_TIME
    if (degr >= start_time) and (degr <= end_time):
        degr_color = "#2E8B57"  
        ax.axvline(degr, color=degr_color, linestyle=(0, (1, 4)), linewidth=1.6, alpha=0.9, label="Degradation start")
        ax.text(degr, y_top - 1.2, f' Degradation\n {degr.strftime("%Y-%m-%d %H:%M")}',
            rotation=90, va='top', ha='left', fontsize=9, color=degr_color)
    
    ax.set_title("Double U-45 Temperature Sensor Recovery: CNN-LSTM Model Prediction", fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel("Temperature [°C]", fontsize=13)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylim(-10, 20)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return output_path
def calculate_baseline_predictions(df: pd.DataFrame) -> np.ndarray:
    """Calculate baseline using spatial average of healthy sensors."""
    healthy_sensors = ["737.003-RT512 [°C]", "737.003-RT513 [°C]", "737.003-RT515 [°C]"]
    baseline = df[healthy_sensors].mean(axis=1).values
    return baseline


def create_baseline_comparison_plot(df: pd.DataFrame, predictions: np.ndarray, 
                                   baseline: np.ndarray, fault_time: pd.Timestamp,
                                   output_path: Path, days_before: int = 28, 
                                   days_after: int = 7) -> Path:
    """Create comparison plot between CNN-LSTM and baseline method."""
    start_time = fault_time - pd.Timedelta(days=days_before)
    end_time = fault_time + pd.Timedelta(days=days_after)
    mask = (df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)
    df_plot = df[mask].copy()
    predictions_plot = predictions[mask]
    baseline_plot = baseline[mask]
    
    # Evaluate during pre-degradation period where RT514 is accurate
    # Use timestamp-based filtering since df_plot maintains original indices
    eval_mask = (df_plot["Timestamp"] >= df["Timestamp"].iloc[rev10.SEQ_LEN]) & (df_plot["Timestamp"] < DEGRADATION_TIME)
    valid_pred = eval_mask.values & (~np.isnan(predictions_plot))
    valid_base = eval_mask.values & (~np.isnan(baseline_plot))
    
    if valid_pred.sum() > 0:
        actual = df_plot.loc[valid_pred, TARGET_SENSOR].values
        pred = predictions_plot[valid_pred]
        mae_model = np.mean(np.abs(actual - pred))
        rmse_model = np.sqrt(np.mean((actual - pred) ** 2))
    else:
        mae_model = rmse_model = np.nan
    
    if valid_base.sum() > 0:
        actual = df_plot.loc[valid_base, TARGET_SENSOR].values
        base = baseline_plot[valid_base]
        mae_baseline = np.mean(np.abs(actual - base))
        rmse_baseline = np.sqrt(np.mean((actual - base) ** 2))
    else:
        mae_baseline = rmse_baseline = np.nan
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    ax.plot(df_plot["Timestamp"], df_plot[TARGET_SENSOR],
            label="RT514 actual", color="#0B67A1", linewidth=1.6, alpha=0.95)
    
    valid_mask_pred = ~np.isnan(predictions_plot)
    if valid_mask_pred.any():
        ax.plot(df_plot.loc[valid_mask_pred, "Timestamp"], predictions_plot[valid_mask_pred],
                label="CNN-LSTM prediction", color="#DC143C", linestyle="--", linewidth=1.5)
    
    valid_mask_base = ~np.isnan(baseline_plot)
    if valid_mask_base.any():
        ax.plot(df_plot.loc[valid_mask_base, "Timestamp"], baseline_plot[valid_mask_base],
                label="Baseline (spatial average)", color="#7570B3", linestyle=":", linewidth=1.5)
    
    fault_line_color = "#CC5500"
    ax.axvline(fault_time, color=fault_line_color, linestyle="-", linewidth=2.0, alpha=0.9)
    
    # Mark degradation start for clarity
    if DEGRADATION_TIME >= start_time and DEGRADATION_TIME <= end_time:
        degr_color = "#2E8B57"
        ax.axvline(DEGRADATION_TIME, color=degr_color, linestyle="--", linewidth=1.5, alpha=0.7, label="Degradation start")
    
    stats_text = (
        f"Pre-degradation Performance (RT514 healthy):\n"
        f"CNN-LSTM: MAE={mae_model:.4f}°C, RMSE={rmse_model:.4f}°C\n"
        f"Baseline: MAE={mae_baseline:.4f}°C, RMSE={rmse_baseline:.4f}°C"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_title("RT514 Reconstruction: CNN-LSTM vs Baseline Method", fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel("Temperature [°C]", fontsize=13)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylim(-10, 20)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return output_path


def train_sensor_recovery_model(df: pd.DataFrame, oe403_df: pd.DataFrame, 
                                 model_path: Path, device: torch.device):
    """Train a new CNN-LSTM model for sensor recovery using rev10 infrastructure.
    
    Uses 3 features (physics-based): supply_temp, flow_rate, power_kw
    
    Uses rev10's:
    - ComprehensiveDataset for data preparation
    - train_model() function for training loop
    - Hyperparameters (EPOCHS, LR, PATIENCE, BATCH_SIZE)
    """
    logging.info("=" * 70)
    logging.info("TRAINING NEW SENSOR RECOVERY MODEL (3 features)")
    logging.info("Physics-based model: operational parameters only")
    logging.info("Using rev10 training infrastructure for consistency")
    logging.info("=" * 70)
    
    # Prepare features from pre-degradation period only
    prefault_df = df[df["Timestamp"] < DEGRADATION_TIME].copy()
    df_feat = reconstruct_features_from_sensors(prefault_df, oe403_df)
    
    # Use 3 features: supply_temp, flow_rate, power_kw
    feature_cols = ["supply_temp", "flow_rate", "power_kw"]
    
    target_col = TARGET_SENSOR
    
    # Create combined dataframe with required columns
    train_df = pd.DataFrame({
        'Timestamp': prefault_df['Timestamp'],
        'supply_temp': df_feat['supply_temp'],
        'flow_rate': df_feat['flow_rate'],
        'power_kw': df_feat['power_kw'],
        'return_temp': prefault_df[target_col]  # Target
    }).dropna()
    
    logging.info(f"Pre-degradation data: {len(train_df)} valid samples")
    
    # CRITICAL: Use time-based split FROM THE END like main model
    # This avoids seasonal drift issues
    latest_timestamp = train_df['Timestamp'].max()
    test_days = 50  # ~20% of 250 days
    val_days = 25   # ~10% of 250 days
    
    test_start = latest_timestamp - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)
    
    train_data = train_df[train_df['Timestamp'] < val_start].copy()
    val_data = train_df[(train_df['Timestamp'] >= val_start) & (train_df['Timestamp'] < test_start)].copy()
    test_data = train_df[train_df['Timestamp'] >= test_start].copy()
    
    logging.info(f"Time-based split: Train={len(train_data)} (up to {val_start}), Val={len(val_data)} ({val_start} to {test_start}), Test={len(test_data)} (from {test_start})")
    
    # Create datasets using rev10's ComprehensiveDataset
    train_dataset = rev10.ComprehensiveDataset(
        df=train_data,
        seq_len=rev10.SEQ_LEN,
        horizon=rev10.PRED_HORIZON,
        feature_cols=feature_cols,
        target_col='return_temp',
        mean=None,  # Will calculate from data
        std=None
    )
    
    val_dataset = rev10.ComprehensiveDataset(
        df=val_data,
        seq_len=rev10.SEQ_LEN,
        horizon=rev10.PRED_HORIZON,
        feature_cols=feature_cols,
        target_col='return_temp',
        mean=train_dataset.mean,  # Use training set normalization
        std=train_dataset.std
    )
    
    test_dataset = rev10.ComprehensiveDataset(
        df=test_data,
        seq_len=rev10.SEQ_LEN,
        horizon=rev10.PRED_HORIZON,
        feature_cols=feature_cols,
        target_col='return_temp',
        mean=train_dataset.mean,  # Use training set normalization
        std=train_dataset.std
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=rev10.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=rev10.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=rev10.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model with 3 input features
    model = rev10.ComprehensiveCNNLSTM(
        input_features=3,
        conv_channels=rev10.CONV_CHANNELS,
        kernel_size=rev10.KERNEL_SIZE,
        lstm_hidden=rev10.LSTM_HIDDEN,
        lstm_layers=rev10.LSTM_LAYERS,
        dropout=rev10.DROPOUT
    ).to(device)
    
    logging.info("Model initialized with 3 input features (supply_temp, flow_rate, power_kw)")
    logging.info(f"Training parameters: EPOCHS={rev10.EPOCHS}, LR={rev10.LR}, PATIENCE={rev10.PATIENCE}, BATCH_SIZE={rev10.BATCH_SIZE}")
    
    # Train using rev10's train_model function
    model, training_history = rev10.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=rev10.EPOCHS,
        lr=rev10.LR,
        device=device,
        patience=rev10.PATIENCE
    )
    
    # Extract loss histories
    train_losses = training_history['train_losses']
    val_losses = training_history['val_losses']
    
    # Evaluate on held-out test set
    logging.info("=" * 70)
    logging.info("Evaluating on held-out test set...")
    model.eval()
    test_preds = []
    test_actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            test_preds.extend(outputs.cpu().numpy())
            test_actuals.extend(batch_y.numpy())
    
    test_preds = np.array(test_preds).flatten()
    test_actuals = np.array(test_actuals).flatten()
    
    test_mae = np.mean(np.abs(test_preds - test_actuals))
    test_rmse = np.sqrt(np.mean((test_preds - test_actuals) ** 2))
    
    logging.info(f"Test Set Performance (held-out {len(test_data)} samples, {test_days} final days):")
    logging.info(f"  MAE:  {test_mae:.4f}°C")
    logging.info(f"  RMSE: {test_rmse:.4f}°C")
    logging.info("=" * 70)
    
    # Save model and feature normalization parameters
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'feature_mean': train_dataset.mean,
        'feature_std': train_dataset.std,
        'feature_cols': feature_cols,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_start': test_start.isoformat()  # Save test set boundary
    }
    torch.save(checkpoint, model_path)
    
    logging.info("=" * 70)
    logging.info(f"Model and normalization parameters saved to: {model_path}")
    logging.info("=" * 70)
    
    # Save results to JSON
    results = {
        "model_performance": {
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "test_samples": int(len(test_data)),
            "test_days": test_days,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        },
        "training_history": {
            "train_losses": [float(x) for x in train_losses],
            "val_losses": [float(x) for x in val_losses],
            "epochs_trained": len(train_losses),
            "best_val_loss": float(min(val_losses))
        },
        "data_split": {
            "train_samples": int(len(train_data)),
            "val_samples": int(len(val_data)),
            "test_samples": int(len(test_data)),
            "val_start": val_start.isoformat(),
            "test_start": test_start.isoformat(),
            "latest_timestamp": latest_timestamp.isoformat()
        },
        "model_config": {
            "input_features": 3,
            "feature_cols": feature_cols,
            "seq_len": rev10.SEQ_LEN,
            "pred_horizon": rev10.PRED_HORIZON,
            "batch_size": rev10.BATCH_SIZE,
            "learning_rate": rev10.LR,
            "patience": rev10.PATIENCE,
            "conv_channels": rev10.CONV_CHANNELS,
            "kernel_size": rev10.KERNEL_SIZE,
            "lstm_hidden": rev10.LSTM_HIDDEN,
            "lstm_layers": rev10.LSTM_LAYERS,
            "dropout": rev10.DROPOUT
        }
    }
    
    json_path = model_path.parent / "rt514_recovery_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to: {json_path}")
    
    # Return model, normalization params, and test_start for evaluation
    return model, train_dataset.mean, train_dataset.std, test_start


def main():
    args = parse_args()
    
    # Setup logging (consistent with rev10)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info("=" * 70)
    logging.info("RT514 SENSOR RECOVERY - CNN-LSTM MODEL (3-feature physics-based)")
    logging.info("Using rev10 infrastructure for consistency")
    logging.info("=" * 70)
    
    # Load data
    df = load_sensor_data(args.data_path)
    logging.info(f"Loaded {len(df)} sensor readings")
    
    # Load OE403 energy meter data using rev10's function
    oe403_df = load_oe403_data(args.data_path)
    logging.info(f"Loaded {len(oe403_df)} OE403 meter readings")
    
    # Detect fault
    fault_time = detect_fault_time(df)
    
    # Train or load model
    if args.train:
        model, feature_mean, feature_std, test_start = train_sensor_recovery_model(df, oe403_df, args.model_path, device)
        logging.info("Training completed successfully")
    else:
        model, feature_mean, feature_std, test_start = load_and_prepare_cnn_lstm_model(args.model_path, device)
    
    # Predict RT514
    logging.info("Predicting RT514 with 3-feature CNN-LSTM model (physics-based)...")
    logging.info("Features: supply_temp, flow_rate, power_kw")
    predictions = predict_rt514_with_cnn_lstm(model, df, oe403_df, device, feature_mean, feature_std)
    
    # Calculate baseline predictions
    logging.info("Calculating baseline predictions (spatial average of RT512, RT513, RT515)...")
    baseline = calculate_baseline_predictions(df)
    
    # Evaluate predictions on test set only (if available) or pre-degradation period
    if test_start is None:
        # Recalculate test_start using same logic as training
        logging.warning("Test set boundary not found in checkpoint - recalculating from data")
        prefault_timestamps = df[df["Timestamp"] < DEGRADATION_TIME]["Timestamp"]
        latest_timestamp = prefault_timestamps.max()
        test_days = 50  # Same as training
        test_start = latest_timestamp - pd.Timedelta(days=test_days)
    
    # Evaluate only on held-out test set
    eval_mask = (df["Timestamp"] >= test_start) & (df["Timestamp"] < DEGRADATION_TIME) & (~np.isnan(predictions))
    eval_label = "TEST SET"
    
    if eval_mask.sum() > 0:
        actual = df.loc[eval_mask, TARGET_SENSOR].values
        pred = predictions[eval_mask]
        
        # Filter out any remaining NaN values in actual RT514
        valid = ~np.isnan(actual)
        if valid.sum() < len(actual):
            logging.info(f"Filtering out {(~valid).sum()} samples with NaN actual RT514 values")
            actual = actual[valid]
            pred = pred[valid]
        
        mae_model = np.mean(np.abs(actual - pred))
        rmse_model = np.sqrt(np.mean((actual - pred) ** 2))
        
        # Also evaluate baseline during same period
        base = baseline[eval_mask][valid]
        mae_baseline = np.mean(np.abs(actual - base))
        rmse_baseline = np.sqrt(np.mean((actual - base) ** 2))
        
        total_degradation = (df["Timestamp"] < DEGRADATION_TIME).sum()
        coverage = (eval_mask.sum() / total_degradation * 100)
        
        logging.info("=" * 70)
        logging.info(f"EVALUATION RESULTS ({eval_label})")
        logging.info("=" * 70)
        logging.info(f"Evaluation period: {df.loc[eval_mask, 'Timestamp'].min()} to {df.loc[eval_mask, 'Timestamp'].max()}")
        logging.info(f"Valid samples: {eval_mask.sum():,} ({coverage:.1f}% of pre-degradation data)")
        logging.info("")
        logging.info(f"CNN-LSTM Model:")
        logging.info(f"  MAE:  {mae_model:.4f}°C")
        logging.info(f"  RMSE: {rmse_model:.4f}°C")
        logging.info("")
        logging.info(f"Baseline (Spatial Average):")
        logging.info(f"  MAE:  {mae_baseline:.4f}°C")
        logging.info(f"  RMSE: {rmse_baseline:.4f}°C")
        logging.info("")
        logging.info(f"Improvement over baseline: {(1 - mae_model/mae_baseline)*100:+.1f}%")
        logging.info("=" * 70)
    else:
        logging.warning("No valid predictions in evaluation period - check data alignment")
    
    # Create main recovery plot
    plot_path = create_simple_recovery_plot(
        df, predictions, fault_time, args.output_path,
        days_before=args.days_before_fault,
        days_after=args.days_after_fault
    )
    
    # Create baseline comparison plot
    comparison_path = args.output_path.parent / "rt514_baseline_comparison.png"
    comparison_plot = create_baseline_comparison_plot(
        df, predictions, baseline, fault_time, comparison_path,
        days_before=args.days_before_fault,
        days_after=args.days_after_fault
    )
    
    # Save evaluation results to JSON if not training
    if not args.train:
        results_path = args.model_path.parent / "rt514_recovery_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Add evaluation metrics
            if eval_mask.sum() > 0:
                results["evaluation"] = {
                    "pre_degradation_mae": float(mae_model),
                    "pre_degradation_rmse": float(rmse_model),
                    "baseline_mae": float(mae_baseline),
                    "baseline_rmse": float(rmse_baseline),
                    "improvement_over_baseline_pct": float((1 - mae_model/mae_baseline)*100),
                    "eval_samples": int(eval_mask.sum()),
                    "eval_coverage_pct": float(coverage),
                    "eval_start": test_start.isoformat(),
                    "eval_end": DEGRADATION_TIME.isoformat()
                }
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logging.info(f"Evaluation results saved to: {results_path}")
    
    logging.info("=" * 70)
    logging.info(f"Recovery plot saved: {plot_path}")
    logging.info(f"Baseline comparison saved: {comparison_plot}")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
