"""RT514 sensor recovery using trained CNN-LSTM model."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import Traindata_geothermal_HybridCNNLSTM_rev10_Fixed as rev10


TARGET_SENSOR = "737.003-RT514 [°C]"
ALL_SENSORS = [
    "737.003-RT512 [°C]",
    "737.003-RT513 [°C]",
    "737.003-RT514 [°C]",
    "737.003-RT515 [°C]",
]
FAULT_THRESHOLD = -10.0
SEQ_LEN = 48
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
        default=Path(rev10.OUTPUT_DIR) / "comprehensive_model.pth",
        help="Path to trained comprehensive model",
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
    print(f"Fault detected: {fault_time}")
    print(f"RT514 value: {df.loc[fault_mask, TARGET_SENSOR].iloc[0]:.1f}°C")
    return fault_time


def load_and_prepare_cnn_lstm_model(model_path: Path, device: torch.device):
    """Load trained CNN-LSTM model with Double U-45 configuration."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Model expects: [supply_temp, flow_rate, power_kw, bhe_type_encoded]
    model = rev10.ComprehensiveCNNLSTM(
        input_features=4,
        conv_channels=rev10.CONV_CHANNELS,
        kernel_size=rev10.KERNEL_SIZE,
        lstm_hidden=rev10.LSTM_HIDDEN,
        lstm_layers=rev10.LSTM_LAYERS,
        dropout=rev10.DROPOUT
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded CNN-LSTM model from {model_path}")
    return model


def load_oe403_data(csv_path: Path) -> pd.DataFrame:
    """Load OE403 energy meter data."""
    oe403_df = rev10.load_double_u45mm_research_data()
    
    if oe403_df.empty:
        raise FileNotFoundError(f"OE403 data not found or failed to load")
    
    oe403_df = oe403_df.rename(columns={
        "flow_rate": "Flow [m³/h]",
        "power_kw": "Power [kW]"
    })
    
    print(f"Loaded OE403 data: {len(oe403_df)} records")
    return oe403_df


def reconstruct_features_from_sensors(df: pd.DataFrame, oe403_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct input features from sensor data."""
    df_feat = df.copy()
    
    df_feat = df_feat.merge(
        oe403_df[["Timestamp", "supply_temp", "Flow [m³/h]", "Power [kW]"]],
        on="Timestamp",
        how="left"
    )
    
    df_feat["flow_rate"] = df_feat["Flow [m³/h]"]
    df_feat["power_kw"] = df_feat["Power [kW]"]
    
    df_feat["supply_temp"] = df_feat["supply_temp"].fillna(method="ffill").fillna(method="bfill")
    df_feat["flow_rate"] = df_feat["flow_rate"].fillna(method="ffill").fillna(method="bfill")
    df_feat["power_kw"] = df_feat["power_kw"].fillna(method="ffill").fillna(method="bfill")
    
    df_feat["bhe_type_encoded"] = 1
    
    return df_feat


def predict_rt514_with_cnn_lstm(model: torch.nn.Module, df: pd.DataFrame, oe403_df: pd.DataFrame, device: torch.device) -> np.ndarray:
    """Use CNN-LSTM model to predict RT514 values."""
    # Prepare features
    df_feat = reconstruct_features_from_sensors(df, oe403_df)
    features = df_feat[["supply_temp", "flow_rate", "power_kw", "bhe_type_encoded"]].values.astype(np.float32)
    
    # Use same normalization as rev10 training (from comprehensive_results.json)
    feature_mean = np.array([7.566, 2.818, -1.508, 1.00])  # From training data statistics
    feature_std = np.array([2.688, 1.380, 6.136, 0.816]) + 1e-8
    features_norm = (features - feature_mean) / feature_std
    
    # Predict using sequences
    predictions = np.full(len(df), np.nan)
    
    with torch.no_grad():
        for i in range(SEQ_LEN, len(features_norm)):
            seq = features_norm[i-SEQ_LEN:i]
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
            pred = model(seq_tensor).cpu().numpy()[0, 0]
            predictions[i] = pred
    
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
    
    prefault_mask = (df_plot["Timestamp"] < fault_time).values
    valid_pred = prefault_mask & (~np.isnan(predictions_plot))
    valid_base = prefault_mask & (~np.isnan(baseline_plot))
    
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
    
    stats_text = (
        f"Pre-fault Performance:\n"
        f"CNN-LSTM: MAE={mae_model:.3f}°C, RMSE={rmse_model:.3f}°C\n"
        f"Baseline: MAE={mae_baseline:.3f}°C, RMSE={rmse_baseline:.3f}°C"
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

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("RT514 SENSOR RECOVERY - CNN-LSTM MODEL")
    print("=" * 70)
    
    # Load data
    df = load_sensor_data(args.data_path)
    print(f"Loaded {len(df)} readings")
    
    # Load OE403 energy meter data
    oe403_df = load_oe403_data(args.data_path)
    print(f"Loaded {len(oe403_df)} OE403 meter readings")
    
    # Detect fault
    fault_time = detect_fault_time(df)
    
    # Load CNN-LSTM model
    model = load_and_prepare_cnn_lstm_model(args.model_path, device)
    
    # Predict RT514
    print("Predicting RT514 with CNN-LSTM model using OE403 data and training normalization...")
    predictions = predict_rt514_with_cnn_lstm(model, df, oe403_df, device)
    
    # Calculate baseline predictions
    print("Calculating baseline predictions...")
    baseline = calculate_baseline_predictions(df)
    
    # Evaluate on entire pre-fault period
    prefault_mask = (df["Timestamp"] < fault_time) & (~np.isnan(predictions))
    if prefault_mask.sum() > 0:
        actual = df.loc[prefault_mask, TARGET_SENSOR].values
        pred = predictions[prefault_mask]
        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        
        total_prefault = (df["Timestamp"] < fault_time).sum()
        coverage = (prefault_mask.sum() / total_prefault * 100)
        
        print(f"Pre-fault MAE: {mae:.3f}°C, RMSE: {rmse:.3f}°C")
        print(f"Valid predictions: {prefault_mask.sum()} / {total_prefault} records ({coverage:.1f}% coverage)")
    else:
        print("No valid predictions in evaluation period")
    
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
    
    print("=" * 70)
    print(f"Recovery plot saved: {plot_path}")
    print(f"Baseline comparison saved: {comparison_plot}")
    print("=" * 70)


if __name__ == "__main__":
    main()
