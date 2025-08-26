
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# ---------------------------
# Config
# ---------------------------
CSV_PATH = os.environ.get("CSV_PATH", "/mnt/data/EDE_with_geothermal_features.csv")
TARGET_HINTS = ["retur temperatur", "returtemperatur", "outlet", "utløpstemperatur"]
INLET_HINTS = ["tur temperatur", "innløp", "frem", "inlet", "supply"]
OUTDOOR_HINTS = ["utetemperatur", "outdoor"]
EFFECT_HINTS = ["avgitt effekt", "kw"]
FLOW_HINTS = ["flow", "gjennomstrøm"]
PRESSURE_HINTS = ["trykk"]
DEPTH_HINTS = ["nivå", "vannstand", "brønn", "grunn", "brønner", "grunnvann"]

PLOT_PATH = os.environ.get("PLOT_PATH", "/mnt/data/forecast_with_geothermal.png")

# ---------------------------
# Helpers
# ---------------------------
def find_time_column(df):
    for c in df.columns:
        s = c.lower()
        if s.startswith("tid") or "time" in s or "timestamp" in s:
            return c
    return df.columns[0]

def find_first(cols, any_keywords, must_have=None):
    must_have = must_have or []
    for c in cols:
        s = c.lower()
        if any(k in s for k in any_keywords) and all(k in s for k in must_have):
            return c
    return None

def add_lags(df, cols, lags=(1,2,3,6,12)):
    for c in cols:
        for L in lags:
            df[f"{c}__lag{L}"] = df[c].shift(L)
    return df

def add_rolls(df, cols, windows=(3,6,12)):
    for c in cols:
        for w in windows:
            df[f"{c}__roll{w}"] = df[c].rolling(w, min_periods=1).mean()
    return df

def is_groundwater_col(name: str):
    s = name.lower()
    return ("nivå" in s or "vannstand" in s) and any(k in s for k in ["grunn", "brønn", "brønner", "brønnpark", "grunnvann"])

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(CSV_PATH)
time_col = find_time_column(df)
df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.sort_values(time_col).reset_index(drop=True)

cols = [c for c in df.columns if c != time_col]

# Target and key predictors
target_col = find_first(cols, TARGET_HINTS) or find_first([c for c in cols if "temperatur" in c.lower()], [""])
inlet_col  = find_first(cols, INLET_HINTS)
outdoor_col = find_first(cols, OUTDOOR_HINTS)

# Other predictors
effect_cols   = [c for c in cols if any(k in c.lower() for k in EFFECT_HINTS)][:5]
flow_cols     = [c for c in cols if any(k in c.lower() for k in FLOW_HINTS)][:2]
pressure_cols = [c for c in cols if any(k in c.lower() for k in PRESSURE_HINTS)][:2]
temp_aux_cols = [c for c in cols if ("temperatur" in c.lower()) and c not in {target_col, inlet_col, outdoor_col}][:6]

# Geothermal features added earlier
geo_cols = [c for c in df.columns if c in {"geo_gradient_C_per_km","geo_heatflow_mW_m2","bore_depth_km","geo_baseline_T_at_depth"}]

# Depth/groundwater level columns (if present)
depth_cols = [c for c in cols if is_groundwater_col(c)][:2]

# Build feature frame
feature_cols = []
if inlet_col: feature_cols.append(inlet_col)
if outdoor_col: feature_cols.append(outdoor_col)
feature_cols += effect_cols + flow_cols + pressure_cols + temp_aux_cols + geo_cols + depth_cols

if target_col is None:
    raise RuntimeError("Could not identify a target (outlet/return temperature) column.")

df_feat = df[[time_col, target_col] + feature_cols].copy()

# Depth change (first difference) if any
if len(depth_cols) > 0:
    for dc in depth_cols:
        df_feat[f"{dc}__d1"] = df_feat[dc].diff()

# Delta T between inlet and outlet
if inlet_col:
    df_feat["delta_T_in_out"] = df_feat[inlet_col] - df_feat[target_col]

# Add lags & rolling means
df_feat = add_lags(df_feat, [target_col] + [c for c in feature_cols if c is not None])
df_feat = add_rolls(df_feat, [target_col])

# Drop initial NaNs from lag/roll creation
df_feat = df_feat.dropna().reset_index(drop=True)

# Split train/test by time (last 20% -> test)
split_idx = int(len(df_feat) * 0.8)
train_df = df_feat.iloc[:split_idx].copy()
test_df  = df_feat.iloc[split_idx:].copy()

X_train = train_df.drop(columns=[time_col, target_col])
y_train = train_df[target_col]
X_test  = test_df.drop(columns=[time_col, target_col])
y_test  = test_df[target_col]

# Models
models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "GradBoost": GradientBoostingRegressor(random_state=42)
}

preds = {}
maes = {}
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    yhat = mdl.predict(X_test)
    preds[name] = yhat
    maes[name] = mean_absolute_error(y_test, yhat)

# Report MAE
print("Mean Absolute Error per model:")
for name, v in maes.items():
    print(f"  {name}: {v:.3f}")

# Plot
plt.figure(figsize=(12,4))
plt.plot(train_df[time_col], train_df[target_col], label="Previous training values", linewidth=2)
plt.plot(test_df[time_col], y_test.values, label="Test actual values")
for name, yhat in preds.items():
    plt.plot(test_df[time_col], yhat, label=f"{name} forecasted values")
plt.axvline(test_df[time_col].iloc[0], linestyle="--")
plt.xlabel("Timeline")
plt.ylabel("Outlet temperature (°C)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=200)
print(f"Saved plot to: {PLOT_PATH}")
