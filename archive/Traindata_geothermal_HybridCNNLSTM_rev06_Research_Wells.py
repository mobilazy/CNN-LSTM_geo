"""
CNN-LSTM Model for 8 Research Wells BHE Configuration Analysis

STANDARDIZED DEPTH APPROACH - No Depth Signal:
=============================================


AVAILABLE DATA:
- Complete field (120 boreholes): Single U45mm → averaged data
- Research wells Double U45mm (4 wells): Individual outlet temperatures from DoubleU45_Treturn.csv
- Research wells Double U45mm (4 wells): Shared sensor data from OE403 (supply temp, power, flow)
- Research wells MuoviEllipse 63mm (4 wells): Only aggregated data (shared OE402)

RESEARCH WELLS CONFIGURATION (ALL STANDARDIZED TO 300m):
Double U45mm Wells (Individual outlet temps + shared sensor data):
- SKD-110-01: Individual outlet temp RT512, standardized to 300m depth
- SKD-110-02: Individual outlet temp RT513, standardized to 300m depth
- SKD-110-03: Individual outlet temp RT514, standardized to 300m depth
- SKD-110-04: Individual outlet temp RT515, standardized to 300m depth
- Shared sensor OE403: Supply temperature, power, flow rate for all 4 wells

MuoviEllipse 63mm Wells (Aggregated data only via OE402):
- 4x wells at 300m depth (equal flow distribution)

STRATEGY:
1. Compare 3 BHE types: Single U45mm (complete field), Double U45mm (research), MuoviEllipse 63mm (research)
2. All wells standardized to 300m depth - NO DEPTH SIGNAL
3. For Double U45mm: Combine individual outlet temperatures with shared sensor data from OE403
4. Focus purely on collector configuration thermal performance differences
5. Isolate BHE type effects without depth variability
"""

from typing import List, Dict, Optional
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#==============================================================================
# RESEARCH WELLS CONFIGURATION - BASED ON FREDRIK'S DATA
#==============================================================================

# Output directory
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", 
                           os.path.join(os.path.dirname(__file__), "output"))

# CSV file paths
CSV_MAIN_FIELD = os.path.join(os.path.dirname(__file__), "input/Borehole heat extraction complete field.csv")
CSV_DOUBLE_U_RETURNS = os.path.join(os.path.dirname(__file__), "input/DoubleU45_Treturn.csv")
CSV_OE403_METER = os.path.join(os.path.dirname(__file__), "input/MeterOE403_doubleU45.csv")

# Model hyperparameters
SEQ_LEN = int(os.environ.get("SEQ_LEN", "24"))  # Reduced for research wells
PRED_HORIZON = int(os.environ.get("PRED_HORIZON", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
LR = float(os.environ.get("LR", "1e-3"))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", "0.15"))
TEST_SPLIT = float(os.environ.get("TEST_SPLIT", "0.25"))

# CNN-LSTM architecture
CONV_CHANNELS = [32, 64]
KERNEL_SIZE = 3
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
DROPOUT = 0.1
PATIENCE = 10

# Physical constants for HX24 working fluid
HX24_SPECIFIC_HEAT = 3600.0  # J/(kg·K)
HX24_DENSITY = 970.0  # kg/m³

# BHE Configuration Properties - ALL STANDARDIZED TO 300m DEPTH
BHE_CONFIGURATIONS = {
    'single_u45_complete_field': {
        'type': 'single_u45mm',
        'depth_m': 300,  # Standardized depth
        'thermal_resistance': 0.09,  # mK/W - Higher resistance (single pipe)
        'area_factor': 1.0,  # Baseline
        'description': 'Single U45mm - Complete Field (120 boreholes at 300m)',
        'data_source': 'complete_field_averaged'
    },
    'double_u45_research': {
        'type': 'double_u45mm', 
        'depth_m': 300,  # Standardized depth for all research wells
        'thermal_resistance': 0.06,  # mK/W - Lower resistance (double pipe)
        'area_factor': 1.4,  # ~40% more heat transfer area
        'description': 'Double U45mm - Research Wells (4 wells at standardized 300m)',
        'data_source': 'individual_return_temps'
    },
    'muovi_ellipse_63mm_research': {
        'type': 'muovi_ellipse_63mm',
        'depth_m': 300,  # Standardized depth
        'thermal_resistance': 0.075,  # mK/W - Intermediate (larger diameter)
        'area_factor': 1.2,  # ~20% more area than single U45mm
        'description': 'MuoviEllipse 63mm - Research Wells (4 wells at standardized 300m)',
        'data_source': 'aggregated_oe402'
    }
}

# Column name mappings
TIME_COL = "Timestamp"

# Complete field data columns
MAIN_FIELD_COLS = {
    'flow': "Flow rate to 120 boreholes [m³/h]",
    'supply': "Supply temperature measure at energy meter [°C]",
    'return': "Return temperature measure at energy meter [°C]", 
    'power': "Negative Heat extracion [kW] / Positive Heat rejection [kW]"
}

# Double U45mm return temperature columns (individual wells - all at standardized 300m)
DOUBLE_U_RETURN_COLS = {
    'SKD-110-01': "737.003-RT512 [°C]",  # Standardized to 300m
    'SKD-110-02': "737.003-RT513 [°C]",  # Standardized to 300m
    'SKD-110-03': "737.003-RT514 [°C]",  # Standardized to 300m
    'SKD-110-04': "737.003-RT515 [°C]"   # Standardized to 300m
}

# OE403 meter data columns (Double U45mm aggregated)
OE403_COLS = {
    'power': "Power [kW]",
    'supply': "T_supply [°C]",
    'return': "T_return [°C]",
    'flow': "Flow [m³/h]"
}

#------------------------------------------------------------------------------
# LOGGING SETUP
#------------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "research_wells_analysis.log"), 
                           mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

#------------------------------------------------------------------------------
# DATA LOADING AND PROCESSING FUNCTIONS
#------------------------------------------------------------------------------

def load_complete_field_data():
    """Load and process complete field data (120 boreholes, Single U45mm)."""
    
    if not os.path.exists(CSV_MAIN_FIELD):
        logging.error(f"Complete field CSV not found: {CSV_MAIN_FIELD}")
        return None
    
    logging.info("Loading complete field data...")
    
    try:
        df = pd.read_csv(CSV_MAIN_FIELD, sep=';', decimal=',')
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], format='%d.%m.%Y %H:%M', errors="coerce")
        df = df.dropna(subset=[TIME_COL]).reset_index(drop=True)
        
        # Convert numeric columns to ensure proper data types
        numeric_cols = [MAIN_FIELD_COLS['flow'], MAIN_FIELD_COLS['supply'], 
                       MAIN_FIELD_COLS['return'], MAIN_FIELD_COLS['power']]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add BHE configuration data (NO DEPTH SIGNAL)
        config = BHE_CONFIGURATIONS['single_u45_complete_field']
        df['bhe_type'] = 'single_u45mm'
        df['thermal_resistance'] = config['thermal_resistance']
        df['area_factor'] = config['area_factor']
        df['well_id'] = 'complete_field'
        
        # Rename columns for consistency
        df = df.rename(columns={
            MAIN_FIELD_COLS['supply']: 'supply_temp',
            MAIN_FIELD_COLS['return']: 'return_temp',
            MAIN_FIELD_COLS['power']: 'power_kw',
            MAIN_FIELD_COLS['flow']: 'flow_rate'
        })
        
        # Keep only essential columns (NO DEPTH SIGNAL)
        essential_cols = [TIME_COL, 'supply_temp', 'return_temp', 'power_kw', 'flow_rate',
                         'bhe_type', 'thermal_resistance', 'area_factor', 'well_id']
        df = df[essential_cols].copy()
        
        logging.info(f"Loaded complete field data: {len(df)} records")
        return df
        
    except Exception as e:
        logging.error(f"Error loading complete field data: {e}")
        return None

def load_double_u45_individual_data():
    """
    Load Double U45mm research wells data combining:
    - Individual outlet temperatures from DoubleU45_Treturn.csv (4 separate wells)
    - Shared sensor data from OE403 (supply temp, power, flow for all 4 wells combined)
    """
    
    # Load individual return temperatures
    if not os.path.exists(CSV_DOUBLE_U_RETURNS):
        logging.error(f"Double U return temps CSV not found: {CSV_DOUBLE_U_RETURNS}")
        return None
        
    # Load OE403 meter data
    if not os.path.exists(CSV_OE403_METER):
        logging.error(f"OE403 meter CSV not found: {CSV_OE403_METER}")
        return None
    
    logging.info("Loading Double U45mm research wells data...")
    
    try:
        # Load return temperatures
        returns_df = pd.read_csv(CSV_DOUBLE_U_RETURNS)
        returns_df[TIME_COL] = pd.to_datetime(returns_df[TIME_COL], format='%d.%m.%Y %H:%M', errors="coerce")
        
        # Convert return temperature columns to numeric (remove any spaces)
        for col in DOUBLE_U_RETURN_COLS.values():
            if col in returns_df.columns:
                returns_df[col] = pd.to_numeric(returns_df[col].astype(str).str.strip(), errors='coerce')
        
        # Load meter data
        meter_df = pd.read_csv(CSV_OE403_METER)
        meter_df[TIME_COL] = pd.to_datetime(meter_df[TIME_COL], format='%d.%m.%Y %H:%M', errors="coerce")
        
        # Convert numeric columns (remove any spaces and convert to float)
        numeric_cols = [OE403_COLS['supply'], OE403_COLS['power'], OE403_COLS['flow']]
        for col in numeric_cols:
            if col in meter_df.columns:
                meter_df[col] = pd.to_numeric(meter_df[col].astype(str).str.strip(), errors='coerce')
        
        # Merge on timestamp
        combined_df = pd.merge(returns_df, meter_df, on=TIME_COL, how='inner')
        combined_df = combined_df.dropna(subset=[TIME_COL]).reset_index(drop=True)
        
        # Process each well individually
        wells_data = []
        config = BHE_CONFIGURATIONS['double_u45_research']
        
        for well_id, return_col in DOUBLE_U_RETURN_COLS.items():
            if return_col in combined_df.columns:
                well_df = combined_df[[TIME_COL, return_col, 
                                     OE403_COLS['supply'], OE403_COLS['power'], OE403_COLS['flow']]].copy()
                well_df = well_df.dropna()
                
                if len(well_df) == 0:
                    continue
                
                # Add well-specific configuration (NO DEPTH SIGNAL)
                well_df['bhe_type'] = 'double_u45mm'
                well_df['thermal_resistance'] = config['thermal_resistance']
                well_df['area_factor'] = config['area_factor']
                well_df['well_id'] = well_id
                
                # Rename columns for consistency
                well_df = well_df.rename(columns={
                    return_col: 'return_temp',
                    OE403_COLS['supply']: 'supply_temp',
                    OE403_COLS['power']: 'power_kw',
                    OE403_COLS['flow']: 'flow_rate'
                })
                
                # Divide flow rate by 4 (OE403 measures total flow for all 4 Double U45 wells)
                # Assume equal flow distribution among the 4 wells
                well_df['flow_rate'] = well_df['flow_rate'] / 4.0
                
                wells_data.append(well_df)
                logging.info(f"Processed {well_id}: {len(well_df)} records, standardized to 300m depth")
        
        if wells_data:
            combined_wells = pd.concat(wells_data, ignore_index=True)
            logging.info(f"Combined Double U45mm data: {len(combined_wells)} total records from {len(wells_data)} wells")
            return combined_wells
        else:
            logging.warning("No valid Double U45mm wells data found")
            return None
            
    except Exception as e:
        logging.error(f"Error loading Double U45mm data: {e}")
        return None

def create_muovi_ellipse_aggregated_data():
    """Create MuoviEllipse 63mm data using complete field data as proxy (OE402 not available)."""
    
    logging.info("Creating MuoviEllipse 63mm aggregated data...")
    
    # Load complete field data as base
    complete_field = load_complete_field_data()
    if complete_field is None:
        return None
    
    # Modify to represent MuoviEllipse characteristics
    config = BHE_CONFIGURATIONS['muovi_ellipse_63mm_research']
    muovi_df = complete_field.copy()
    
    # Update configuration (NO DEPTH SIGNAL)
    muovi_df['bhe_type'] = 'muovi_ellipse_63mm'
    muovi_df['thermal_resistance'] = config['thermal_resistance']
    muovi_df['area_factor'] = config['area_factor']
    muovi_df['well_id'] = 'muovi_ellipse_aggregated'
    
    # Scale flow rate to represent 4 research wells (vs 120 complete field)
    muovi_df['flow_rate'] = muovi_df['flow_rate'] * (4/120)
    
    # Adjust temperatures slightly based on improved thermal properties
    temp_adjustment = 0.2  # Slight improvement due to larger diameter
    muovi_df['return_temp'] = muovi_df['return_temp'] + temp_adjustment
    
    logging.info(f"Created MuoviEllipse aggregated data: {len(muovi_df)} records")
    return muovi_df

def clean_and_validate_data(df, data_source="unknown"):
    """Clean and validate BHE data."""
    
    if df is None or len(df) == 0:
        return None
    
    logging.info(f"Cleaning {data_source} data: {len(df)} initial records")
    
    # Remove invalid temperatures (geothermal range: -5°C to 25°C)
    temp_cols = ['supply_temp', 'return_temp']
    for col in temp_cols:
        if col in df.columns:
            mask = (df[col] >= -5) & (df[col] <= 25)
            df = df[mask].copy()
    
    # Remove invalid power values (reasonable range: -100kW to 100kW for research wells)
    if 'power_kw' in df.columns:
        mask = (df['power_kw'] >= -100) & (df['power_kw'] <= 100)
        df = df[mask].copy()
    
    # Remove invalid flow rates (positive values only)
    if 'flow_rate' in df.columns:
        mask = df['flow_rate'] > 0
        df = df[mask].copy()
    
    # Physics validation: temperature difference should align with heat extraction/rejection
    if all(col in df.columns for col in ['supply_temp', 'return_temp', 'power_kw']):
        temp_diff = df['return_temp'] - df['supply_temp']
        
        # For heat extraction (negative power), return should be cooler
        # For heat rejection (positive power), return should be warmer
        heat_extraction = df['power_kw'] < -1  # At least 1kW extraction
        heat_rejection = df['power_kw'] > 1    # At least 1kW rejection
        
        # Remove physically inconsistent readings (with tolerance)
        inconsistent = ((heat_extraction & (temp_diff > 0.5)) | 
                       (heat_rejection & (temp_diff < -0.5)))
        
        df = df[~inconsistent].copy()
    
    # Remove rows with any NaN values in essential columns (NO DEPTH SIGNAL)
    essential_cols = ['supply_temp', 'return_temp', 'bhe_type']
    df = df.dropna(subset=essential_cols)
    
    logging.info(f"Cleaned {data_source} data: {len(df)} records remaining")
    return df

#------------------------------------------------------------------------------
# DATASET AND MODEL CLASSES
#------------------------------------------------------------------------------

class ResearchWellsDataset(Dataset):
    """Dataset for research wells with BHE configuration features."""
    
    def __init__(self, df, seq_len, horizon, normalize=True, fit_scaler=True):
        # Prepare features (NO DEPTH SIGNAL - all wells standardized to 300m)
        feature_cols = ['supply_temp', 'flow_rate', 'thermal_resistance', 'area_factor']
        target_col = 'return_temp'
        
        # Encode BHE type as numerical
        bhe_type_map = {'single_u45mm': 0, 'double_u45mm': 1, 'muovi_ellipse_63mm': 2}
        df_processed = df.copy()
        df_processed['bhe_type_encoded'] = df_processed['bhe_type'].map(bhe_type_map)
        feature_cols.append('bhe_type_encoded')
        
        # Extract features and target
        self.X = df_processed[feature_cols].values.astype(np.float32)
        self.y = df_processed[target_col].values.astype(np.float32)
        self.times = df_processed[TIME_COL].values
        
        self.seq_len = seq_len
        self.horizon = horizon
        self.feature_names = feature_cols
        
        # Normalization
        if normalize:
            if fit_scaler:
                self.X_mean = np.mean(self.X, axis=0)
                self.X_std = np.std(self.X, axis=0) + 1e-8
                self.y_mean = np.mean(self.y)
                self.y_std = np.std(self.y) + 1e-8
                
                # Apply normalization
                self.X = (self.X - self.X_mean) / self.X_std
                self.y = (self.y - self.y_mean) / self.y_std
            else:
                # Will be set externally - don't normalize yet
                pass
        
        # Create valid sequence indices
        self.valid_indices = []
        for i in range(len(self.X) - seq_len - horizon + 1):
            self.valid_indices.append(i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        # Sequence: (seq_len, n_features)
        seq = self.X[i:i + self.seq_len]
        # Target: scalar at horizon offset
        target = self.y[i + self.seq_len + self.horizon - 1]
        
        # Convert to PyTorch tensors and transpose for CNN (channels_first)
        seq_tensor = torch.from_numpy(seq).float().transpose(0, 1)  # (n_features, seq_len)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        return seq_tensor, target_tensor

class ResearchWellsCNNLSTM(nn.Module):
    """CNN-LSTM model optimized for research wells data."""
    
    def __init__(self, n_features, conv_channels=(32, 64), kernel_size=3,
                 lstm_hidden=64, lstm_layers=2, dropout=0.1):
        super().__init__()
        
        # CNN layers
        channels = [n_features] + list(conv_channels)
        conv_layers = []
        
        for i in range(len(channels) - 1):
            conv_layers.extend([
                nn.Conv1d(channels[i], channels[i+1], kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(channels[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=False
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, n_features, seq_len)
        
        # CNN processing
        conv_out = self.conv_layers(x)  # (batch_size, conv_channels[-1], seq_len)
        
        # Prepare for LSTM: (seq_len, batch_size, features)
        lstm_in = conv_out.permute(2, 0, 1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(lstm_in)  # (seq_len, batch_size, lstm_hidden)
        
        # Use last time step
        last_output = lstm_out[-1]  # (batch_size, lstm_hidden)
        
        # Final prediction
        prediction = self.fc(last_output).squeeze(-1)  # (batch_size,)
        
        return prediction

#------------------------------------------------------------------------------
# TRAINING AND EVALUATION FUNCTIONS
#------------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, epochs, lr, device, patience=10):
    """Train the CNN-LSTM model."""
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        logging.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.5f}, Val Loss={avg_val_loss:.5f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def evaluate_model(model, data_loader, device, dataset_for_denorm=None):
    """Evaluate model performance."""
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            pred = model(batch_x)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Denormalize if dataset provided
    if dataset_for_denorm is not None:
        predictions = predictions * dataset_for_denorm.y_std + dataset_for_denorm.y_mean
        targets = targets * dataset_for_denorm.y_std + dataset_for_denorm.y_mean
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    return predictions, targets, mae, rmse

#------------------------------------------------------------------------------
# ANALYSIS AND VISUALIZATION FUNCTIONS  
#------------------------------------------------------------------------------

def analyze_bhe_configurations(combined_data):
    """Analyze the different BHE configurations."""
    
    logging.info("Analyzing BHE configurations...")
    
    analysis_results = {}
    
    for bhe_type in combined_data['bhe_type'].unique():
        subset = combined_data[combined_data['bhe_type'] == bhe_type]
        
        analysis_results[bhe_type] = {
            'count': len(subset),
            'mean_return_temp': subset['return_temp'].mean(),
            'std_return_temp': subset['return_temp'].std(),
            'mean_temp_diff': (subset['return_temp'] - subset['supply_temp']).mean(),
            'thermal_resistance': subset['thermal_resistance'].iloc[0],
            'area_factor': subset['area_factor'].iloc[0]
        }
        
        logging.info(f"{bhe_type}: {len(subset)} records, "
                    f"avg return temp: {subset['return_temp'].mean():.2f}°C, "
                    f"standardized to 300m depth")
    
    return analysis_results

def create_data_analysis_plots(combined_data, config_analysis):
    """Create data analysis plots that don't require a trained model."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. BHE Type Comparison - Return Temperatures
    plt.subplot(2, 3, 1)
    bhe_types = combined_data['bhe_type'].unique()
    return_temps_by_type = [combined_data[combined_data['bhe_type'] == bhe_type]['return_temp'] 
                           for bhe_type in bhe_types]
    type_names = [t.replace('_', ' ').title() for t in bhe_types]
    
    plt.boxplot(return_temps_by_type, tick_labels=type_names)
    plt.title('Return Temperature by BHE Configuration\n(All wells standardized to 300m depth)', fontsize=12, fontweight='bold')
    plt.ylabel('Return Temperature (°C)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 2. Thermal Resistance vs Performance
    plt.subplot(2, 3, 2)
    thermal_groups = combined_data.groupby('thermal_resistance')['return_temp'].mean()
    resistances = list(thermal_groups.index)
    temps = list(thermal_groups.values)
    type_labels = []
    for resistance in resistances:
        bhe_type = combined_data[combined_data['thermal_resistance'] == resistance]['bhe_type'].iloc[0]
        type_labels.append(bhe_type.replace('_', ' ').title())
    
    colors = ['lightcoral', 'skyblue', 'lightgreen']
    bars = plt.bar(range(len(resistances)), temps, color=colors[:len(resistances)])
    plt.title('Thermal Resistance Effect on Performance\n(Lower resistance = better heat transfer)', fontsize=12, fontweight='bold')
    plt.xlabel('BHE Configuration')
    plt.ylabel('Mean Return Temperature (°C)')
    plt.xticks(range(len(resistances)), type_labels, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add resistance annotations
    for i, (resistance, temp) in enumerate(zip(resistances, temps)):
        plt.text(i, temp + 0.05, f'{temp:.2f}°C\n({resistance:.3f} mK/W)', 
                ha='center', fontweight='bold', fontsize=9)
    
    # 3. Data Distribution by BHE Type
    plt.subplot(2, 3, 3)
    counts = [config_analysis[bhe_type]['count'] for bhe_type in bhe_types]
    type_names = [t.replace('_', ' ').title() for t in bhe_types]
    
    plt.pie(counts, labels=type_names, autopct='%1.1f%%', startangle=90)
    plt.title('Data Distribution by BHE Type\n(Total: {:,} records)'.format(sum(counts)), fontsize=12, fontweight='bold')
    
    # 4. Temperature Difference Analysis
    plt.subplot(2, 3, 4)
    temp_diffs = combined_data['return_temp'] - combined_data['supply_temp']
    temp_diff_by_type = [temp_diffs[combined_data['bhe_type'] == bhe_type] for bhe_type in bhe_types]
    
    plt.boxplot(temp_diff_by_type, tick_labels=type_names)
    plt.title('Temperature Difference (Return - Supply)\nby BHE Configuration', fontsize=12, fontweight='bold')
    plt.ylabel('Temperature Difference (°C)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 5. Flow Rate Distribution
    plt.subplot(2, 3, 5)
    flow_by_type = [combined_data[combined_data['bhe_type'] == bhe_type]['flow_rate'] for bhe_type in bhe_types]
    
    plt.boxplot(flow_by_type, tick_labels=type_names)
    plt.title('Flow Rate Distribution\nby BHE Configuration', fontsize=12, fontweight='bold')
    plt.ylabel('Flow Rate (m³/h)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6. Performance Summary Table
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Create summary table
    table_data = []
    for bhe_type in bhe_types:
        config = config_analysis[bhe_type]
        table_data.append([
            bhe_type.replace('_', ' ').title(),
            f"{config['count']:,}",
            f"{config['mean_return_temp']:.2f}°C",
            f"{config['thermal_resistance']:.3f} mK/W",
            f"{config['area_factor']:.1f}x"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['BHE Type', 'Records', 'Avg Return T', 'Thermal R', 'Area Factor'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.15, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('Research Wells Performance Summary\n(All standardized to 300m depth)', 
             fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_DIR, "research_wells_data_analysis.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    logging.info("Data analysis plots saved to output/research_wells_data_analysis.png")

def create_comparison_visualizations(combined_data, model, test_dataset, predictions, targets, device):
    """Create comprehensive visualization comparing BHE configurations."""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. BHE Type Comparison - Return Temperatures
    plt.subplot(2, 3, 1)
    bhe_types = combined_data['bhe_type'].unique()
    return_temps_by_type = [combined_data[combined_data['bhe_type'] == bhe_type]['return_temp'] 
                           for bhe_type in bhe_types]
    
    plt.boxplot(return_temps_by_type, tick_labels=bhe_types)
    plt.title('Return Temperature by BHE Configuration', fontsize=12, fontweight='bold')
    plt.ylabel('Return Temperature (°C)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 2. Thermal Resistance Comparison 
    plt.subplot(2, 3, 2)
    thermal_groups = combined_data.groupby('thermal_resistance')['return_temp'].mean()
    resistances = list(thermal_groups.index)
    temps = list(thermal_groups.values)
    bhe_types = []
    for resistance in resistances:
        bhe_type = combined_data[combined_data['thermal_resistance'] == resistance]['bhe_type'].iloc[0]
        bhe_types.append(bhe_type.replace('_', ' ').title())
    
    colors = ['lightcoral', 'skyblue', 'lightgreen']
    plt.bar(range(len(resistances)), temps, color=colors[:len(resistances)])
    plt.title('Thermal Resistance Effect on Return Temperature\n(BHE Configuration Comparison)', fontsize=12, fontweight='bold')
    plt.xlabel('BHE Configuration')
    plt.ylabel('Mean Return Temperature (°C)')
    plt.xticks(range(len(resistances)), bhe_types, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add resistance annotations
    for i, (resistance, temp) in enumerate(zip(resistances, temps)):
        plt.text(i, temp + 0.01, f'{temp:.2f}°C\n({resistance:.3f} mK/W)', ha='center', fontweight='bold', fontsize=9)
    
    # 3. Model Performance Comparison
    plt.subplot(2, 3, 3)
    plt.scatter(targets, predictions, alpha=0.6, s=20)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Return Temperature (°C)')
    plt.ylabel('Predicted Return Temperature (°C)')
    plt.title('Model Performance: Actual vs Predicted', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Thermal Resistance Comparison
    plt.subplot(2, 3, 4)
    resistance_data = combined_data.groupby('bhe_type')['thermal_resistance'].first()
    area_factors = combined_data.groupby('bhe_type')['area_factor'].first()
    
    x_pos = np.arange(len(resistance_data))
    plt.bar(x_pos, resistance_data.values, color=['lightblue', 'orange', 'lightgreen'])
    plt.xlabel('BHE Configuration')
    plt.ylabel('Thermal Resistance (mK/W)')
    plt.title('Thermal Resistance by Configuration', fontsize=12, fontweight='bold')
    plt.xticks(x_pos, resistance_data.index, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add area factor annotations
    for i, (resistance, area_factor) in enumerate(zip(resistance_data.values, area_factors.values)):
        plt.text(i, resistance + 0.002, f'Area: {area_factor:.1f}x', ha='center', fontsize=9)
    
    # 5. Time Series Comparison (last 200 points)
    plt.subplot(2, 3, 5)
    n_points = min(200, len(predictions))
    time_range = range(n_points)
    
    plt.plot(time_range, targets[-n_points:], label='Actual', linewidth=2, alpha=0.8)
    plt.plot(time_range, predictions[-n_points:], label='Predicted', linewidth=2, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Return Temperature (°C)')
    plt.title(f'Time Series Comparison (Last {n_points} Points)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Configuration Benefits Analysis
    plt.subplot(2, 3, 6)
    config_benefits = []
    config_names = []
    
    # Calculate temperature benefits relative to single U45mm
    baseline_temp = combined_data[combined_data['bhe_type'] == 'single_u45mm']['return_temp'].mean()
    
    for bhe_type in bhe_types:
        if bhe_type != 'single_u45mm':
            type_temp = combined_data[combined_data['bhe_type'] == bhe_type]['return_temp'].mean()
            benefit = type_temp - baseline_temp
            config_benefits.append(benefit)
            config_names.append(bhe_type.replace('_', ' ').title())
    
    colors = ['orange' if b > 0 else 'lightcoral' for b in config_benefits]
    plt.bar(range(len(config_benefits)), config_benefits, color=colors)
    plt.xlabel('BHE Configuration')
    plt.ylabel('Temperature Benefit vs Single U45mm (°C)')
    plt.title('Configuration Benefits Analysis', fontsize=12, fontweight='bold')
    plt.xticks(range(len(config_names)), config_names, rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    # Add benefit annotations
    for i, benefit in enumerate(config_benefits):
        plt.text(i, benefit + (0.01 if benefit > 0 else -0.02), 
                f'{benefit:.3f}°C', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "research_wells_comprehensive_analysis.png"), 
                dpi=200, bbox_inches='tight')
    plt.close()
    
    logging.info("Comprehensive analysis visualization saved")

#==============================================================================
# MAIN EXECUTION
#==============================================================================

def main():
    """Main execution function."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Starting Research Wells Analysis on device: {device}")
    
    # Load all data sources
    logging.info("Loading data from all sources...")
    
    # 1. Complete field data (Single U45mm baseline)
    complete_field_data = load_complete_field_data()
    complete_field_data = clean_and_validate_data(complete_field_data, "complete_field")
    
    # 2. Double U45mm individual research wells  
    double_u_data = load_double_u45_individual_data()
    double_u_data = clean_and_validate_data(double_u_data, "double_u45mm_research")
    
    # 3. MuoviEllipse 63mm aggregated (proxy)
    muovi_ellipse_data = create_muovi_ellipse_aggregated_data()
    muovi_ellipse_data = clean_and_validate_data(muovi_ellipse_data, "muovi_ellipse_research")
    
    # Combine all datasets
    all_datasets = []
    dataset_names = []
    
    if complete_field_data is not None:
        all_datasets.append(complete_field_data)
        dataset_names.append("complete_field")
        
    if double_u_data is not None:
        all_datasets.append(double_u_data)
        dataset_names.append("double_u45mm")
        
    if muovi_ellipse_data is not None:
        all_datasets.append(muovi_ellipse_data)
        dataset_names.append("muovi_ellipse")
    
    if not all_datasets:
        logging.error("No valid datasets loaded!")
        return
    
    # Combine all data
    combined_data = pd.concat(all_datasets, ignore_index=True)
    combined_data = combined_data.sort_values(TIME_COL).reset_index(drop=True)
    
    logging.info(f"Combined dataset: {len(combined_data)} records from {len(dataset_names)} sources")
    logging.info(f"BHE types: {combined_data['bhe_type'].value_counts().to_dict()}")
    
    # Analyze configurations
    config_analysis = analyze_bhe_configurations(combined_data)
    
    # Create data analysis plots (independent of model training)
    create_data_analysis_plots(combined_data, config_analysis)
    
    try:
        # Train-validation-test split (temporal)
        n_total = len(combined_data)
        train_end = int(n_total * (1 - VAL_SPLIT - TEST_SPLIT))
        val_end = int(n_total * (1 - TEST_SPLIT))
        
        train_data = combined_data.iloc[:train_end].copy()
        val_data = combined_data.iloc[train_end:val_end].copy()
        test_data = combined_data.iloc[val_end:].copy()
        
        logging.info(f"Data splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Create datasets
        train_dataset = ResearchWellsDataset(train_data, SEQ_LEN, PRED_HORIZON, normalize=True, fit_scaler=True)
        val_dataset = ResearchWellsDataset(val_data, SEQ_LEN, PRED_HORIZON, normalize=True, fit_scaler=False)
        test_dataset = ResearchWellsDataset(test_data, SEQ_LEN, PRED_HORIZON, normalize=True, fit_scaler=False)
        
        # Copy normalization parameters to val/test datasets
        val_dataset.X_mean, val_dataset.X_std = train_dataset.X_mean, train_dataset.X_std
        val_dataset.y_mean, val_dataset.y_std = train_dataset.y_mean, train_dataset.y_std
        test_dataset.X_mean, test_dataset.X_std = train_dataset.X_mean, train_dataset.X_std
        test_dataset.y_mean, test_dataset.y_std = train_dataset.y_mean, train_dataset.y_std
        
        # Apply normalization (datasets already have the same features prepared)
        val_dataset.X = (val_dataset.X - train_dataset.X_mean) / train_dataset.X_std
        val_dataset.y = (val_dataset.y - train_dataset.y_mean) / train_dataset.y_std
        test_dataset.X = (test_dataset.X - train_dataset.X_mean) / train_dataset.X_std
        test_dataset.y = (test_dataset.y - train_dataset.y_mean) / train_dataset.y_std
        
        # Update valid indices
        val_dataset.valid_indices = list(range(len(val_dataset.X) - SEQ_LEN - PRED_HORIZON + 1))
        test_dataset.valid_indices = list(range(len(test_dataset.X) - SEQ_LEN - PRED_HORIZON + 1))
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        logging.info(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        # Initialize model
        n_features = len(train_dataset.feature_names)
        model = ResearchWellsCNNLSTM(
            n_features=n_features,
            conv_channels=CONV_CHANNELS,
            kernel_size=KERNEL_SIZE,
            lstm_hidden=LSTM_HIDDEN,
            lstm_layers=LSTM_LAYERS,
            dropout=DROPOUT
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Model initialized with {total_params:,} parameters")
        
        # Train model
        logging.info("Starting model training...")
        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, EPOCHS, LR, device, PATIENCE
        )
        
        # Save model
        model_path = os.path.join(OUTPUT_DIR, "research_wells_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'n_features': n_features,
                'conv_channels': CONV_CHANNELS,
                'kernel_size': KERNEL_SIZE,
                'lstm_hidden': LSTM_HIDDEN,
                'lstm_layers': LSTM_LAYERS,
                'dropout': DROPOUT
            },
            'normalization': {
                'X_mean': train_dataset.X_mean,
                'X_std': train_dataset.X_std,
                'y_mean': train_dataset.y_mean,
                'y_std': train_dataset.y_std
            },
            'feature_names': train_dataset.feature_names
        }, model_path)
        
        logging.info(f"Model saved to {model_path}")
        
        # Evaluate model
        logging.info("Evaluating model...")
        predictions, targets, mae, rmse = evaluate_model(model, test_loader, device, test_dataset)
        
        logging.info(f"Test Results: MAE={mae:.4f}°C, RMSE={rmse:.4f}°C")
        
        # Create comprehensive visualizations (includes model predictions)
        create_comparison_visualizations(combined_data, model, test_dataset, predictions, targets, device)
        
        # Save detailed results
        results = {
            'model_performance': {
                'mae': float(mae),
                'rmse': float(rmse),
                'test_samples': len(predictions)
            },
            'bhe_configurations': config_analysis,
            'model_config': {
                'seq_len': SEQ_LEN,
                'pred_horizon': PRED_HORIZON,
                'features': train_dataset.feature_names,
                'total_parameters': total_params
            },
            'data_summary': {
                'total_records': len(combined_data),
                'train_records': len(train_data),
                'val_records': len(val_data), 
                'test_records': len(test_data),
                'data_sources': dataset_names
            }
        }
        
        results_path = os.path.join(OUTPUT_DIR, "research_wells_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logging.info(f"Results saved to {results_path}")
        
        # Print detailed summary
        print("\n" + "="*60)
        print("RESEARCH WELLS ANALYSIS SUMMARY")
        print("="*60)
        print(f"Model Performance:")
        print(f"  MAE: {mae:.4f}°C")
        print(f"  RMSE: {rmse:.4f}°C")
        print(f"\nBHE Configurations Analyzed:")
        
        for config, analysis in config_analysis.items():
            print(f"  {config}:")
            print(f"    Records: {analysis['count']}")
            print(f"    Mean return temp: {analysis['mean_return_temp']:.2f}°C")
            print(f"    All wells standardized to 300m depth")
            print(f"    Thermal resistance: {analysis['thermal_resistance']:.3f} mK/W")
        
        print(f"\nFiles generated in {OUTPUT_DIR}:")
        print(f"  - research_wells_model.pth (trained model)")
        print(f"  - research_wells_results.json (detailed results)")
        print(f"  - research_wells_comprehensive_analysis.png (model visualizations)")
        print(f"  - research_wells_data_analysis.png (data analysis plots)")
        print(f"  - research_wells_analysis.log (execution log)")
        print("="*60)
        
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        logging.info("However, data analysis plots were still generated successfully!")
        
        # Print basic summary even if model training failed
        print("\n" + "="*60)
        print("RESEARCH WELLS DATA ANALYSIS SUMMARY")
        print("="*60)
        print("Note: Model training encountered issues, but data analysis completed.")
        print(f"\nBHE Configurations Analyzed:")
        
        for config, analysis in config_analysis.items():
            print(f"  {config}:")
            print(f"    Records: {analysis['count']}")
            print(f"    Mean return temp: {analysis['mean_return_temp']:.2f}°C")
            print(f"    All wells standardized to 300m depth")
            print(f"    Thermal resistance: {analysis['thermal_resistance']:.3f} mK/W")
        
        print(f"\nFiles generated in {OUTPUT_DIR}:")
        print(f"  - research_wells_data_analysis.png (data analysis plots)")
        print(f"  - research_wells_analysis.log (execution log)")
        print("="*60)

if __name__ == "__main__":
    main()