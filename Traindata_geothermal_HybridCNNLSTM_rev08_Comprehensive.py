"""
CNN-LSTM Model for Comprehensive BHE Configuration Analysis

Combines:
- Data handling from rev07 (pure sensor data)
- Training architecture from rev03 (proper CNN-LSTM)
- Advanced visualization (time-series comparison)

Features: supply_temp, return_temp, flow_rate, power_kw, bhe_type
- 120x Single U45mm wells (complete field data)
- 4x Double U45mm wells (research wells SKD-110-01 to 04)  
- 4x wells at 300m depth (equal flow distribution)
"""

from typing import List, Dict, Optional
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model hyperparameters
SEQ_LEN = 48
PRED_HORIZON = 1
BATCH_SIZE = 1024  # Maximum GPU utilization - can handle 1024 with only 0.7% VRAM usage
EPOCHS = 10
LR = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.25

# CNN-LSTM architecture
CONV_CHANNELS = [32, 64]
KERNEL_SIZE = 3
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
DROPOUT = 0.1
PATIENCE = 16

# Setup logging with both file and console output
log_file_path = os.path.join(OUTPUT_DIR, "comprehensive_analysis.log")

# Create a custom formatter
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

# Console handler - always shows output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handler - saves to log file
file_handler = logging.FileHandler(log_file_path, mode="w")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler],
    force=True  # Force reconfiguration if already configured
)

def load_complete_field_data():
    """Load and process complete field data (120 boreholes, Single U45mm)."""
    
    csv_path = os.path.join(os.path.dirname(__file__), "input/Borehole heat extraction complete field.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Complete field data not found: {csv_path}")
    
    logging.info("Loading complete field data...")
    
    try:
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        logging.info(f"Raw complete field data loaded: {len(df)} records")
        
        # Process timestamp
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M', errors="coerce")
        df = df.sort_values('Timestamp').dropna(subset=['Timestamp']).reset_index(drop=True)
        
        # Rename columns for consistency
        df = df.rename(columns={
            'Supply temperature measure at energy meter [°C]': 'supply_temp',
            'Return temperature measure at energy meter [°C]': 'return_temp',
            'Flow rate to 120 boreholes [m³/h]': 'flow_rate',
            'Negative Heat extracion [kW] / Positive Heat rejection [kW]': 'power_kw'  # Negative = extraction, Positive = rejection
        })
        
        # Keep only essential columns
        essential_cols = ['Timestamp', 'supply_temp', 'return_temp', 'flow_rate', 'power_kw']
        df = df[essential_cols].copy()
        
        # Add BHE type
        df['bhe_type'] = 'single_u45mm'
        df['bhe_type_encoded'] = 0  # Single U45mm = 0
        
        logging.info(f"Loaded complete field data: {len(df)} records")
        return df
        
    except Exception as e:
        logging.error(f"Error loading complete field data: {e}")
        raise

def load_double_u45mm_research_data():
    """Load and process Double U45mm research wells data from MeterOE403."""
    
    # Load shared sensor data for Double U45mm wells
    oe403_path = os.path.join(os.path.dirname(__file__), "input/MeterOE403_doubleU45.csv")
    
    if not os.path.exists(oe403_path):
        logging.warning("Double U45mm research data file not found, skipping...")
        return pd.DataFrame()
    
    logging.info("Loading Double U45mm research wells data...")
    
    try:
        # Load shared sensor data
        oe403_df = pd.read_csv(oe403_path, sep=',', decimal='.')
        logging.info(f"Raw Double U45mm data loaded: {len(oe403_df)} records")
        
        # Process timestamp
        oe403_df['Timestamp'] = pd.to_datetime(oe403_df['Timestamp'], format='%d.%m.%Y %H:%M', errors="coerce")
        oe403_df = oe403_df.sort_values('Timestamp').dropna(subset=['Timestamp']).reset_index(drop=True)
        
        # Clean column names and data
        oe403_df.columns = oe403_df.columns.str.strip()
        
        # Clean and convert all numeric columns
        for col in ['Power [kW]', 'T_supply [°C]', 'T_return [°C]', 'Flow [m³/h]']:
            if col in oe403_df.columns:
                oe403_df[col] = pd.to_numeric(oe403_df[col].astype(str).str.strip(), errors='coerce')
        
        # Rename columns for consistency
        oe403_df = oe403_df.rename(columns={
            'T_supply [°C]': 'supply_temp',
            'T_return [°C]': 'return_temp',
            'Flow [m³/h]': 'flow_rate', 
            'Power [kW]': 'power_kw'  # Negative = heat extraction, Positive = heat rejection
        })
        
        # Keep only essential columns
        essential_cols = ['Timestamp', 'supply_temp', 'return_temp', 'flow_rate', 'power_kw']
        oe403_df = oe403_df[essential_cols].copy()
        
        # Clean data
        oe403_df = oe403_df.dropna()
        oe403_df = oe403_df[(oe403_df['supply_temp'] > -20) & (oe403_df['supply_temp'] < 50)]
        oe403_df = oe403_df[(oe403_df['return_temp'] > -20) & (oe403_df['return_temp'] < 50)]
        oe403_df = oe403_df[oe403_df['flow_rate'] > 0]
        oe403_df = oe403_df[np.abs(oe403_df['power_kw']) < 1000]
        
        # Add BHE type
        oe403_df['bhe_type'] = 'double_u45mm'
        oe403_df['bhe_type_encoded'] = 1  # Double U45mm = 1
        
        logging.info(f"Processed Double U45mm data: {len(oe403_df)} records")
        return oe403_df
            
    except Exception as e:
        logging.error(f"Error loading Double U45mm research data: {e}")
        print(f"ERROR loading Double U45mm research data: {e}")  # Debug output
        return pd.DataFrame()

def load_muovi_ellipse_research_data():
    """Load and process MuoviEllipse 63mm research data."""
    
    csv_path = os.path.join(os.path.dirname(__file__), "input/MeterOE401_Ellipse63.csv")
    
    if not os.path.exists(csv_path):
        logging.warning("MuoviEllipse 63mm data file not found, skipping...")
        return pd.DataFrame()
    
    logging.info("Loading MuoviEllipse 63mm research data...")
    
    try:
        df = pd.read_csv(csv_path, sep=',', decimal='.')
        logging.info(f"Raw MuoviEllipse data loaded: {len(df)} records")
        
        # Process timestamp
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M', errors="coerce")
        df = df.sort_values('Timestamp').dropna(subset=['Timestamp']).reset_index(drop=True)
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Clean and convert all numeric columns
        for col in ['Power [kW]', 'T_supply [°C]', 'T_return [°C]', 'Flow [m³/h]']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
        
        # Rename columns for consistency 
        df = df.rename(columns={
            'T_supply [°C]': 'supply_temp',
            'T_return [°C]': 'return_temp', 
            'Flow [m³/h]': 'flow_rate',
            'Power [kW]': 'power_kw'  # Negative = heat extraction, Positive = heat rejection
        })
        
        # Keep only essential columns
        essential_cols = ['Timestamp', 'supply_temp', 'return_temp', 'flow_rate', 'power_kw']
        df = df[essential_cols].copy()
        
        # Add BHE type
        df['bhe_type'] = 'muovi_ellipse_63mm'
        df['bhe_type_encoded'] = 2  # MuoviEllipse 63mm = 2
        
        # Clean data
        df = df.dropna()
        df = df[(df['supply_temp'] > -20) & (df['supply_temp'] < 50)]
        df = df[(df['return_temp'] > -20) & (df['return_temp'] < 50)]
        df = df[df['flow_rate'] > 0]
        df = df[np.abs(df['power_kw']) < 1000]  # Remove extreme power values
        
        logging.info(f"Processed MuoviEllipse 63mm data: {len(df)} records")
        return df
        
    except Exception as e:
        logging.error(f"Error loading MuoviEllipse 63mm data: {e}")
        print(f"ERROR loading MuoviEllipse 63mm data: {e}")  # Debug output
        return pd.DataFrame()

def create_muovi_ellipse_research_data():
    """DEPRECATED: Use load_muovi_ellipse_research_data() instead."""
    
    logging.warning("create_muovi_ellipse_research_data() is deprecated - use load_muovi_ellipse_research_data()")
    return load_muovi_ellipse_research_data()

def clean_bhe_data(df, name=""):
    """Clean and validate BHE data."""
    
    if len(df) == 0:
        return df
    
    logging.info(f"Cleaning {name} data: {len(df)} initial records")
    
    # Remove obvious outliers
    df = df[(df['supply_temp'] > -20) & (df['supply_temp'] < 50)]
    df = df[(df['return_temp'] > -20) & (df['return_temp'] < 50)]
    df = df[df['flow_rate'] > 0]
    df = df[np.abs(df['power_kw']) < 1000]  # Remove extreme power values
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Remove rows with NaN values
    df = df.dropna()
    
    logging.info(f"Cleaned {name} data: {len(df)} records remaining")
    return df

class ComprehensiveDataset(Dataset):
    """Dataset class for comprehensive BHE analysis."""
    
    def __init__(self, df, seq_len, horizon, feature_cols, target_col, mean=None, std=None):
        self.seq_len = seq_len
        self.horizon = horizon
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Prepare data
        self.data = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)
        
        # Standardization
        if mean is None or std is None:
            self.mean = np.mean(self.data, axis=0)
            self.std = np.std(self.data, axis=0) + 1e-8
        else:
            self.mean = mean
            self.std = std
        
        # Standardize features
        self.data = (self.data - self.mean) / self.std
        
        # Create sequences
        self.sequences = []
        self.labels = []
        
        for i in range(len(self.data) - seq_len - horizon + 1):
            self.sequences.append(self.data[i:i+seq_len])
            self.labels.append(self.targets[i+seq_len+horizon-1])
        
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)
        
        logging.info(f"Created dataset with {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.labels[idx]])

class ComprehensiveCNNLSTM(nn.Module):
    """Comprehensive CNN-LSTM model for BHE analysis."""
    
    def __init__(self, input_features, conv_channels, kernel_size, lstm_hidden, lstm_layers, dropout):
        super(ComprehensiveCNNLSTM, self).__init__()
        
        self.input_features = input_features
        
        # CNN layers
        layers = []
        in_channels = input_features
        
        for out_channels in conv_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        batch_size, seq_len, features = x.shape
        
        # Reshape for CNN: (batch_size, features, seq_len)
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        x = self.conv_layers(x)
        
        # Reshape back for LSTM: (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        x = lstm_out[:, -1, :]
        
        # Output layers
        x = self.output_layers(x)
        
        return x

def train_model(model, train_loader, val_loader, epochs, lr, device, patience):
    """Train the CNN-LSTM model with optimized performance and GPU monitoring."""
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience, verbose=True
    )
    
    # Enable mixed precision for better GPU utilization if using CUDA
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    # Monitor GPU memory if CUDA is available
    if device == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()  # Clear cache before training
    
    logging.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        # Monitor GPU memory at start of epoch
        if device == 'cuda' and epoch == 1:
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"GPU memory allocated before training: {allocated_before:.2f} GB")
        
        # Training loop with potential mixed precision
        for batch_data, batch_targets in train_loader:
            batch_data = batch_data.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler is not None:  # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:  # Standard training
                outputs = model(batch_data)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_data, batch_targets in val_loader:
                batch_data = batch_data.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_data)
                        loss = criterion(outputs, batch_targets)
                else:
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_targets)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Log with GPU memory info
        if epoch % 2 == 0 or epoch == 1 or patience_counter >= patience:
            gpu_info = ""
            if device == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                gpu_info = f", GPU: {allocated:.1f}GB/{cached:.1f}GB"
            
            logging.info(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                        f"Best Val: {best_val_loss:.6f}, Patience: {patience_counter}/{patience}{gpu_info}")
        
        if patience_counter >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final GPU memory report
    if device == 'cuda':
        final_allocated = torch.cuda.memory_allocated() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        logging.info(f"Training complete - GPU memory: {final_allocated:.2f}GB allocated, {max_allocated:.2f}GB peak")
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}

def evaluate_model(model, data_loader, device):
    """Evaluate model and return predictions and targets."""
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data, batch_targets in data_loader:
            batch_data = batch_data.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            
            outputs = model(batch_data)
            
            # Keep everything on GPU and convert to CPU only at the end
            all_predictions.append(outputs.cpu())
            all_targets.append(batch_targets.cpu())
    
    # Concatenate all tensors first, then convert to numpy once
    predictions = torch.cat(all_predictions, dim=0).numpy().flatten()
    targets = torch.cat(all_targets, dim=0).numpy().flatten()
    
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    
    return predictions, targets, mae, rmse

def create_comprehensive_collector_analysis(test_df, predictions, targets, feature_cols, config_analysis):
    """Create comprehensive collector configuration analysis with multiple panels."""
    
    logging.info("Creating comprehensive collector configuration analysis...")
    
    # Prepare test data
    test_data = test_df.copy()
    
    # Add predictions with proper alignment
    if len(predictions) < len(test_data):
        padded_predictions = np.full(len(test_data), np.nan)
        padded_predictions[-len(predictions):] = predictions
        predictions = padded_predictions
    
    test_data['predicted_temp'] = predictions[:len(test_data)]
    test_data['actual_temp'] = test_data['return_temp']
    
    # Filter for consistent time range across all collector types
    valid_data = test_data.dropna(subset=['predicted_temp'])
    
    # Find common time range where all collector types have data
    type_counts = valid_data.groupby(['Timestamp', 'bhe_type']).size().unstack(fill_value=0)
    common_times = type_counts[(type_counts > 0).sum(axis=1) >= 2].index
    
    if len(common_times) > 1000:
        common_times = common_times[-1000:]  # Last 1000 time points for clarity
    
    plot_data = valid_data[valid_data['Timestamp'].isin(common_times)].copy()
    
    if len(plot_data) == 0:
        logging.warning("No valid data for comparison plot")
        return
    
    # Professional color scheme
    colors = {
        'single_u45mm': '#2E86AB',      # Deep blue
        'double_u45mm': '#A23B72',      # Deep magenta 
        'muovi_ellipse_63mm': '#F18F01' # Orange
    }
    
    # Create figure with 2x3 subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Time series comparison (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    for bhe_type in plot_data['bhe_type'].unique():
        type_data = plot_data[plot_data['bhe_type'] == bhe_type].sort_values('Timestamp')
        
        if len(type_data) == 0:
            continue
        
        color = colors.get(bhe_type, '#333333')
        label = bhe_type.replace('_', ' ').replace('mm', 'mm').title()
        
        # Plot actual temperatures
        ax1.plot(type_data['Timestamp'], type_data['actual_temp'], 
               color=color, alpha=0.8, linewidth=2.5, label=f'Actual {label}')
        
        # Plot predicted temperatures
        ax1.plot(type_data['Timestamp'], type_data['predicted_temp'], 
               color=color, alpha=0.6, linewidth=2, linestyle='--', 
               label=f'Predicted {label}')
    
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Outlet Temperature (°C)', fontsize=12, fontweight='bold')
    ax1.set_title('Collector Configuration Performance Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Performance metrics comparison
    ax2 = fig.add_subplot(gs[0, 2])
    
    collector_types = []
    mean_temps = []
    temp_benefits = []
    
    for bhe_type in plot_data['bhe_type'].unique():
        type_data = plot_data[plot_data['bhe_type'] == bhe_type]
        collector_types.append(bhe_type.replace('_', ' ').title())
        mean_temps.append(type_data['predicted_temp'].mean())
    
    # Calculate relative benefits
    if len(mean_temps) > 1:
        baseline = min(mean_temps)
        temp_benefits = [temp - baseline for temp in mean_temps]
    else:
        temp_benefits = [0] * len(mean_temps)
    
    bars = ax2.bar(range(len(collector_types)), temp_benefits, 
                   color=[colors.get(k.lower().replace(' ', '_').replace('mm', 'mm'), '#333333') 
                         for k in [t.lower().replace(' ', '_') for t in collector_types]])
    
    ax2.set_xlabel('Collector Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Temperature Benefit (°C)', fontsize=12, fontweight='bold')
    ax2.set_title('Relative Performance Benefits', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(collector_types)))
    ax2.set_xticklabels(collector_types, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, benefit) in enumerate(zip(bars, temp_benefits)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'+{benefit:.3f}°C', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Heat transfer efficiency analysis
    ax3 = fig.add_subplot(gs[1, 0])
    
    efficiency_data = []
    efficiency_labels = []
    
    for bhe_type in plot_data['bhe_type'].unique():
        type_data = plot_data[plot_data['bhe_type'] == bhe_type]
        # Calculate heat transfer efficiency as temperature difference per unit power
        temp_diff = type_data['return_temp'] - type_data['supply_temp']
        power_abs = np.abs(type_data['power_kw'])
        efficiency = temp_diff / (power_abs + 1e-6)  # Avoid division by zero
        
        efficiency_data.append(efficiency.mean())
        efficiency_labels.append(bhe_type.replace('_', ' ').title())
    
    bars3 = ax3.bar(range(len(efficiency_labels)), efficiency_data,
                    color=[colors.get(k.lower().replace(' ', '_').replace('mm', 'mm'), '#333333') 
                          for k in [t.lower().replace(' ', '_') for t in efficiency_labels]])
    
    ax3.set_xlabel('Collector Type', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Heat Transfer Efficiency\n(°C/kW)', fontsize=12, fontweight='bold')
    ax3.set_title('Heat Transfer Efficiency', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(efficiency_labels)))
    ax3.set_xticklabels(efficiency_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Flow rate response analysis
    ax4 = fig.add_subplot(gs[1, 1])
    
    flow_response_data = {}
    for bhe_type in plot_data['bhe_type'].unique():
        type_data = plot_data[plot_data['bhe_type'] == bhe_type]
        
        # Calculate temperature response vs flow rate
        flow_bins = np.linspace(type_data['flow_rate'].min(), type_data['flow_rate'].max(), 5)
        temp_response = []
        
        for i in range(len(flow_bins)-1):
            mask = (type_data['flow_rate'] >= flow_bins[i]) & (type_data['flow_rate'] < flow_bins[i+1])
            if mask.sum() > 0:
                temp_response.append(type_data[mask]['return_temp'].mean())
            else:
                temp_response.append(np.nan)
        
        flow_response_data[bhe_type] = {
            'flow_bins': (flow_bins[:-1] + flow_bins[1:]) / 2,
            'temp_response': temp_response
        }
    
    for bhe_type, data in flow_response_data.items():
        color = colors.get(bhe_type, '#333333')
        label = bhe_type.replace('_', ' ').title()
        ax4.plot(data['flow_bins'], data['temp_response'], 
                color=color, marker='o', linewidth=2.5, markersize=6, label=label)
    
    ax4.set_xlabel('Flow Rate (m³/h)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Return Temperature (°C)', fontsize=12, fontweight='bold')
    ax4.set_title('Flow Rate Response', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Statistical performance summary
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Calculate prediction accuracy for each collector type
    mae_by_type = {}
    rmse_by_type = {}
    
    for bhe_type in plot_data['bhe_type'].unique():
        type_data = plot_data[plot_data['bhe_type'] == bhe_type]
        valid_mask = ~(np.isnan(type_data['predicted_temp']) | np.isnan(type_data['actual_temp']))
        
        if valid_mask.sum() > 0:
            pred_vals = type_data[valid_mask]['predicted_temp']
            actual_vals = type_data[valid_mask]['actual_temp']
            
            mae_by_type[bhe_type] = mean_absolute_error(actual_vals, pred_vals)
            rmse_by_type[bhe_type] = np.sqrt(mean_squared_error(actual_vals, pred_vals))
    
    types = list(mae_by_type.keys())
    mae_vals = [mae_by_type[t] for t in types]
    rmse_vals = [rmse_by_type[t] for t in types]
    
    x_pos = np.arange(len(types))
    width = 0.35
    
    bars1 = ax5.bar(x_pos - width/2, mae_vals, width, label='MAE', 
                    color='#4CAF50', alpha=0.8)
    bars2 = ax5.bar(x_pos + width/2, rmse_vals, width, label='RMSE', 
                    color='#FF9800', alpha=0.8)
    
    ax5.set_xlabel('Collector Type', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Prediction Error (°C)', fontsize=12, fontweight='bold')
    ax5.set_title('Model Accuracy by Collector Type', fontsize=14, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([t.replace('_', ' ').title() for t in types], rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add summary statistics
    if 'muovi_ellipse_63mm' in config_analysis and 'double_u45mm' in config_analysis:
        muovi_temp = config_analysis['muovi_ellipse_63mm']['avg_return_temp']
        double_temp = config_analysis['double_u45mm']['avg_return_temp']
        benefit = muovi_temp - double_temp
        
        fig.suptitle(f'Comprehensive Collector Configuration Analysis\n'
                    f'MuoviEllipse 63mm vs Double U 45mm: {benefit:+.3f}°C benefit',
                    fontsize=16, fontweight='bold', y=0.95)
    else:
        fig.suptitle('Comprehensive Collector Configuration Analysis', 
                    fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, 'comprehensive_collector_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f"Comprehensive collector analysis saved to: {plot_path}")
    
    return plot_path

def test_gpu_utilization(model, data_loader, device):
    """Test GPU utilization with the current model and batch size."""
    
    if device != 'cuda':
        logging.info("GPU utilization test skipped - not using CUDA")
        return
    
    logging.info("Testing GPU utilization...")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model.eval()
    with torch.no_grad():
        # Run a few batches to test GPU usage
        for i, (batch_data, batch_targets) in enumerate(data_loader):
            if i >= 5:  # Test with 5 batches
                break
                
            batch_data = batch_data.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(batch_data)
            
            # Check memory usage after each batch
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            if i == 0:
                logging.info(f"Batch {i+1}: GPU memory allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
    
    # Final memory stats
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    utilization_percent = (peak_allocated / total_memory) * 100
    
    logging.info(f"GPU Utilization Test Results:")
    logging.info(f"  Peak memory used: {peak_allocated:.2f}GB / {total_memory:.1f}GB ({utilization_percent:.1f}%)")
    logging.info(f"  Batch size: {data_loader.batch_size}")
    logging.info(f"  Recommended: {'Good utilization' if utilization_percent > 50 else 'Consider increasing batch size'}")
    
    return peak_allocated, utilization_percent

def optimize_batch_size_for_gpu(model, sample_batch, device, max_batch_size=1024):
    """Find optimal batch size for GPU memory."""
    
    if device != 'cuda':
        return 64  # Default for CPU
    
    logging.info("Finding optimal batch size for GPU...")
    
    optimal_batch_size = 64
    seq_len, features = sample_batch.shape[1], sample_batch.shape[2]
    
    for batch_size in [64, 128, 256, 384, 512, 768, 1024]:
        if batch_size > max_batch_size:
            break
            
        try:
            torch.cuda.empty_cache()
            
            # Create test batch
            test_batch = torch.randn(batch_size, seq_len, features).to(device)
            test_targets = torch.randn(batch_size, 1).to(device)
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                outputs = model(test_batch)
            
            # Test backward pass
            model.train()
            outputs = model(test_batch)
            loss = nn.MSELoss()(outputs, test_targets)
            loss.backward()
            
            # If we reach here, this batch size works
            optimal_batch_size = batch_size
            allocated = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"  Batch size {batch_size}: OK ({allocated:.2f}GB)")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.info(f"  Batch size {batch_size}: Out of memory")
                break
            else:
                raise e
    
    logging.info(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size

def analyze_bhe_configurations(combined_data):
    """Analyze the different BHE configurations."""
    
    logging.info("Analyzing BHE configurations...")
    
    analysis = {}
    for bhe_type in combined_data['bhe_type'].unique():
        subset = combined_data[combined_data['bhe_type'] == bhe_type]
        
        analysis[bhe_type] = {
            'count': len(subset),
            'avg_return_temp': subset['return_temp'].mean(),
            'avg_supply_temp': subset['supply_temp'].mean(),
            'avg_temp_diff': (subset['return_temp'] - subset['supply_temp']).mean(),
            'avg_flow_rate': subset['flow_rate'].mean(),
            'avg_power': subset['power_kw'].mean()
        }
        
        logging.info(f"{bhe_type}: {len(subset)} records, "
                    f"avg return temp: {analysis[bhe_type]['avg_return_temp']:.2f}°C, "
                    f"avg temp diff: {analysis[bhe_type]['avg_temp_diff']:.2f}°C")
    
    return analysis

def main():
    """Main execution function."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting comprehensive CNN-LSTM analysis on device: {device}")
    logging.info(f"Starting comprehensive analysis on device: {device}")
    
    # Load all data sources
    print("Loading data from all sources...")
    logging.info("Loading data from all sources...")
    
    # Load complete field data (Single U45mm)
    print("  Loading complete field data...")
    complete_field_df = load_complete_field_data()
    complete_field_df = clean_bhe_data(complete_field_df, "complete_field")
    
    # Load Double U45mm research data
    print("  Loading Double U45mm research wells data...")
    double_u45mm_df = load_double_u45mm_research_data()
    if len(double_u45mm_df) > 0:
        double_u45mm_df = clean_bhe_data(double_u45mm_df, "double_u45mm_research")
    
    # Load MuoviEllipse data
    print("  Loading MuoviEllipse 63mm research data...")
    muovi_ellipse_df = load_muovi_ellipse_research_data()
    if len(muovi_ellipse_df) > 0:
        muovi_ellipse_df = clean_bhe_data(muovi_ellipse_df, "muovi_ellipse_research")
    
    # Find common timestamp range across all datasets
    datasets = [df for df in [complete_field_df, double_u45mm_df, muovi_ellipse_df] if len(df) > 0]
    
    if not datasets:
        raise ValueError("No valid datasets loaded")
    
    # Find overlapping timestamp range
    start_times = [df['Timestamp'].min() for df in datasets]
    end_times = [df['Timestamp'].max() for df in datasets]
    
    common_start = max(start_times)
    common_end = min(end_times)
    
    logging.info(f"Common timestamp range: {common_start} to {common_end}")
    
    # Filter all datasets to common timestamp range
    filtered_datasets = []
    for i, df in enumerate(datasets):
        original_len = len(df)
        filtered_df = df[(df['Timestamp'] >= common_start) & (df['Timestamp'] <= common_end)].copy()
        filtered_datasets.append(filtered_df)
        
        dataset_name = ['complete_field', 'double_u45mm', 'muovi_ellipse'][i] if i < 3 else f'dataset_{i}'
        logging.info(f"Filtered {dataset_name}: {original_len} -> {len(filtered_df)} records")
    
    # Combine filtered datasets
    combined_data = pd.concat(filtered_datasets, ignore_index=True)
    combined_data = combined_data.sort_values('Timestamp').reset_index(drop=True)
    
    logging.info(f"Combined dataset: {len(combined_data)} records from {len(filtered_datasets)} sources")
    logging.info(f"Time range: {combined_data['Timestamp'].min()} to {combined_data['Timestamp'].max()}")
    
    # Analyze BHE configurations
    config_analysis = analyze_bhe_configurations(combined_data)
    
    # Define features and target
    feature_cols = ['supply_temp', 'flow_rate', 'power_kw', 'bhe_type_encoded']
    target_col = 'return_temp'
    
    # Prepare data for training
    model_data = combined_data[feature_cols + [target_col, 'Timestamp', 'bhe_type']].dropna()
    
    logging.info(f"Model data: {len(model_data)} records with {len(feature_cols)} features")
    
    # Train/validation/test split (chronological)
    n = len(model_data)
    train_end = int(n * (1 - VAL_SPLIT - TEST_SPLIT))
    val_end = int(n * (1 - TEST_SPLIT))
    
    train_df = model_data.iloc[:train_end].copy()
    val_df = model_data.iloc[train_end:val_end].copy()
    test_df = model_data.iloc[val_end:].copy()
    
    logging.info(f"Data splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Create datasets
    train_dataset = ComprehensiveDataset(train_df, SEQ_LEN, PRED_HORIZON, feature_cols, target_col)
    val_dataset = ComprehensiveDataset(val_df, SEQ_LEN, PRED_HORIZON, feature_cols, target_col, 
                                     train_dataset.mean, train_dataset.std)
    test_dataset = ComprehensiveDataset(test_df, SEQ_LEN, PRED_HORIZON, feature_cols, target_col,
                                      train_dataset.mean, train_dataset.std)
    
    # Initialize model for batch size optimization
    temp_model = ComprehensiveCNNLSTM(
        input_features=len(feature_cols),
        conv_channels=CONV_CHANNELS,
        kernel_size=KERNEL_SIZE,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Optimize batch size for GPU if using CUDA
    if device == 'cuda':
        sample_batch, _ = next(iter(DataLoader(train_dataset, batch_size=1)))
        optimal_batch_size = optimize_batch_size_for_gpu(temp_model, sample_batch, device)
        # Use the optimal batch size or stick with configured one if it's smaller
        actual_batch_size = min(BATCH_SIZE, optimal_batch_size)
        logging.info(f"Using batch size: {actual_batch_size} (configured: {BATCH_SIZE}, optimal: {optimal_batch_size})")
    else:
        actual_batch_size = BATCH_SIZE
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=actual_batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True, persistent_workers=True)
    
    # Initialize final model
    model = ComprehensiveCNNLSTM(
        input_features=len(feature_cols),
        conv_channels=CONV_CHANNELS,
        kernel_size=KERNEL_SIZE,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Test GPU utilization before training
    if device == 'cuda':
        print("Testing GPU utilization...")
        peak_memory, utilization = test_gpu_utilization(model, train_loader, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model initialized: {total_params:,} parameters")
    logging.info(f"Model initialized with {total_params:,} parameters ({trainable_params:,} trainable)")
    logging.info(f"Features used: {feature_cols}")
    
    # Train model
    print("Starting model training...")
    logging.info("Starting model training...")
    model, history = train_model(model, train_loader, val_loader, EPOCHS, LR, device, PATIENCE)
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, "comprehensive_model.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")
    
    # Evaluate model
    print("Evaluating model performance...")
    logging.info("Evaluating model performance...")
    predictions, targets, mae, rmse = evaluate_model(model, test_loader, device)
    
    print(f"Test Performance: MAE={mae:.4f}°C, RMSE={rmse:.4f}°C")
    logging.info(f"Test Performance: MAE={mae:.4f}°C, RMSE={rmse:.4f}°C")
    
    # Create comprehensive visualization
    print("Creating comprehensive collector analysis...")
    create_comprehensive_collector_analysis(test_df, predictions, targets, feature_cols, config_analysis)
    
    # Save results
    results = {
        'model_performance': {
            'mae': float(mae),
            'rmse': float(rmse),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        },
        'training_history': {
            'train_losses': [float(x) for x in history['train_losses']],
            'val_losses': [float(x) for x in history['val_losses']]
        },
        'data_summary': {
            'total_records': len(combined_data),
            'train_records': len(train_df),
            'val_records': len(val_df), 
            'test_records': len(test_df),
            'features': feature_cols
        },
        'bhe_analysis': config_analysis
    }
    
    results_path = os.path.join(OUTPUT_DIR, "comprehensive_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("COMPREHENSIVE CNN-LSTM ANALYSIS SUMMARY")
    print("="*70)
    print(f"Model Performance:")
    print(f"  MAE: {mae:.4f}°C")
    print(f"  RMSE: {rmse:.4f}°C")
    print(f"  Parameters: {total_params:,}")
    print(f"\nBHE Configurations Analyzed:")
    
    for config, analysis in config_analysis.items():
        print(f"  {config}:")
        print(f"    Records: {analysis['count']}")
        print(f"    Mean return temp: {analysis['avg_return_temp']:.2f}°C")
        print(f"    Mean temp difference: {analysis['avg_temp_diff']:.3f}°C")
    
    print(f"\nFiles generated in {OUTPUT_DIR}:")
    print(f"  - comprehensive_model.pth (trained CNN-LSTM model)")
    print(f"  - comprehensive_results.json (detailed results)")
    print(f"  - comprehensive_collector_analysis.png (multi-panel visualization)")
    print(f"  - comprehensive_analysis.log (execution log)")
    print("="*70)

if __name__ == "__main__":
    main()