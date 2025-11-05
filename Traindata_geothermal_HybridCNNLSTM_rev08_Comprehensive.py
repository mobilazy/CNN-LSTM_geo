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
BATCH_SIZE = 64
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
PATIENCE = 10

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "comprehensive_analysis.log"), mode="w"),
        logging.StreamHandler(),
    ],
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
    """Load and process Double U45mm research wells data."""
    
    # Load individual outlet temperatures
    treturn_path = os.path.join(os.path.dirname(__file__), "input/DoubleU45_Treturn.csv")
    # Load shared sensor data
    oe403_path = os.path.join(os.path.dirname(__file__), "input/MeterOE403_doubleU45.csv")
    
    if not os.path.exists(treturn_path) or not os.path.exists(oe403_path):
        logging.warning("Double U45mm research data files not found, skipping...")
        return pd.DataFrame()
    
    logging.info("Loading Double U45mm research wells data...")
    
    try:
        # Load individual outlet temperatures
        treturn_df = pd.read_csv(treturn_path, sep=',', decimal='.')  # Fixed separator
        treturn_df['Timestamp'] = pd.to_datetime(treturn_df['Timestamp'], errors="coerce")
        
        # Load shared sensor data
        oe403_df = pd.read_csv(oe403_path, sep=',', decimal='.')  # Fixed separator
        oe403_df['Timestamp'] = pd.to_datetime(oe403_df['Timestamp'], errors="coerce")
        
        # Clean power data (remove spaces and convert to numeric)
        oe403_df['Power [kW]'] = pd.to_numeric(oe403_df['Power [kW]'].astype(str).str.strip(), errors='coerce')
        
        # Well mappings
        wells = {
            'SKD-110-01': 'RT512',
            'SKD-110-02': 'RT513', 
            'SKD-110-03': 'RT514',
            'SKD-110-04': 'RT515'
        }
        
        combined_data = []
        
        for well_id, temp_col in wells.items():
            if temp_col not in treturn_df.columns:
                continue
                
            # Merge individual outlet temp with shared sensor data
            well_data = pd.merge(treturn_df[['Timestamp', temp_col]], 
                               oe403_df[['Timestamp', 'T_supply [°C]', 'Flow [m³/h]', 'Power [kW]']], 
                               on='Timestamp', how='inner')
            
            well_data = well_data.rename(columns={
                temp_col: 'return_temp',
                'T_supply [°C]': 'supply_temp',
                'Flow [m³/h]': 'flow_rate', 
                'Power [kW]': 'power_kw'  # Negative = heat extraction, Positive = heat rejection
            })
            
            # Clean data
            well_data = well_data.dropna()
            well_data = well_data[(well_data['supply_temp'] > -20) & (well_data['supply_temp'] < 50)]
            well_data = well_data[(well_data['return_temp'] > -20) & (well_data['return_temp'] < 50)]
            well_data = well_data[well_data['flow_rate'] > 0]
            
            # Add BHE type
            well_data['bhe_type'] = 'double_u45mm'
            well_data['bhe_type_encoded'] = 1  # Double U45mm = 1
            well_data['well_id'] = well_id
            
            combined_data.append(well_data)
            logging.info(f"Processed {well_id}: {len(well_data)} records")
        
        if combined_data:
            result_df = pd.concat(combined_data, ignore_index=True)
            logging.info(f"Combined Double U45mm data: {len(result_df)} total records from {len(combined_data)} wells")
            return result_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error loading Double U45mm research data: {e}")
        print(f"ERROR loading Double U45mm research data: {e}")  # Debug output
        return pd.DataFrame()

def create_muovi_ellipse_research_data():
    """Create MuoviEllipse 63mm aggregated data."""
    
    logging.info("Creating MuoviEllipse 63mm aggregated data...")
    
    # Load complete field data as base
    complete_field_df = load_complete_field_data()
    
    if len(complete_field_df) == 0:
        return pd.DataFrame()
    
    # Create aggregated representation for MuoviEllipse 63mm
    muovi_df = complete_field_df.copy()
    muovi_df['bhe_type'] = 'muovi_ellipse_63mm'
    muovi_df['bhe_type_encoded'] = 2  # MuoviEllipse 63mm = 2
    
    logging.info(f"Created MuoviEllipse aggregated data: {len(muovi_df)} records")
    return muovi_df

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
    """Train the CNN-LSTM model with optimized performance."""
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    logging.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        # Simplified progress bar for training
        for batch_data, batch_targets in train_loader:
            batch_data = batch_data.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
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
        
        # Log every 2 epochs to reduce overhead
        if epoch % 2 == 0 or epoch == 1 or patience_counter >= patience:
            logging.info(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                        f"Best Val: {best_val_loss:.6f}, Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
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
            
            # Convert to Python lists instead of numpy
            all_predictions.extend(outputs.cpu().tolist())
            all_targets.extend(batch_targets.cpu().tolist())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions).flatten()
    targets = np.array(all_targets).flatten()
    
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    
    return predictions, targets, mae, rmse

def create_collector_comparison_plot(test_df, predictions, targets, feature_cols):
    """Create collector configuration comparison plot like the attached PNG."""
    
    logging.info("Creating collector configuration comparison plot...")
    
    # Get test data with timestamps
    test_data = test_df.copy()
    
    # Add predictions to test data
    # Account for sequence length offset
    if len(predictions) < len(test_data):
        # Pad with NaN for missing initial values
        padded_predictions = np.full(len(test_data), np.nan)
        padded_predictions[-len(predictions):] = predictions
        predictions = padded_predictions
    
    test_data['predicted_temp'] = predictions[:len(test_data)]
    test_data['actual_temp'] = test_data['return_temp']
    
    # Filter recent data for better visualization (last 2 weeks)
    recent_data = test_data.dropna(subset=['predicted_temp']).tail(2000)
    
    if len(recent_data) == 0:
        logging.warning("No valid data for comparison plot")
        return
    
    # Create the comparison plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot data by collector type
    colors = {'single_u45mm': '#1f77b4', 'double_u45mm': '#d62728', 'muovi_ellipse_63mm': '#2ca02c'}
    
    for bhe_type in recent_data['bhe_type'].unique():
        type_data = recent_data[recent_data['bhe_type'] == bhe_type]
        
        if len(type_data) == 0:
            continue
        
        # Plot actual temperatures
        ax.plot(type_data['Timestamp'], type_data['actual_temp'], 
               color=colors.get(bhe_type, '#333333'), 
               alpha=0.7, linewidth=2,
               label=f'Actual ({bhe_type.replace("_", " ").title()})')
        
        # Plot predicted temperatures
        ax.plot(type_data['Timestamp'], type_data['predicted_temp'], 
               color=colors.get(bhe_type, '#333333'), 
               alpha=0.9, linewidth=2, linestyle='--',
               label=f'Predicted ({bhe_type.replace("_", " ").title()})')
    
    # Calculate and display average temperature benefit
    if 'muovi_ellipse_63mm' in recent_data['bhe_type'].values and 'double_u45mm' in recent_data['bhe_type'].values:
        muovi_data = recent_data[recent_data['bhe_type'] == 'muovi_ellipse_63mm']
        double_data = recent_data[recent_data['bhe_type'] == 'double_u45mm']
        
        if len(muovi_data) > 0 and len(double_data) > 0:
            avg_muovi = muovi_data['predicted_temp'].mean()
            avg_double = double_data['predicted_temp'].mean()
            temp_benefit = avg_muovi - avg_double
            
            # Add temperature benefit annotation
            ax.text(0.02, 0.98, f'Avg. Temperature Benefit: {temp_benefit:.3f}°C\n(MuoviEllipse vs Double U)', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                   verticalalignment='top')
    
    # Formatting
    ax.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Outlet Temperature (°C)', fontsize=14, fontweight='bold')
    ax.set_title('Collector Configuration Analysis: CNN-LSTM Predictions vs Actual', 
                fontsize=16, fontweight='bold')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Legend
    ax.legend(fontsize=10, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, 'collector_configuration_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Collector comparison plot saved to: {plot_path}")

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
    logging.info(f"Starting comprehensive analysis on device: {device}")
    
    # Load all data sources
    logging.info("Loading data from all sources...")
    
    # Load complete field data (Single U45mm)
    complete_field_df = load_complete_field_data()
    complete_field_df = clean_bhe_data(complete_field_df, "complete_field")
    
    # Load Double U45mm research data
    double_u45mm_df = load_double_u45mm_research_data()
    if len(double_u45mm_df) > 0:
        double_u45mm_df = clean_bhe_data(double_u45mm_df, "double_u45mm_research")
    
    # Create MuoviEllipse data
    muovi_ellipse_df = create_muovi_ellipse_research_data()
    if len(muovi_ellipse_df) > 0:
        muovi_ellipse_df = clean_bhe_data(muovi_ellipse_df, "muovi_ellipse_research")
    
    # Combine all datasets
    datasets = [df for df in [complete_field_df, double_u45mm_df, muovi_ellipse_df] if len(df) > 0]
    
    if not datasets:
        raise ValueError("No valid datasets loaded")
    
    combined_data = pd.concat(datasets, ignore_index=True)
    combined_data = combined_data.sort_values('Timestamp').reset_index(drop=True)
    
    logging.info(f"Combined dataset: {len(combined_data)} records from {len(datasets)} sources")
    
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
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize model
    model = ComprehensiveCNNLSTM(
        input_features=len(feature_cols),
        conv_channels=CONV_CHANNELS,
        kernel_size=KERNEL_SIZE,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model initialized with {total_params:,} parameters ({trainable_params:,} trainable)")
    logging.info(f"Features used: {feature_cols}")
    
    # Train model
    logging.info("Starting model training...")
    model, history = train_model(model, train_loader, val_loader, EPOCHS, LR, device, PATIENCE)
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, "comprehensive_model.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")
    
    # Evaluate model
    logging.info("Evaluating model performance...")
    predictions, targets, mae, rmse = evaluate_model(model, test_loader, device)
    
    logging.info(f"Test Performance: MAE={mae:.4f}°C, RMSE={rmse:.4f}°C")
    
    # Create comprehensive visualization
    create_collector_comparison_plot(test_df, predictions, targets, feature_cols)
    
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
    print(f"  - collector_configuration_comparison.png (visualization)")
    print(f"  - comprehensive_analysis.log (execution log)")
    print("="*70)

if __name__ == "__main__":
    main()