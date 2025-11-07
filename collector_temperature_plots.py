import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

def load_and_clean_data(file_path, collector_name):
    """Load and clean temperature data from CSV file."""
    print(f"Loading {collector_name} data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            print(f"Failed to load {file_path}")
            return None, None, None, None
    
    # Clean column names
    df.columns = df.columns.str.strip().str.replace('\ufeff', '').str.replace('°', 'deg').str.replace('³', '3').str.replace('�', 'deg')
    
    print(f"Available columns: {df.columns.tolist()}")
    
    # Find temperature columns
    supply_cols = [col for col in df.columns if 'T_supply' in col or 'supply' in col.lower()]
    return_cols = [col for col in df.columns if 'T_return' in col or 'return' in col.lower()]
    
    if not supply_cols or not return_cols:
        print(f"Warning: Could not find temperature columns for {collector_name}")
        return None, None, None, None
    
    supply_col = supply_cols[0]
    return_col = return_cols[0]
    
    print(f"Using columns: '{supply_col}' and '{return_col}'")
    
    # Extract temperature data
    t_supply = pd.to_numeric(df[supply_col], errors='coerce')
    t_return = pd.to_numeric(df[return_col], errors='coerce')
    
    # Handle timestamp column (first column)
    timestamp_col = df.columns[0]
    try:
        timestamps = pd.to_datetime(df[timestamp_col], format='%d.%m.%Y %H:%M', errors='coerce')
    except:
        timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
    
    # Remove NaN values (keeping timestamps aligned)
    valid_mask = ~(pd.isna(t_supply) | pd.isna(t_return) | pd.isna(timestamps))
    t_supply_clean = t_supply[valid_mask]
    t_return_clean = t_return[valid_mask]
    timestamps_clean = timestamps[valid_mask]
    
    print(f"Data points: {len(t_supply_clean):,}")
    print(f"Supply temp range: {t_supply_clean.min():.1f}°C to {t_supply_clean.max():.1f}°C")
    print(f"Return temp range: {t_return_clean.min():.1f}°C to {t_return_clean.max():.1f}°C")
    
    return t_supply_clean, t_return_clean, len(t_supply_clean), timestamps_clean

def create_temperature_plot(t_supply, t_return, collector_name, color='blue'):
    """Create temperature distribution scatter plot."""
    
    # Sample data if too many points for visualization
    if len(t_supply) > 50000:
        sample_indices = np.random.choice(len(t_supply), 50000, replace=False)
        t_supply_plot = t_supply.iloc[sample_indices]
        t_return_plot = t_return.iloc[sample_indices]
        print(f"Sampling {len(sample_indices):,} points for visualization")
    else:
        t_supply_plot = t_supply
        t_return_plot = t_return
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with transparency
    plt.scatter(t_supply_plot, t_return_plot, alpha=0.5, s=1, c=color, label='Data points')
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(t_supply, t_return)
    min_temp = min(t_supply.min(), t_return.min())
    max_temp = max(t_supply.max(), t_return.max())
    
    # Generate regression line points
    x_line = np.array([min_temp, max_temp])
    y_line = slope * x_line + intercept
    
    plt.plot(x_line, y_line, 'r-', alpha=0.8, linewidth=2, 
             label=f'Regression (R² = {r_value**2:.3f})')
    
    plt.xlabel('Supply Temperature [°C]', fontsize=12)
    plt.ylabel('Return Temperature [°C]', fontsize=12)
    plt.title(f'Temperature Distribution - {collector_name} Wells', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics text box
    temp_diff = t_supply - t_return
    stats_text = f"""Statistics:
Data points: {len(t_supply):,}
Supply: {t_supply.mean():.1f}°C ± {t_supply.std():.1f}°C
Return: {t_return.mean():.1f}°C ± {t_return.std():.1f}°C
Temp diff: {temp_diff.mean():.1f}°C ± {temp_diff.std():.1f}°C"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{collector_name.lower().replace(' ', '_')}_temperature_distribution.png"
    output_path = os.path.join('output', filename)
    os.makedirs('output', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.close()  # Close the figure instead of showing it
    return output_path

def create_time_series_plot(data_dict):
    """Create a time series plot showing temperature difference over time for all collectors."""
    plt.figure(figsize=(16, 8))
    
    colors = ['#ff7f0e', '#1f77b4', '#d62728']  # Orange, Blue, Red
    collector_labels = {
        'MuoviEllipse (OE402)': 'MuoviEllipse 63mm (Research)',
        'SingleU45 (OE401)': 'Single U45mm (Complete Field)',
        'DoubleU45 (OE403)': 'Double U45mm (Research)'
    }
    
    # Define plotting order to put orange behind blue and red
    plot_order = ['MuoviEllipse (OE402)', 'SingleU45 (OE401)', 'DoubleU45 (OE403)']
    
    for i, collector_name in enumerate(plot_order):
        if collector_name not in data_dict:
            continue
            
        t_supply, t_return, count, timestamps = data_dict[collector_name]
        if t_supply is None:
            continue
            
        # Calculate temperature difference
        temp_diff = t_supply - t_return
        
        # Apply moving average smoothing (50-point window)
        window_size = 50
        temp_diff_smooth = temp_diff.rolling(window=window_size, center=True, min_periods=1).mean()
        
        # Use the correct label
        label = collector_labels.get(collector_name, collector_name)
        color = colors[i % len(colors)]
        
        # Plot only smoothed temperature difference over time
        plt.plot(timestamps, temp_diff_smooth, color=color, alpha=0.9, linewidth=2, label=label)
    
    # Add horizontal reference line at 0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Temperature Difference (°C)', fontsize=12)
    plt.title('BHE Temperature Response by Collector Configuration (Smoothed)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join('output', 'bhe_temperature_response_time_series_smooth.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Smoothed time series plot saved to: {output_path}")
    
    plt.close()
    return output_path

def create_comparison_plot(data_dict):
    """Create a comparison plot with all collectors."""
    plt.figure(figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (name, (t_supply, t_return, count)) in enumerate(data_dict.items()):
        if t_supply is None:
            continue
            
        # Sample data for comparison plot
        if len(t_supply) > 10000:
            sample_indices = np.random.choice(len(t_supply), 10000, replace=False)
            t_supply_plot = t_supply.iloc[sample_indices]
            t_return_plot = t_return.iloc[sample_indices]
        else:
            t_supply_plot = t_supply
            t_return_plot = t_return
        
        plt.scatter(t_supply_plot, t_return_plot, alpha=0.6, s=1, 
                   c=colors[i % len(colors)], label=f'{name} ({count:,} points)')
    
    # Add diagonal reference line
    plt.plot([0, 20], [0, 20], 'k--', alpha=0.5, label='Supply = Return')
    
    plt.xlabel('Supply Temperature [°C]', fontsize=12)
    plt.ylabel('Return Temperature [°C]', fontsize=12)
    plt.title('Temperature Distribution Comparison - All Collectors', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save comparison plot
    output_path = os.path.join('output', 'all_collectors_temperature_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    
    plt.show()
    return output_path

def main():
    """Main function to create all temperature distribution plots."""
    
    # Define data files and their corresponding collector names
    data_files = {
        'MuoviEllipse (OE402)': 'input/MeterOE402_Ellipse63.csv',
        'DoubleU45 (OE403)': 'input/MeterOE403_doubleU45.csv',
        'SingleU45 (OE401)': 'input/MeterOE401_singleU45.csv'
    }
    
    # Store data for time series plot
    all_data = {}
    
    # Create individual plots for each collector
    for collector_name, file_path in data_files.items():
        if os.path.exists(file_path):
            t_supply, t_return, count, timestamps = load_and_clean_data(file_path, collector_name)
            
            if t_supply is not None and t_return is not None:
                # Store for time series plot
                all_data[collector_name] = (t_supply, t_return, count, timestamps)
                
                # Create individual plot
                color = 'blue' if 'Ellipse' in collector_name else ('red' if 'DoubleU45' in collector_name else 'green')
                create_temperature_plot(t_supply, t_return, collector_name, color)
                print(f"Completed plot for {collector_name}\n")
            else:
                print(f"Failed to process {collector_name}\n")
        else:
            print(f"File not found: {file_path}\n")
    
    # Create time series plot
    if all_data:
        print("Creating time series plot...")
        create_time_series_plot(all_data)
    
    print("All plots completed!")

if __name__ == "__main__":
    main()