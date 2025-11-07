import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def create_muovi_ellipse_temperature_plot():
    """Create scatter plot showing actual temperature distribution for MuoviEllipse wells."""
    
    # Load MuoviEllipse data
    file_path = os.path.join('input', 'MeterOE401_Ellipse63.csv')
    
    print("Loading MuoviEllipse data...")
    df = pd.read_csv(file_path, encoding='latin1')  # Handle special characters
    
    # Clean column names and remove special characters
    df.columns = df.columns.str.strip()
    
    # Remove BOM character from column names
    df.columns = df.columns.str.replace('\ufeff', '')
    
    # Rename columns to remove special characters
    column_mapping = {}
    for col in df.columns:
        new_col = col.replace('°', 'deg').replace('³', '3').replace('�', 'deg')
        column_mapping[col] = new_col
    df = df.rename(columns=column_mapping)
    
    print("Available columns:", df.columns.tolist())
    
    # Use first column (timestamp) regardless of BOM issues
    timestamp_col = df.columns[0]
    
    # Convert timestamp to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='%d.%m.%Y %H:%M', errors='coerce')
    
    # Find temperature columns by pattern matching
    supply_col = [col for col in df.columns if 'T_supply' in col][0]
    return_col = [col for col in df.columns if 'T_return' in col][0]
    
    print(f"Using columns: '{supply_col}' and '{return_col}'")
    
    # Extract temperature columns
    t_supply = pd.to_numeric(df[supply_col], errors='coerce')
    t_return = pd.to_numeric(df[return_col], errors='coerce')
    
    # Remove NaN values
    valid_mask = ~(pd.isna(t_supply) | pd.isna(t_return))
    t_supply_clean = t_supply[valid_mask]
    t_return_clean = t_return[valid_mask]
    
    print(f"Data points: {len(t_supply_clean):,}")
    print(f"Supply temp range: {t_supply_clean.min():.1f}°C to {t_supply_clean.max():.1f}°C")
    print(f"Return temp range: {t_return_clean.min():.1f}°C to {t_return_clean.max():.1f}°C")
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    # Use alpha for transparency due to large number of points
    plt.scatter(t_supply_clean, t_return_clean, alpha=0.3, s=1, c='blue', label='Data points')
    
    # Add diagonal line for reference (supply = return)
    min_temp = min(t_supply_clean.min(), t_return_clean.min())
    max_temp = max(t_supply_clean.max(), t_return_clean.max())
    plt.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', alpha=0.7, label='Supply = Return')
    
    plt.xlabel('Supply Temperature [°C]', fontsize=12)
    plt.ylabel('Return Temperature [°C]', fontsize=12)
    plt.title('Temperature Distribution - MuoviEllipse Wells (OE401)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics text box
    stats_text = f"""Statistics:
Data points: {len(t_supply_clean):,}
Supply: {t_supply_clean.mean():.1f}°C ± {t_supply_clean.std():.1f}°C
Return: {t_return_clean.mean():.1f}°C ± {t_return_clean.std():.1f}°C
Temp diff: {(t_supply_clean - t_return_clean).mean():.1f}°C ± {(t_supply_clean - t_return_clean).std():.1f}°C"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join('output', 'muovi_ellipse_temperature_distribution.png')
    os.makedirs('output', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()
    
    return output_path

if __name__ == "__main__":
    create_muovi_ellipse_temperature_plot()