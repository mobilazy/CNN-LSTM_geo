"""Compare OE403 supply/return temperatures with individual sensor readings."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Load OE403 data
oe403 = pd.read_csv('input/MeterOE403_doubleU45.csv')
oe403.columns = [c.strip() for c in oe403.columns]
oe403['Timestamp'] = pd.to_datetime(oe403['Timestamp'], format='%d.%m.%Y %H:%M')
oe403['T_supply [°C]'] = pd.to_numeric(oe403['T_supply [°C]'], errors='coerce')
oe403['T_return [°C]'] = pd.to_numeric(oe403['T_return [°C]'].astype(str).str.strip(), errors='coerce')

# Load sensor data
sensors = pd.read_csv('input/DoubleU45_Treturn.csv')
sensors.columns = [c.strip() for c in sensors.columns]
sensors['Timestamp'] = pd.to_datetime(sensors['Timestamp'], format='%d.%m.%Y %H:%M')
for col in ['737.003-RT512 [°C]', '737.003-RT513 [°C]', '737.003-RT514 [°C]', '737.003-RT515 [°C]']:
    sensors[col] = pd.to_numeric(sensors[col], errors='coerce')

# Merge datasets
merged = sensors.merge(oe403[['Timestamp', 'T_supply [°C]', 'T_return [°C]']], on='Timestamp', how='inner')

# Focus on September period
start = pd.Timestamp('2025-09-08')
end = pd.Timestamp('2025-09-18')
mask = (merged['Timestamp'] >= start) & (merged['Timestamp'] <= end)
plot_data = merged[mask]

# Calculate average of healthy sensors
healthy = ['737.003-RT512 [°C]', '737.003-RT513 [°C]', '737.003-RT515 [°C]']
plot_data['Healthy_Avg'] = plot_data[healthy].mean(axis=1)

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Top plot: Individual sensors vs OE403 return
ax1.plot(plot_data['Timestamp'], plot_data['737.003-RT512 [°C]'], 
         label='RT512 (sensor)', color='#1B9E77', linewidth=1.2, alpha=0.8)
ax1.plot(plot_data['Timestamp'], plot_data['737.003-RT513 [°C]'], 
         label='RT513 (sensor)', color='#D95F02', linewidth=1.2, alpha=0.8)
ax1.plot(plot_data['Timestamp'], plot_data['737.003-RT515 [°C]'], 
         label='RT515 (sensor)', color='#7570B3', linewidth=1.2, alpha=0.8)
ax1.plot(plot_data['Timestamp'], plot_data['T_return [°C]'], 
         label='OE403 return (total)', color='#E41A1C', linewidth=2.0, linestyle='--', alpha=0.9)
ax1.plot(plot_data['Timestamp'], plot_data['Healthy_Avg'], 
         label='Healthy sensors avg', color='#000000', linewidth=1.5, linestyle=':', alpha=0.7)

ax1.set_ylabel('Temperature [°C]', fontsize=12)
ax1.set_title('Return Temperature Comparison: Individual Sensors vs OE403 Total', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Bottom plot: Supply from OE403 vs return sensors
ax2.plot(plot_data['Timestamp'], plot_data['T_supply [°C]'], 
         label='OE403 supply', color='#377EB8', linewidth=2.0, alpha=0.9)
ax2.plot(plot_data['Timestamp'], plot_data['Healthy_Avg'], 
         label='Healthy sensors avg (return)', color='#000000', linewidth=1.5, linestyle=':', alpha=0.7)
ax2.plot(plot_data['Timestamp'], plot_data['T_return [°C]'], 
         label='OE403 return', color='#E41A1C', linewidth=1.5, linestyle='--', alpha=0.7)

ax2.set_ylabel('Temperature [°C]', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_title('Supply vs Return Temperature Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')

fig.tight_layout()
output_path = Path('output/oe403_sensor_comparison.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Comparison plot saved to: {output_path}")

# Print statistics
print("\nTemperature Statistics (Sept 8-18, 2025):")
print(f"\nOE403 Supply:        {plot_data['T_supply [°C]'].mean():.2f}°C (std: {plot_data['T_supply [°C]'].std():.2f})")
print(f"OE403 Return:        {plot_data['T_return [°C]'].mean():.2f}°C (std: {plot_data['T_return [°C]'].std():.2f})")
print(f"Healthy sensors avg: {plot_data['Healthy_Avg'].mean():.2f}°C (std: {plot_data['Healthy_Avg'].std():.2f})")
print(f"\nDifference (OE403 return - Healthy avg): {(plot_data['T_return [°C]'] - plot_data['Healthy_Avg']).mean():.2f}°C")
print(f"Difference (OE403 supply - Healthy avg): {(plot_data['T_supply [°C]'] - plot_data['Healthy_Avg']).mean():.2f}°C")

plt.close()
