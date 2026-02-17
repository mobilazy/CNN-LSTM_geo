import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Load data
df = pd.read_csv('input/MeterOE403_doubleU45.csv', parse_dates=['Timestamp'], dayfirst=True)
df = df.sort_values('Timestamp')

# Clean numeric columns
numeric_cols = ['Power [kW]', 'T_supply [°C]', 'T_return [°C]', 'Flow [m³/h]']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')

# Select window
start_idx = 15000
n_timesteps = 72
df_window = df.iloc[start_idx:start_idx+n_timesteps].copy()
df_window['minutes'] = [(t - df_window['Timestamp'].iloc[0]).total_seconds() / 60 
                         for t in df_window['Timestamp']]

input_window_end = 48
prediction_step = 48

# Create layout: 3 stacked input plots left, 1 output plot right
fig = plt.figure(figsize=(12, 5.5))
gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1], height_ratios=[1, 1, 1],
                       hspace=0.15, wspace=0.3)

# LEFT: 3 STACKED INPUT FEATURE PLOTS
features_config = [
    ('T_supply [°C]', 'Supply temperature (°C)', '#1f77b4'),
    ('Flow [m³/h]', 'Flow rate (m³/h)', '#ff7f0e'),
    ('Power [kW]', 'Power (kW)', '#2ca02c')
]

ax_inputs = []
for idx, (col, label, color) in enumerate(features_config):
    ax = fig.add_subplot(gs[idx, 0])
    ax_inputs.append(ax)
    
    # Plot feature
    ax.plot(df_window['minutes'], df_window[col], color=color, linewidth=2, alpha=0.8)
    
    # Highlight input window (indices 0-47, 48 timesteps)
    ax.axvspan(0, df_window['minutes'].iloc[input_window_end-1], 
               alpha=0.15, color='#1f77b4', zorder=0)
    
    # Set y-axis with proper range
    ymin, ymax = df_window[col].min(), df_window[col].max()
    yrange = ymax - ymin
    ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
    
    # Set y-ticks with integer values for clarity
    y_tick_min = int(np.floor(ymin))
    y_tick_max = int(np.ceil(ymax))
    ax.set_yticks([y_tick_min, y_tick_max])
    ax.tick_params(axis='y', labelsize=9)
    
    ax.set_ylabel(label, fontsize=7 if idx == 0 else 9, fontweight='bold', labelpad=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, df_window['minutes'].max() + 5)
    
    # Only show x-label on bottom plot
    if idx < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')

# Align Y-axis labels across all input plots
fig.align_ylabels(ax_inputs)

# Add input window label on top plot (positioned at index 5)
ax_inputs[0].text(df_window['minutes'].iloc[5], 
                  ax_inputs[0].get_ylim()[0] + (ax_inputs[0].get_ylim()[1] - ax_inputs[0].get_ylim()[0]) * 0.1, 
                  'Input window\n48 timesteps (4 hours)', 
                  ha='left', fontsize=9, color='#1f77b4',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', 
                           edgecolor='#1f77b4', linewidth=1))


# Add sliding window arrows on bottom plot with vertical lines at tips
window_y_base = ax_inputs[2].get_ylim()[0]
window_y_range = ax_inputs[2].get_ylim()[1] - window_y_base
window_y_offset = window_y_range * 0.15

# Arrow positions (arrows point to prediction timestep)
arrow_configs = [
    (0, 48, 24, 'Step 1', 0.7, '-'),
    (1, 49, 25, 'Step 2', 0.5, '--'),
    (2, 50, 26, 'Step ...n', 0.3, '--')
]

for step_idx, (start_idx, end_idx, text_idx, label, alpha, linestyle) in enumerate(arrow_configs):
    y_pos = window_y_base - window_y_offset * (1 + step_idx * 2)
    
    # Arrow pointing to prediction timestep
    arrow = FancyArrowPatch((df_window['minutes'].iloc[start_idx], y_pos), 
                           (df_window['minutes'].iloc[end_idx], y_pos),
                           arrowstyle='->', mutation_scale=20, linewidth=2, 
                           linestyle=linestyle, color='#1f77b4', alpha=alpha)
    ax_inputs[2].add_patch(arrow)
    
    # Vertical line at arrow tip 
    for ax in ax_inputs:
        line_width = 2 if step_idx == 0 else 1.5
        ax.axvline(df_window['minutes'].iloc[end_idx - 1], 
                   color='#1f77b4', linewidth=line_width, linestyle=linestyle, 
                   alpha=alpha*0.8, zorder=1 if step_idx == 0 else 1)
    
    # Label
    ax_inputs[2].text(df_window['minutes'].iloc[text_idx], y_pos - window_y_offset*0.5, 
                      label, ha='center', fontsize=8, color='#1f77b4')

# Expand bottom plot y-limits to accommodate arrows
ax_inputs[2].set_ylim(window_y_base - window_y_offset*6.5, ax_inputs[2].get_ylim()[1])

# RIGHT: OUTPUT
ax_output = fig.add_subplot(gs[:, 1])
ax_output.plot(df_window['minutes'], df_window['T_return [°C]'], 
               color='#1f77b4', linewidth=2.5, marker='o', markersize=4,
               label='Return temperature')

# Highlight prediction points (3 consecutive predictions)
pred_times = [df_window['minutes'].iloc[48], df_window['minutes'].iloc[49], 
              df_window['minutes'].iloc[50]]
pred_vals = [df_window['T_return [°C]'].iloc[48], df_window['T_return [°C]'].iloc[49],
             df_window['T_return [°C]'].iloc[50]]

# Add vertical lines for predictions (matching arrow styles)
ax_output.axvline(pred_times[0], color='#1f77b4', linewidth=2, linestyle='-', alpha=0.6, zorder=2)  # Solid
ax_output.axvline(pred_times[1], color='#1f77b4', linewidth=1.5, linestyle='--', alpha=0.5, zorder=1)  # Dashed
ax_output.axvline(pred_times[2], color='#1f77b4', linewidth=1.5, linestyle='--', alpha=0.4, zorder=1)  # Dashed

# Mark prediction points
for i, (pt, pv) in enumerate(zip(pred_times, pred_vals)):
    alpha_val = 1.0 - i*0.25
    ax_output.scatter(pt, pv, color='#ff7f0e', s=150, zorder=5, 
                     marker='*', edgecolor='#d62728', linewidth=2, alpha=alpha_val)

# Mark input region (indices 0-47)
ax_output.axvspan(0, df_window['minutes'].iloc[input_window_end-1], 
                  alpha=0.1, color='#1f77b4', zorder=0)

# Add prediction label
ax_output.text(pred_times[0], pred_vals[0] + 0.15, 'Predictions', 
               ha='center', fontsize=9, color='#d62728',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffcccc', 
                        edgecolor='#d62728', linewidth=2))

ax_output.set_ylabel('Return temperature (°C)', fontsize=11, fontweight='bold')
ax_output.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
ax_output.grid(True, alpha=0.3)
ax_output.set_xlim(ax_inputs[0].get_xlim())

# Add legend in bottom left corner
ax_output.legend(loc='lower left', fontsize=9, framealpha=0.95)

# Main title
fig.suptitle('CNN-LSTM model: input window to output prediction', 
             fontsize=13, fontweight='bold', y=0.98)

plt.savefig('output/model_input_output_window.png', dpi=300, bbox_inches='tight')
print("Figure saved to output/model_input_output_window.png")
plt.show()
