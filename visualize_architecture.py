import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def draw_cnn_lstm_architecture():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    color_input = '#E8F4F8'
    color_conv = '#B8E6F0'
    color_lstm = '#7EC8E3'
    color_fc = '#4A90A4'
    color_output = '#2E5266'
    color_text = '#1a1a1a'
    
    input_box = FancyBboxPatch((0.5, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor=color_input, linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 8.2, 'Input Layer', ha='center', va='center', fontsize=11, weight='bold', color=color_text)
    ax.text(1.5, 7.7, '[batch, 48, 4]', ha='center', va='center', fontsize=9, color=color_text)
    ax.text(1.5, 7.3, '48 timesteps', ha='center', va='center', fontsize=8, style='italic', color=color_text)
    
    conv1_box = FancyBboxPatch((3.5, 6.8), 2.2, 1.9, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor=color_conv, linewidth=2)
    ax.add_patch(conv1_box)
    ax.text(4.6, 8.3, 'Conv Block 1', ha='center', va='center', fontsize=11, weight='bold', color=color_text)
    ax.text(4.6, 7.9, 'Conv1D: 4→32', ha='center', va='center', fontsize=9, color=color_text)
    ax.text(4.6, 7.5, 'Kernel=3, Pad=1', ha='center', va='center', fontsize=8, color=color_text)
    ax.text(4.6, 7.2, 'BatchNorm + ReLU', ha='center', va='center', fontsize=8, color=color_text)
    ax.text(4.6, 6.9, 'Dropout=0.1', ha='center', va='center', fontsize=8, color=color_text)
    
    conv2_box = FancyBboxPatch((6.5, 6.8), 2.2, 1.9, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor=color_conv, linewidth=2)
    ax.add_patch(conv2_box)
    ax.text(7.6, 8.3, 'Conv Block 2', ha='center', va='center', fontsize=11, weight='bold', color=color_text)
    ax.text(7.6, 7.9, 'Conv1D: 32→64', ha='center', va='center', fontsize=9, color=color_text)
    ax.text(7.6, 7.5, 'Kernel=3, Pad=1', ha='center', va='center', fontsize=8, color=color_text)
    ax.text(7.6, 7.2, 'BatchNorm + ReLU', ha='center', va='center', fontsize=8, color=color_text)
    ax.text(7.6, 6.9, 'Dropout=0.1', ha='center', va='center', fontsize=8, color=color_text)
    
    lstm_box = FancyBboxPatch((4.5, 4.5), 4, 1.8, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=color_lstm, linewidth=2)
    ax.add_patch(lstm_box)
    ax.text(6.5, 5.9, 'LSTM Network', ha='center', va='center', fontsize=11, weight='bold', color=color_text)
    ax.text(6.5, 5.5, '2-layer Unidirectional', ha='center', va='center', fontsize=9, color=color_text)
    ax.text(6.5, 5.1, '64 hidden units per layer', ha='center', va='center', fontsize=9, color=color_text)
    ax.text(6.5, 4.7, 'Dropout=0.1 between layers', ha='center', va='center', fontsize=8, color=color_text)
    
    fc1_box = FancyBboxPatch((5, 2.8), 3, 1.2, boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor=color_fc, linewidth=2)
    ax.add_patch(fc1_box)
    ax.text(6.5, 3.7, 'FC Layer 1', ha='center', va='center', fontsize=10, weight='bold', color='white')
    ax.text(6.5, 3.3, 'Linear: 64→32', ha='center', va='center', fontsize=9, color='white')
    ax.text(6.5, 2.95, 'ReLU + Dropout=0.1', ha='center', va='center', fontsize=8, color='white')
    
    fc2_box = FancyBboxPatch((5.3, 1.3), 2.4, 1, boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor=color_fc, linewidth=2)
    ax.add_patch(fc2_box)
    ax.text(6.5, 2, 'FC Layer 2', ha='center', va='center', fontsize=10, weight='bold', color='white')
    ax.text(6.5, 1.65, 'Linear: 32→1', ha='center', va='center', fontsize=9, color='white')
    
    output_box = FancyBboxPatch((5.8, 0.2), 1.4, 0.7, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=color_output, linewidth=2)
    ax.add_patch(output_box)
    ax.text(6.5, 0.55, 'Temperature', ha='center', va='center', fontsize=10, weight='bold', color='white')
    
    arrow_props = dict(arrowstyle='->', lw=2.5, color='#333333')
    
    ax.annotate('', xy=(3.5, 7.75), xytext=(2.5, 7.75), arrowprops=arrow_props)
    ax.annotate('', xy=(6.5, 7.75), xytext=(5.7, 7.75), arrowprops=arrow_props)
    ax.annotate('', xy=(6.5, 6.8), xytext=(6.5, 6.3), arrowprops=arrow_props)
    ax.annotate('', xy=(6.5, 4.5), xytext=(6.5, 4.0), arrowprops=arrow_props)
    ax.annotate('', xy=(6.5, 2.8), xytext=(6.5, 2.3), arrowprops=arrow_props)
    ax.annotate('', xy=(6.5, 1.3), xytext=(6.5, 0.9), arrowprops=arrow_props)
    
    ax.text(2.9, 8.5, '[48, 4]', ha='center', va='center', fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(6.1, 8.5, '[48, 32]', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(9.2, 7.75, '[48, 64]', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(9.0, 5.4, '[64]', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(8.3, 3.4, '[32]', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(8.1, 1.8, '[1]', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    param_box = FancyBboxPatch((10, 7.5), 3.5, 1.2, boxstyle="round,pad=0.1",
                               edgecolor='#666666', facecolor='#f5f5f5', linewidth=1.5)
    ax.add_patch(param_box)
    ax.text(11.75, 8.4, 'Total Parameters', ha='center', va='center', 
            fontsize=10, weight='bold', color=color_text)
    ax.text(11.75, 7.9, '75,489', ha='center', va='center', 
            fontsize=12, weight='bold', color='#d9534f')
    
    feature_box = FancyBboxPatch((10, 5.2), 3.5, 1.7, boxstyle="round,pad=0.1",
                                 edgecolor='#666666', facecolor='#f5f5f5', linewidth=1.5)
    ax.add_patch(feature_box)
    ax.text(11.75, 6.7, 'Input Features', ha='center', va='center', 
            fontsize=10, weight='bold', color=color_text)
    ax.text(11.75, 6.3, '1. Supply Temp', ha='center', va='center', fontsize=8, color=color_text)
    ax.text(11.75, 6.0, '2. Flow Rate', ha='center', va='center', fontsize=8, color=color_text)
    ax.text(11.75, 5.7, '3. Power', ha='center', va='center', fontsize=8, color=color_text)
    ax.text(11.75, 5.4, '4. BHE Type', ha='center', va='center', fontsize=8, color=color_text)
    
    ax.text(7, 9.7, 'CNN-LSTM Architecture for Geothermal Temperature Prediction', 
            ha='center', va='center', fontsize=14, weight='bold', color=color_text)
    
    plt.tight_layout()
    plt.savefig('c:\\Users\\H259507\\CNN-LSTM_geo\\output\\architecture_diagram.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("Architecture diagram saved to output/architecture_diagram.png")
    plt.close()

if __name__ == "__main__":
    draw_cnn_lstm_architecture()