import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_ROOT / "output" / "comprehensive_results.json"
OUTPUT_PATH = PROJECT_ROOT / "output" / "training_convergence.png"

with RESULTS_PATH.open("r", encoding="utf-8") as f:
    results = json.load(f)

train_losses = np.array(results["training_history"]["train_losses"])
val_losses = np.array(results["training_history"]["val_losses"])
epochs = np.arange(1, len(train_losses) + 1)

best_epoch = val_losses.argmin() + 1
best_val = val_losses.min()
patience_window = 16
# Find actual early stopping epoch (when validation stopped improving)
no_improve_epochs = 0
early_stop_epoch = len(epochs)
for i in range(best_epoch, len(epochs) + 1):
    if i > best_epoch:
        no_improve_epochs += 1
    if no_improve_epochs >= patience_window:
        early_stop_epoch = i
        break

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(epochs, train_losses, 'o-', label='Training Loss', 
         linewidth=2, markersize=4, color='#2E86AB', alpha=0.8)
ax1.plot(epochs, val_losses, 's-', label='Validation Loss', 
         linewidth=2, markersize=4, color='#A23B72', alpha=0.8)

ax1.axvline(best_epoch, color='#F18F01', linestyle='--', linewidth=2, 
            alpha=0.7, label=f'Best Val Loss (Epoch {best_epoch})')
ax1.scatter(best_epoch, best_val, color='#F18F01', s=150, 
            zorder=5, edgecolors='black', linewidths=1.5)

if early_stop_epoch <= len(epochs):
    ax1.axvline(early_stop_epoch, color='red', linestyle=':', linewidth=2, 
                alpha=0.7, label=f'Early Stop Trigger (Epoch {early_stop_epoch})')
else:
    ax1.text(len(epochs) * 0.98, train_losses[0] * 0.15, 
             'Training reached\nepoch limit', 
             fontsize=9, ha='right', style='italic',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax1.set_title('CNN-LSTM Training Convergence', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0.15, 1.0))
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, len(epochs) + 1)

early_stop_status = 'Not triggered' if early_stop_epoch > len(epochs) else f'Epoch {early_stop_epoch}'

textstr = '\n'.join([
    'Training Summary:',
    f'Epoch Limit: {len(epochs)}',
    f'Best Val: Epoch {best_epoch}',
    f'Early Stop: {early_stop_status}',
    f'Patience: {patience_window} epochs',
    f'',
    f'Initial Train Loss: {train_losses[0]:.2f}',
    f'Final Train Loss: {train_losses[-1]:.2f}',
    f'Reduction: {(1 - train_losses[-1]/train_losses[0])*100:.1f}%',
    f'',
    f'Best Val Loss: {best_val:.3f}'
])

ax1.text(40, 0.50 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]) + ax1.get_ylim()[0], 
         textstr, fontsize=9,
         verticalalignment='center', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))

loss_reduction = train_losses - train_losses[0]
val_reduction = val_losses - val_losses[0]

ax2.plot(epochs, loss_reduction, 'o-', label='Training Loss Reduction', 
         linewidth=2, markersize=4, color='#2E86AB', alpha=0.8)
ax2.plot(epochs, val_reduction, 's-', label='Validation Loss Reduction', 
         linewidth=2, markersize=4, color='#A23B72', alpha=0.8)

ax2.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.axvline(best_epoch, color='#F18F01', linestyle='--', linewidth=2, alpha=0.7)

ax2.set_title('Loss Reduction from Initial State', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss Change from Epoch 1', fontsize=12, fontweight='bold')
legend2 = ax2.legend(fontsize=10, loc='upper left')
legend2.set_bbox_to_anchor((4.0 / (len(epochs) + 1), (-1 - ax2.get_ylim()[0]) / (ax2.get_ylim()[1] - ax2.get_ylim()[0])))
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, len(epochs) + 1)

improvement_pct = (1 - train_losses[-1] / train_losses[0]) * 100
val_improvement_pct = (1 - val_losses[-1] / val_losses[0]) * 100

convergence_note = 'Model converged' if early_stop_epoch > len(epochs) else 'Early stopped'

improvement_text = '\n'.join([
    'Improvement Analysis:',
    f'Train: {improvement_pct:.1f}% reduction',
    f'Val: {val_improvement_pct:.1f}% reduction',
    f'',
    f'Status: {convergence_note}',
    f'Best epoch: {best_epoch}',
    f'Rapid phase: Epochs 1-{best_epoch}',
    f'Stable phase: Epochs {best_epoch}-{len(epochs)}'
])

legend_bbox = legend2.get_window_extent(fig.canvas.get_renderer())
legend_left_data = ax2.transData.inverted().transform(legend_bbox.get_points())[0, 0]

ax2.text(legend_left_data, 0.50 * (ax2.get_ylim()[1] - ax2.get_ylim()[0]) + ax2.get_ylim()[0], 
         improvement_text, fontsize=9,
         verticalalignment='center', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Training convergence plot saved to: {OUTPUT_PATH}")
print(f"\nKey Metrics:")
print(f"  Total epochs run: {len(epochs)}")
print(f"  Best validation epoch: {best_epoch}")
print(f"  Training improvement: {improvement_pct:.1f}%")
print(f"  Validation improvement: {val_improvement_pct:.1f}%")
print(f"  Final train-val gap: {abs(train_losses[-1] - val_losses[-1]):.4f}")