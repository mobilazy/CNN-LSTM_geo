# CNN-LSTM Model for Geothermal Borehole Heat Exchanger Analysis

## Introduction

This project implements a comprehensive CNN-LSTM deep learning model for predicting return temperatures in geothermal borehole heat exchanger (BHE) systems. The model analyzes three different collector configurations at the University of Stavanger geothermal facility, combining convolutional neural networks for feature extraction with LSTM networks for temporal sequence modeling. The system processes real-world sensor data from 120 production wells and 8 research wells operating at 300m depth.

## System Overview

### Geothermal Installation
The geothermal system at University of Stavanger consists of:
- **120 production boreholes** with Single U-tube 45mm configuration (complete field)
- **4 research boreholes** with Double U-tube 45mm configuration (SKD-110-01 to 04)
- **4 research boreholes** with MuoviEllipse 63mm configuration (SKD-110-05 to 08)
- All boreholes reach 300m depth with HX24 energy brine as heat transfer fluid
- Flow rate: 0.9 L/s per borehole, total system capacity: 1350 kW

### Data Collection
Sensor measurements collected at 5-minute intervals from three energy meters:
- **OE401**: Complete field data (120 Single U-45mm wells aggregated)
- **OE402**: MuoviEllipse 63mm research wells (4 wells aggregated)
- **OE403**: Double U-45mm research wells (4 wells aggregated)

Time period: January through November 2025, yielding approximately 95,000 measurement cycles per configuration after quality filtering.

## Model Architecture

The ComprehensiveCNNLSTM model implements a hybrid architecture for time-series regression:

### Architecture Components

1. **Convolutional Feature Extraction**
   - Two 1D convolutional layers with progressive channel expansion (32 → 64)
   - Kernel size: 3 with padding to preserve sequence length
   - ReLU activation, batch normalization, and dropout (0.1) after each layer
   - Extracts hierarchical temporal patterns while maintaining full sequence context

2. **LSTM Temporal Modeling**
   - 2-layer bidirectional LSTM with 64 hidden units per layer
   - Dropout of 0.1 between LSTM layers
   - Processes 48-timestep sequences (4 hours at 5-minute intervals)
   - Captures long-term dependencies in thermal dynamics

3. **Output Layer**
   - Fully connected linear layer mapping from 64 to 1 dimension
   - Predicts single-step ahead return temperature (regression task)

### Model Hyperparameters
- **Input features**: 4 (supply_temp, flow_rate, power_kw, bhe_type_encoded)
- **Sequence length**: 48 time steps (4 hours)
- **CNN channels**: [32, 64]
- **LSTM hidden size**: 64
- **LSTM layers**: 2
- **Dropout**: 0.1
- **Total parameters**: 75,489 trainable parameters

## Data Processing Pipeline

### Input Data Sources
Located in `input/` directory:
- `MeterOE401_singleU45.csv` - Complete field data (120 Single U-45mm wells)
- `MeterOE402_Ellipse63.csv` - MuoviEllipse 63mm research wells
- `MeterOE403_doubleU45.csv` - Double U-45mm research wells

### Eight-Stage Data Cleaning Process

1. **Numeric Conversion**: Raw CSV data converted to floating-point with original sensor precision
2. **Physical Constraint Validation**: 
   - Temperatures: -10°C to 50°C
   - Power: -500 to 500 kW per well
   - Flow rates: positive, below 100 m³/h per well
3. **Sensor Anomaly Detection**: Flag sequences with >18 identical consecutive readings (90 minutes)
4. **Median Filtering**: 
   - 3-point (15-minute) for temperatures
   - 5-point (25-minute) for power and flow rate
5. **Gap Interpolation**: Linear interpolation for gaps up to 4 readings (20 minutes)
6. **Physics Validation**: Verify temperature-power consistency for heating/cooling modes
7. **Missing Data Threshold**: Require 80% valid measurements (complete field) or 50% (research wells)
8. **Final Quality Assurance**: Remove invalid entries, verify time ordering

### Dataset Statistics
- **Raw dataset**: ~315,000 combined measurements
- **After cleaning**: 190,732 records with timestamp alignment
- **Training set**: 171,542 sequences
- **Validation set**: 4,943 sequences (7-day window)
- **Test set**: 14,247 sequences (21-day forecast window)

## Feature Engineering

### Input Features (4 dimensions)
1. **supply_temp** (°C): Measured temperature at heat pump outlet before entering borehole field
2. **flow_rate** (m³/h): Volumetric flow rate normalized per well
   - Complete field: divided by 120 wells
   - Research sections: divided by 4 wells
3. **power_kw** (kW): Thermal energy transfer rate per well
   - Positive values: heat extraction (heating mode)
   - Negative values: heat rejection (cooling mode)
4. **bhe_type_encoded**: Categorical encoding of collector configuration
   - 0: Single U45mm
   - 1: Double U45mm
   - 2: MuoviEllipse 63mm

### Target Variable
- **return_temp** (°C): Temperature measured after fluid exits borehole field

### Temporal Sequence Generation
- 48-timestep sliding window (4-hour history at 5-minute intervals)
- Single-step ahead prediction
- Overlapping sequences with single-step advancement

## Training Configuration

### Hyperparameters
- **Epochs**: 50
- **Batch size**: 1024 (optimized for GPU utilization)
- **Learning rate**: 0.001
- **Optimizer**: Adam with ReduceLROnPlateau scheduler
  - Factor: 0.5
  - Patience: 16 epochs
- **Loss function**: Mean Squared Error (MSE)
- **Early stopping patience**: 16 epochs

### Training Features
- Mixed precision training (AMP) for CUDA devices
- GPU memory monitoring and optimization
- Gradient clipping for stability
- Best model checkpoint saving based on validation loss

### Temporal Data Splitting
- **Training period**: January through early October 2025
- **Validation window**: 7 days in mid-October (held out for hyperparameter tuning)
- **Test period**: 21-day forecast window (late October through early November)
  - Single U45mm: 4,390 sequences
  - Double U45mm: 4,999 sequences
  - MuoviEllipse 63mm: 4,810 sequences

## Performance Metrics

### Evaluation Metrics
The model is evaluated using regression metrics for temperature prediction:

- **MAE (Mean Absolute Error)**: Average of absolute differences between predicted and actual temperatures
  ```
  MAE = (1/n) * Σ|y_i - ŷ_i|
  ```

- **RMSE (Root Mean Squared Error)**: Square root of average squared differences (penalizes larger errors)
  ```
  RMSE = √[(1/n) * Σ(y_i - ŷ_i)²]
  ```

### Model Performance by Collector Type
Performance varies by BHE configuration based on operational characteristics and data quality:

- **Single U45mm (Complete Field)**: Aggregated data from 120 wells provides statistical smoothing
- **Double U45mm (Research)**: Individual well monitoring with 4-well aggregation
- **MuoviEllipse 63mm (Research)**: Advanced elliptical design with mixed SDR materials

Performance metrics are calculated on the 21-day forecast window and reported separately for each collector configuration.

## Output Visualizations

All visualizations are saved to the `output/` directory at 300 DPI resolution.

### Generated Plots

1. **comprehensive_collector_analysis.png**
   - Per-collector time-series comparison showing training history and forecast window
   - Blends 21 days of training data with 21-day forecast predictions
   - Vertical line marks transition from training to prediction period
   - Separate subplots for each BHE configuration
   - Summary table with MAE, RMSE, and sample counts

2. **model_performance_comparison.png**
   - Bar charts comparing MAE and RMSE across collector types
   - Side-by-side comparison for direct performance assessment
   - Color-coded by collector configuration

3. **training_convergence.png** (via plot_training_convergence.py)
   - Training and validation loss curves over epochs
   - Visualizes model convergence and potential overfitting

4. **architecture_diagram.png** (via visualize_architecture.py)
   - Visual representation of CNN-LSTM model architecture
   - Shows layer dimensions and data flow

5. **rt514_sensor_recovery.png** (via sensor_recovery_rt514.py)
   - Demonstrates sensor malfunction detection and recovery
   - Compares actual sensor readings with CNN-LSTM predictions
   - Useful for maintenance and quality assurance

## Model and Data Storage

### Output Directory Structure
All outputs are saved to `output/` directory:

- **comprehensive_model.pth**: Trained model state dictionary (PyTorch format)
- **comprehensive_results.json**: Training metrics, hyperparameters, and performance statistics
- **comprehensive_analysis.log**: Detailed execution log with timestamps
- **Visualization files**: PNG images at 300 DPI resolution

### Model Checkpoint Format
The saved model includes:
- Model state dictionary with all learned weights
- Architecture configuration (layers, channels, dropout rates)
- Training statistics (best validation loss, epoch count)
- Feature normalization parameters (mean and std for each input)

## Usage

### Running the Main Training Script

```bash
python Traindata_geothermal_HybridCNNLSTM_rev10_Fixed.py
```

This script executes the complete workflow:
1. Loads and cleans data from all three collector configurations
2. Combines datasets with proper BHE type encoding
3. Splits data temporally (training/validation/test)
4. Trains the CNN-LSTM model with GPU optimization
5. Evaluates performance on 21-day forecast window
6. Generates comprehensive visualizations
7. Saves model checkpoint and results to `output/`

### Additional Analysis Scripts

**Temperature Distribution Analysis:**
```bash
python temperature_distribution_plot.py
```

**Training Convergence Visualization:**
```bash
python plot_training_convergence.py
```

**Architecture Visualization:**
```bash
python visualize_architecture.py
```

**Sensor Recovery Analysis (RT514):**
```bash
python sensor_recovery_rt514.py
```

**Confusion Matrix Generation:**
```bash
python confusionMatrix.py
```
## Software Implementation

### Development Environment
- **Python**: 3.10.18
- **Operating System**: Windows 10 (build 26100)
- **Deep Learning Framework**: PyTorch 2.0.1+cu117
- **CUDA**: 11.7 with cuDNN 8.5.0

### Key Dependencies
```
torch==2.0.1+cu117
pandas==2.3.0
numpy==1.26.4
scikit-learn==1.7.2
matplotlib==3.10.6
```

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
- **Memory**: 16GB RAM minimum
- **Storage**: 2GB for data and model checkpoints

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mobilazy/CNN-LSTM_geo.git
cd CNN-LSTM_geo
```

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate msgeothermal-env
```

3. Verify installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Key Features

- **Comprehensive Data Cleaning**: Eight-stage pipeline removes sensor artifacts while preserving thermal dynamics
- **Multi-Configuration Analysis**: Simultaneous training on three different BHE designs
- **GPU Optimization**: Mixed precision training with batch size optimization (up to 1024)
- **Temporal Validation**: Calendar-based splitting preserves temporal coherence
- **Rich Visualizations**: Automated generation of performance plots and architecture diagrams
- **Sensor Malfunction Detection**: CNN-LSTM model reconstructs faulty sensor readings

## Project Structure

```
CNN-LSTM_geo/
├── input/                          # Raw sensor data CSV files
│   ├── MeterOE401_singleU45.csv
│   ├── MeterOE402_Ellipse63.csv
│   └── MeterOE403_doubleU45.csv
├── output/                         # Generated results and visualizations
│   ├── comprehensive_model.pth
│   ├── comprehensive_results.json
│   └── *.png
├── docs/                           # Documentation and analysis reports
├── archive/                        # Previous model versions
├── Traindata_geothermal_HybridCNNLSTM_rev10_Fixed.py  # Main training script
├── sensor_recovery_rt514.py        # Sensor malfunction recovery
├── plot_training_convergence.py   # Training metrics visualization
├── visualize_architecture.py      # Model architecture diagram
└── environment.yml                 # Conda environment specification
```

## Publications and References

This work is part of ongoing research at the University of Stavanger on geothermal energy systems and machine learning applications for predictive maintenance.

### Related Documentation
- `docs/Chapter_4_Results.md` - Comprehensive results analysis
- `docs/Chapter_4_6_Sensor_Malfunction_Recovery.md` - Sensor recovery methodology
- `docs/Chapter_U-shaped_BHE.md` - U-shaped BHE configuration analysis
- `docs/Literature_Gap_Section.md` - Research context and literature review

## Future Development

- **Real-time Prediction**: Deploy model for live sensor data streaming
- **Extended Forecasting**: Multi-step ahead predictions beyond 21 days
- **Transfer Learning**: Apply trained model to other geothermal installations
- **Anomaly Detection**: Automated fault detection and alerting system
- **Optimization**: Recommend optimal operating parameters for efficiency

---

## Contact and Contributions

For questions, issues, or contributions, please open an issue on the GitHub repository or contact the research team at the University of Stavanger.

**Repository**: [https://github.com/mobilazy/CNN-LSTM_geo](https://github.com/mobilazy/CNN-LSTM_geo)
- `torch`, `torch.nn`, `torch.optim`: For deep learning operations and model training.
- `matplotlib` and `pandas`: For plotting training/testing metrics and handling data.
- `os`: For file and folder operations.
- `HybridCNNLSTMAttention` and `CustomDataset` are imported from custom files `model.py` and `data_loader.py`.

## 2. **Defining Model and File Paths**

```python
model_name = "HybridCNNLSTMAttention"
ext = "TS"
```

- `model_name` specifies the name of the model.
- `ext` is used as an additional identifier for file naming.

### Model Parameters:
```python
input_size = 8
cnn_channels = 16
lstm_hidden_size = 8
lstm_num_layers = 1
output_size = 4
```
- **input_size**: The number of features in each time series input.
- **cnn_channels**: The number of channels in the first CNN layer.
- **lstm_hidden_size**: Hidden size of the LSTM layers.
- **lstm_num_layers**: Number of stacked LSTM layers.
- **output_size**: Number of output classes (4 in this case).

### Initialize Model:

```python
model = HybridCNNLSTMAttention(input_size, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size)
```

- The `HybridCNNLSTMAttention` model is initialized using the defined parameters.

### Set Device (CPU or GPU):

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

- The model is moved to GPU if available, otherwise it uses the CPU.

## 3. **Defining Loss Function and Optimizer**

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

- **Loss Function**: Cross-entropy loss is used as the classification criterion.
- **Optimizer**: Stochastic Gradient Descent (SGD) is used with a learning rate of `0.01` for updating the model parameters during training.

### Alternatives to Cross-Entropy Loss and Optimizers:

#### **1. Alternatives to Cross-Entropy Loss**
Cross-entropy loss is widely used for classification problems, but other loss functions may be suitable based on your task:

- **Focal Loss**
  - Suitable for imbalanced datasets.
  - Focuses more on hard-to-classify samples by down-weighting easy samples.
  - Implementation: Available in libraries like PyTorch or custom implementations.

- **Mean Squared Error (MSE)**
  - Typically used for regression but can be applied to classification when one-hot encoding is used.
  - Not ideal for classification as it treats probabilities linearly.

- **Kullback-Leibler Divergence Loss (KLDivLoss)**
  - Measures the divergence between two probability distributions.
  - Useful when comparing soft labels or probabilistic outputs.

- **Hinge Loss**
  - Commonly used for binary classification tasks with Support Vector Machines (SVMs).
  - Encourages a margin of separation between classes.

- **Label Smoothing**
  - A variation of cross-entropy loss that smooths target labels to prevent overconfidence.
  - Useful in cases prone to overfitting or noisy labels.

- **Binary Cross-Entropy (BCE)**
  - Specialized for binary classification tasks.
  - Can also be extended to multi-label classification problems.

- **Contrastive Loss**
  - Useful in tasks like face recognition or similarity learning.
  - Operates on pairs of samples to measure the similarity or dissimilarity.

---

#### **2. Alternatives to Stochastic Gradient Descent (SGD) Optimizer**
Depending on the nature of your problem and dataset, alternative optimizers may provide better convergence:

- **Adam (Adaptive Moment Estimation)**
  - Combines the advantages of RMSProp and momentum.
  - Well-suited for sparse data and non-stationary objectives.
  - Common usage: `optim.Adam(model.parameters(), lr=0.001)`

- **AdamW (Adam with Weight Decay Regularization)**
  - Variation of Adam with improved weight decay regularization.
  - Helps prevent overfitting.
  - Common usage: `optim.AdamW(model.parameters(), lr=0.001)`

- **RMSProp (Root Mean Square Propagation)**
  - Divides the learning rate by a running average of the magnitudes of recent gradients.
  - Well-suited for recurrent neural networks (RNNs).
  - Common usage: `optim.RMSprop(model.parameters(), lr=0.001)`

- **Adagrad (Adaptive Gradient Algorithm)**
  - Adapts learning rates based on historical gradient information.
  - Suitable for sparse data or parameters.
  - Common usage: `optim.Adagrad(model.parameters(), lr=0.001)`

- **Adadelta**
  - Addresses some limitations of Adagrad by restricting step size.
  - Common usage: `optim.Adadelta(model.parameters(), lr=1.0)`

- **NAdam (Nesterov-accelerated Adam)**
  - Extends Adam by incorporating Nesterov momentum.
  - Common usage: `optim.NAdam(model.parameters(), lr=0.001)`

- **LBFGS (Limited-memory BFGS)**
  - A quasi-Newton method optimizer.
  - Suitable for smaller datasets and optimization problems with second-order behavior.
  - Common usage: `optim.LBFGS(model.parameters(), lr=0.1)`

---

### Selecting Alternatives:
- **Classification Tasks**: 
  - Use Focal Loss or Label Smoothing if data is imbalanced.
  - Use Hinge Loss for binary classification with margin-based separation.

- **Optimizers for Stability**:
  - Adam and AdamW are generally more stable for deep learning tasks.
  - RMSProp is preferred for RNNs or non-stationary datasets.

- **Fine-Tuning Hyperparameters**:
  - Experiment with learning rates, momentum, and weight decay to adapt optimizers to your dataset.

### Example:
```python
# Alternative Loss and Optimizer
criterion = nn.CrossEntropyLoss()  # Or FocalLoss(), KLDivLoss(), etc.
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Or RMSProp, Adagrad
```
  
### Table of Alternatives to Cross-Entropy Loss and SGD Optimizer

| **Type**               | **Name**              | **Description**                                                                                                                                   | **Implementation (PyTorch)**                                                                                     |
|------------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| **Loss Functions**     | **Cross-Entropy Loss** | Default for classification tasks.                                                                                                                 | `nn.CrossEntropyLoss()`                                                                                          |
|                        | **Focal Loss**         | Focuses on hard-to-classify samples; reduces the influence of easy samples.                                                                       | [Focal Loss Implementation](https://github.com/AdeelH/pytorch-multi-class-focal-loss)                            |
|                        | **Mean Squared Error** | Regression-based loss, less common for classification.                                                                                           | `nn.MSELoss()`                                                                                                   |
|                        | **KL Divergence Loss** | Measures the divergence between predicted and target distributions.                                                                               | `nn.KLDivLoss()`                                                                                                 |
|                        | **Hinge Loss**         | Encourages a margin of separation between classes; used in SVMs.                                                                                  | `nn.HingeEmbeddingLoss()`                                                                                        |
|                        | **Label Smoothing**    | Reduces overconfidence by smoothing target labels.                                                                                                | `nn.CrossEntropyLoss(label_smoothing=0.1)` (PyTorch 1.10+)                                                       |
|                        | **Binary Cross-Entropy** | For binary or multi-label classification.                                                                                                        | `nn.BCELoss()` or `nn.BCEWithLogitsLoss()`                                                                       |
|                        | **Contrastive Loss**   | Used for similarity or metric learning tasks.                                                                                                    | Custom: See [Contrastive Loss Implementation](https://omoindrot.github.io/triplet-loss)                          |
| **Optimizers**         | **SGD**               | Basic optimizer with momentum.                                                                                                                    | `optim.SGD(model.parameters(), lr=0.01, momentum=0.9)`                                                           |
|                        | **Adam**              | Combines RMSProp and momentum; adapts learning rates.                                                                                            | `optim.Adam(model.parameters(), lr=0.001)`                                                                       |
|                        | **AdamW**             | Adam with decoupled weight decay for better regularization.                                                                                       | `optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)`                                                  |
|                        | **RMSProp**           | Scales learning rates based on recent gradient magnitudes; good for RNNs.                                                                        | `optim.RMSprop(model.parameters(), lr=0.001)`                                                                    |
|                        | **Adagrad**           | Adapts learning rates for parameters with infrequent updates.                                                                                    | `optim.Adagrad(model.parameters(), lr=0.01)`                                                                     |
|                        | **Adadelta**          | Improves Adagrad by limiting step sizes for better stability.                                                                                     | `optim.Adadelta(model.parameters(), lr=1.0)`                                                                     |
|                        | **NAdam**             | Combines Adam and Nesterov momentum for faster convergence.                                                                                      | `optim.NAdam(model.parameters(), lr=0.001)`                                                                      |
|                        | **LBFGS**             | Quasi-Newton method for small datasets or second-order optimization.                                                                              | `optim.LBFGS(model.parameters(), lr=0.1)`                                                                        |

### Example Code Snippet

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Example Loss Function: Cross-Entropy Loss
criterion = nn.CrossEntropyLoss()

# Example Optimizer: AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Focal Loss Example (Custom Implementation)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss

criterion = FocalLoss(alpha=0.25, gamma=2)
```

### Notes
- Use **Cross-Entropy Loss** for most classification tasks, unless specific challenges like class imbalance or noisy labels exist.
- Use **AdamW** or **RMSProp** as alternatives to SGD for better convergence in deep learning tasks.

## 4. **Loading the Dataset**

### Paths to Training and Testing Data:

```python
train_csv_path = r"train.csv"
test_csv_path = r"test.csv"
```

- **train_csv_path**: Path to the training dataset (normalized time series).
- **test_csv_path**: Path to the testing dataset.

### Create Dataset Objects:

```python
train_dataset = CustomDataset(train_csv_path)
test_dataset = CustomDataset(test_csv_path)
```

- **train_dataset** and **test_dataset** are instances of the `CustomDataset` class for loading the training and testing data.

### Create DataLoader Objects:

```python
train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
```

- **train_data_loader**: Loads the training data in batches of size `1` and shuffles the data.
- **test_data_loader**: Loads the testing data in batches of size `1` without shuffling.

## 5. **Training and Testing Loop**

### Initialize Lists for Storing Metrics:
```python
epochs = 200
train_loss_values = []
test_loss_values = []
train_accuracy_values = []
test_accuracy_values = []
```

- These lists will store loss and accuracy values for each epoch during training and testing.

### Epoch Loop:

```python
for epoch in range(epochs):
```

- A loop that runs for 200 epochs to train the model. Each epoch consists of two phases: training and testing.

### Training Phase:
```python
model.train()
epoch_train_loss = 0.0
correct_train = 0
total_train = 0
```

- The model is set to training mode using `model.train()`.
- Initialize the loss and accuracy counters for the current epoch.

#### Mini-Batch Training:
```python
for inputs, labels in train_data_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

- The inputs and labels are moved to the selected device (CPU/GPU).
- The model processes the input and calculates the loss.
- The optimizer updates the model weights using backpropagation.

#### Track Accuracy and Loss:
```python
epoch_train_loss += loss.item()
_, predicted_train = torch.max(outputs.data, 1)
total_train += labels.size(0)
correct_train += (predicted_train == labels).sum().item()
```

- The total training loss and the number of correct predictions are tracked for each batch.
- After completing all batches in the epoch, the loss and accuracy are averaged and stored.

### Testing Phase:

```python
model.eval()
epoch_test_loss = 0.0
correct_test = 0
total_test = 0
```

- The model is set to evaluation mode using `model.eval()`.
- The loss and accuracy for testing data are tracked similarly to the training phase, but without backpropagation (`torch.no_grad()`).

### Store Loss and Accuracy:
```python
train_loss_values.append(epoch_train_loss)
train_accuracy_values.append(train_accuracy)
test_loss_values.append(epoch_test_loss)
test_accuracy_values.append(test_accuracy)
```

- Loss and accuracy values for both training and testing phases are stored in the respective lists.

### Print Epoch Results:

```python
if (epoch + 1) % 5 == 0:
    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
```

- Results are printed every 5 epochs, displaying the training and testing loss and accuracy.

## 6. **Saving the Model and Data**

### Create Output Folders:
```python
os.makedirs(output_folder1, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)
os.makedirs(output_folder3, exist_ok=True)
```

- These lines ensure that the specified output directories exist, and create them if they don't.

### Save Model:
```python
model_path = os.path.join(output_folder1, f"{model_name}_{ext}.pth")
torch.save(model.state_dict(), model_path)
```

- The trained model's weights are saved in a `.pth` file for future inference or further training.

### Save Training Data:
```python
train_info_df = pd.DataFrame(train_info)
csv_path = os.path.join(output_folder2, f"{model_name}_{ext}.csv")
train_info_df.to_csv(csv_path, index=False)
```

- The collected training and testing loss/accuracy values are saved in a CSV file for further analysis.

## 7. **Plotting the Results**

### Create Plot for Loss and Accuracy:
```python
plt.figure(figsize=(12, 4))
plt.subplot(2, 1, 1)
plt.plot(range(1, epochs + 1), train_loss_values, label='Training Loss')
plt.plot(range(1, epochs + 1), test_loss_values, label='Testing Loss')
```

- A figure is created with two subplots: one for loss and one for accuracy over epochs.

### Save the Plot:
```python
png_file_path = os.path.join(output_folder3, f"{model_name}_{ext}.png")
pdf_file_path = os.path.join(output_folder3, f"{model_name}_{ext}.pdf")
plt.savefig(png_file_path, format='png', dpi=600)
plt.savefig(pdf_file_path, format='pdf', dpi=600)
```

- The plots are saved as both PNG and PDF files.

## 8. **Displaying the Plot**

```python
plt.tight_layout()
plt.show()
```

- The layout is adjusted to avoid overlap, and the plot is displayed.

---

This step-by-step breakdown explains the entire process of building, training, testing, saving, and visualizing the performance of the **Hybrid CNN-LSTM Attention model**.


# Hybrid CNN-LSTM Model Training Script

## 1. Import Required Libraries
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import os
from model import HybridCNNLSTM
from data_loader import CustomDataset
```
- Imports essential PyTorch libraries for deep learning
- Includes data handling, neural network modules, and optimization tools
- Imports custom model and data loader classes

## 2. Model and Training Configuration
```python
model_name = "HybridCNNLSTM"
ext = "TS"

# Model parameters
input_channels = 10001
cnn_channels = 24
lstm_hidden_size = 12
lstm_num_layers = 2
output_size = 9
num_epochs = 100
learning_Rate = 0.01
batch_Size = 64
```
- Defines model hyperparameters
- `input_channels`: Number of input features
- `cnn_channels`: Convolutional layers channels
- `lstm_hidden_size`: LSTM hidden layer size
- `lstm_num_layers`: Number of LSTM layers
- `output_size`: Number of classification categories
- Configures training settings like epochs, learning rate, and batch size

## 3. Model Initialization and Device Selection
```python
model = HybridCNNLSTM(input_channels, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```
- Instantiates the Hybrid CNN-LSTM model
- Selects computational device (GPU or CPU)
- Moves model to the selected device

## 4. Loss Function and Optimizer Configuration
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_Rate)
```
- Uses Cross-Entropy Loss for multi-class classification
- Configures Stochastic Gradient Descent (SGD) optimizer

## 5. Data Preparation
```python
train_dataset = CustomDataset(train_csv_path)
test_dataset = CustomDataset(test_csv_path)

train_data_loader = DataLoader(train_dataset, batch_size=batch_Size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_Size, shuffle=False)
```
- Creates custom datasets from CSV files
- Prepares DataLoaders for training and testing
- Enables batch processing and data shuffling

## 6. Training Loop
```python
for epoch in range(epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_data_loader:
        # Move data to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track training metrics
        epoch_train_loss += loss.item()
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()
```
Key Training Steps:
- Set model to training mode
- Process data in batches
- Perform forward and backward passes
- Apply gradient clipping
- Calculate training loss and accuracy

## 7. Testing Phase
```python
    # Testing phase
    model.eval()
    epoch_test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Track testing metrics
            epoch_test_loss += loss.item()
            _, predicted_test = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted_test == labels).sum().item()
```
Testing Steps:
- Set model to evaluation mode
- Disable gradient computation
- Calculate test loss and accuracy

## 8. Model and Results Saving
```python
# Create output directories
os.makedirs(output_folder1, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)
os.makedirs(output_folder3, exist_ok=True)

# Save model state
model_path = os.path.join(output_folder1, f"{model_name}_{ext}.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'hyperparameters': {...}
}, model_path)

# Save training metrics
train_info_df = pd.DataFrame(train_info)
csv_path = os.path.join(output_folder2, f"{model_name}_{ext}.csv")
train_info_df.to_csv(csv_path, index=False)
```
- Saves model state and hyperparameters
- Stores training and testing metrics in a CSV file

## 9. Visualization of Training Metrics
```python
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(2, 1, 1)
plt.plot(range(1, epochs + 1), train_loss_values, label='Training Loss')
plt.plot(range(1, epochs + 1), test_loss_values, label='Testing Loss')

# Accuracy plot
plt.subplot(2, 1, 2)
plt.plot(range(1, epochs + 1), train_accuracy_values, label='Training Accuracy')
plt.plot(range(1, epochs + 1), test_accuracy_values, label='Testing Accuracy')

# Save plots
plt.savefig(png_file_path, format='png', dpi=600)
plt.savefig(pdf_file_path, format='pdf', dpi=600)
```
- Creates subplots for loss and accuracy
- Plots training and testing metrics
- Saves visualizations in PNG and PDF formats

## Key Features and Best Practices
- GPU/CPU device selection
- Gradient clipping
- Detailed metric tracking
- Flexible model configuration
- Comprehensive result saving and visualization

I'll break down the model code and explain the different hybrid neural network architectures:

# Hybrid CNN-LSTM Neural Network Architectures

## 1. HybridCNNLSTMAttention Model

### Key Components
```python
class HybridCNNLSTMAttention(nn.Module):
    def __init__(self, input_size, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size):
        # CNN Layers
        self.cnn = nn.Sequential(
            nn.Conv1d(...),
            nn.ReLU(),
            nn.MaxPool1d(...),
            nn.Conv1d(...),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(...)
        )
        
        # Spatial Attention Mechanism
        self.attention = SpatialAttention(...)
        
        # LSTM Layers
        self.lstm = nn.LSTM(...)
        
        # Fully Connected Classification Layer
        self.fc = nn.Linear(...)
```

### Unique Features
- Incorporates a Spatial Attention mechanism
- Uses adaptive max pooling
- Flexible input handling

### Spatial Attention Mechanism
```python
class SpatialAttention(nn.Module):
    def __init__(self, input_dim):
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=1, kernel_size=1)

    def forward(self, x):
        # Compute attention weights
        attn_weights = torch.softmax(self.conv(x), dim=-1)
        return x * attn_weights
```
- Learns to focus on important spatial features
- Uses a 1D convolution to generate attention weights

## 2. Standard HybridCNNLSTM Model

### Architecture
```python
class HybridCNNLSTM(nn.Module):
    def __init__(self, input_channels, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size):
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(...),  # First convolutional layer
            nn.ReLU(),
            nn.MaxPool1d(...),  # Downsampling
            nn.Conv1d(...),  # Second convolutional layer
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(...)  # Adaptive pooling
        )
        
        # LSTM for Sequential Modeling
        self.lstm = nn.LSTM(...)
        
        # Classification Layer
        self.fc = nn.Linear(...)
```

### Key Characteristics
- Combines Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM)
- CNN extracts spatial features
- LSTM captures temporal dependencies
- Final fully connected layer for classification

## 3. Input Preprocessing and Handling

### Input Shape Transformation
```python
def forward(self, x):
    # Handle 2D inputs by adding channel dimension
    if x.dim() == 2:
        x = x.unsqueeze(1)

    # Validate input dimensions
    if x.dim() != 3:
        raise ValueError("Expected 3D input tensor")

    # Ensure correct input channels
    if x.size(1) == 1:
        x = x.repeat(1, input_channels, 1)
```

### Processing Steps
1. Check input tensor dimensions
2. Add channel dimension if missing
3. Repeat single channel to match required input channels

## 4. Model Flow

### Forward Propagation
1. CNN Feature Extraction
   - Convolutional layers
   - ReLU activation
   - Pooling for dimensionality reduction

2. LSTM Sequential Processing
   - Process CNN features
   - Capture temporal dependencies

3. Classification
   - Use final LSTM output
   - Fully connected layer for multi-class prediction

## 5. Model Variants and Customization

### Attention-Based Model
- Adds spatial attention mechanism
- Dynamically focuses on important features

### Standard Model
- Simple CNN-LSTM architecture
- Fixed feature extraction

## 6. Recommended Modifications

### Hyperparameter Tuning
- Adjust `cnn_channels`
- Modify `lstm_hidden_size`
- Experiment with layer depths

### Input Handling
- Implement more robust input validation
- Add flexible channel replication

## 7. Performance Considerations
- Use GPU for faster computation
- Monitor overfitting
- Experiment with different architectures

## Conclusion
These hybrid models combine:
- Spatial feature extraction (CNN)
- Temporal sequence modeling (LSTM)
- Optional attention mechanisms

Choose based on:
- Dataset characteristics
- Computational resources
- Specific problem requirements


# Split Dataset using Rolling Window

## Purpose of the Script
This Python script is designed to handle a large time-series dataset by:
- **Creating a rolling window of data:** This allows for sequential analysis where each sample includes historical data, which is crucial for time-series forecasting.
- **Splitting the data into training and testing sets:** This is done to prepare data for machine learning models, specifically for models that benefit from time-series data structuring like RNNs or LSTMs.
- **Processing data incrementally:** This approach helps manage memory by not loading the entire dataset into memory at once, which is particularly useful for very large datasets.

### Step-by-Step Implementation
1. **Imports and Initial Setup**
- **Libraries:** pandas, numpy, and os are imported for data manipulation, numerical operations, and file path operations respectively.
- **Data Loading:** The dataset is loaded from a CSV file into a pandas DataFrame.

```python
import pandas as pd
import numpy as np
import os
file_path = r"C:\data.csv"
data = pd.read_csv(file_path)
```

2. **Configuration**
- **Parameters:** train_ratio, window_size, stride, and chunk_size are defined to control data splitting, window creation, and chunk processing. 
- **Paths:** Paths for output files are set.

```python
train_ratio = 0.8
window_size = 10
stride = 1
chunk_size = 1000
output_folder = r"C:\MLFiles"
train_csv_path = os.path.join(output_folder, 'train.csv')
test_csv_path = os.path.join(output_folder, 'test.csv')
```

3. **Function for Incremental Processing**
- Function process_and_save_incrementally:
  - Defines how features and labels are structured based on the window size.
  - Writes a header to the output CSV file.
  - Processes data in chunks to manage memory:
   - Reads a chunk of data.
   - Creates windows of data, where each window is window_size rows from the past, predicting one step ahead.
   - Writes these windows to the CSV file in an append mode.

```python
def process_and_save_incrementally(data, window_size, stride, output_path, chunk_size=1000):
    # ... (function body as described)
```

4. **Data Splitting**
- The dataset is split into training and testing sets based on train_ratio.

```python
split_index = int(len(data) * train_ratio)
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]
```

5. **Processing and Saving Data**
- Both training and test data are processed using the process_and_save_incrementally function, which writes the reformatted data into CSV files.

```python
process_and_save_incrementally(train_data, window_size, stride, train_csv_path, chunk_size)
process_and_save_incrementally(test_data, window_size, stride, test_csv_path, chunk_size)
```

6. **Verification**
- Checks file sizes to confirm data was written.
- Reads and prints the first few rows of both files to verify the data structure.

```python
train_size = os.path.getsize(train_csv_path) / (1024 * 1024)
test_size = os.path.getsize(test_csv_path) / (1024 * 1024)
print(f"Train dataset saved as {train_csv_path} ({train_size:.1f} MB)")
print(f"Test dataset saved as {test_csv_path} ({test_size:.1f} MB)")
print(pd.read_csv(train_csv_path, nrows=10))
print(pd.read_csv(test_csv_path, nrows=10))
```

**Conclusion**
This script essentially transforms raw time-series data into a format suitable for sequential learning models by applying a sliding window technique, all while managing memory usage through incremental processing. This approach ensures that even very large datasets can be processed without overwhelming system resources.

<div align="center">
  <a href="https://maazsalman.org/">
    <img width="70" src="click-svgrepo-com.svg" alt="gh" />
  </a>
  <p> Explore More! 🚀</p>
</div>


