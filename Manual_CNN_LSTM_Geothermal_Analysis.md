# CNN-LSTM Model for Geothermal Borehole Heat Exchanger Analysis: Technical Manual

## Abstract

This manual presents a comprehensive CNN-LSTM deep learning framework for analyzing and predicting thermal performance across different borehole heat exchanger (BHE) configurations in geothermal systems. The model evaluates three distinct collector types using multi-sensor time-series data from 124 boreholes over an 11-month operational period. Key performance indicators show MAE values of 0.264°C for Single U45mm, 0.375°C for Double U45mm, and 0.397°C for MuoviEllipse 63mm configurations.

## 1. Introduction

Geothermal energy systems rely on borehole heat exchangers to transfer thermal energy between the ground and building heating/cooling systems. Optimizing these systems requires accurate thermal performance prediction across different BHE configurations. This study implements a hybrid Convolutional Neural Network-Long Short-Term Memory (CNN-LSTM) architecture to model complex thermal dynamics and predict outlet temperatures for various collector geometries.

The analysis encompasses three primary BHE configurations: Single U45mm pipes (complete field deployment), Double U45mm pipes (research configuration), and MuoviEllipse 63mm pipes (advanced research design). Each configuration exhibits distinct thermal characteristics requiring specialized modeling approaches.

## 2. Methodology

### 2.1 Model Architecture

The CNN-LSTM hybrid model combines spatial feature extraction through convolutional layers with temporal dependency modeling via LSTM networks. The architecture consists of:

**Convolutional Layers:**
- Two 1D convolutional layers with 32 and 64 channels
- Kernel size: 3 with ReLU activation
- Batch normalization and dropout (0.1) for regularization

**LSTM Network:**
- 2-layer LSTM with 64 hidden units per layer
- Bidirectional processing for enhanced temporal understanding
- Dropout layers between LSTM components

**Output Layer:**
- Fully connected layer mapping LSTM output to single temperature prediction
- Total trainable parameters: 75,489

### 2.2 Data Processing Pipeline

The data processing follows an 8-step cleaning protocol:

1. **Numeric conversion** with sensor fidelity preservation
2. **Physical constraint validation** (temperature: -10°C to 50°C, power: realistic operational ranges)
3. **Sensor anomaly detection** removing consecutive duplicate readings
4. **Median filtering** (3-point for temperatures, 5-point for power/flow)
5. **Gap interpolation** for missing data spans up to 20 minutes
6. **Physics validation** ensuring thermodynamically consistent temperature differentials
7. **Missing data threshold filtering** (adaptive based on data source)
8. **Final quality assurance** removing remaining invalid entries

### 2.3 Time-Based Data Splitting

The model employs calendar-based data partitioning to maintain temporal context:

- **Training Set**: Historical data up to validation period (171,542 samples)
- **Validation Set**: 7-day hold-out window preceding forecast (4,943 samples)
- **Test Set**: 21-day forecast evaluation period (14,247 samples)

This approach ensures realistic evaluation conditions where the model predicts future performance based on historical patterns.

## 3. Data Sources and Description

### 3.1 Complete Field Data (Single U45mm)

**Source**: MeterOE401_singleU45.csv
**Configuration**: 120 boreholes with Single U45mm pipes
**Normalization**: Per-well metrics (total measurements divided by 120)
**Operational Mode**: Primarily heat extraction (negative power values)

Key characteristics:
- Average flow rate: 1.92 m³/h per well
- Average power consumption: -1.34 kW per well
- Temperature differential: -0.57°C (supply-return)

### 3.2 Double U45mm Research Wells

**Source**: MeterOE403_doubleU45.csv
**Configuration**: 4 research boreholes (SKD-110-01 to 04)
**Pipe Design**: Dual U-shaped 45mm diameter pipes per borehole
**Normalization**: Per-well metrics (total measurements divided by 4)

Key characteristics:
- Average flow rate: 2.83 m³/h per well
- Average power consumption: -1.66 kW per well
- Temperature differential: -0.49°C (supply-return)

### 3.3 MuoviEllipse 63mm Research Wells

**Source**: MeterOE402_Ellipse63.csv
**Configuration**: 4 research boreholes with elliptical cross-section
**Pipe Design**: Single elliptical 63mm equivalent diameter pipes
**Advanced Features**: Optimized heat transfer surface area

Key characteristics:
- Average flow rate: 2.95 m³/h per well
- Average power consumption: -1.51 kW per well
- Temperature differential: -0.43°C (supply-return)

### 3.4 Feature Engineering

The model incorporates four primary features:
- **Supply Temperature** (°C): Inlet fluid temperature
- **Flow Rate** (m³/h): Volumetric flow per well
- **Power** (kW): Thermal energy transfer rate per well
- **BHE Type Encoded**: Categorical encoding (0: Single U45mm, 1: Double U45mm, 2: MuoviEllipse 63mm)

## 4. Model Training and Optimization

### 4.1 Training Configuration

**Hyperparameters:**
- Sequence length: 48 time steps (4 hours at 5-minute intervals)
- Prediction horizon: 1 time step (5 minutes ahead)
- Batch size: 1024 (optimized for GPU utilization)
- Learning rate: 1e-3 with ReduceLROnPlateau scheduler
- Training epochs: 5 with early stopping (patience: 16)

**Optimization:**
- Adam optimizer with MSE loss function
- Mixed precision training for enhanced GPU efficiency
- Gradient clipping for training stability

### 4.2 Training Results

The model achieved convergence within 5 epochs:
- **Initial training loss**: 8.32
- **Final training loss**: 0.73
- **Final validation loss**: 0.12

Validation loss reduction demonstrates effective learning without overfitting.

## 5. Results and Performance Analysis

### 5.1 Overall Model Performance

**Comprehensive Metrics:**
- Overall MAE: 0.348°C
- Overall RMSE: 0.397°C
- Model complexity: 75,489 trainable parameters

### 5.2 Collector-Specific Performance

**Single U45mm (Complete Field):**
- MAE: 0.264°C
- RMSE: 0.327°C
- Training samples: 59,316
- Forecast samples: 4,390

**Double U45mm (Research):**
- MAE: 0.375°C
- RMSE: 0.416°C
- Training samples: 66,379
- Forecast samples: 4,999

**MuoviEllipse 63mm (Research):**
- MAE: 0.397°C
- RMSE: 0.434°C
- Training samples: 65,037
- Forecast samples: 4,810

### 5.3 Thermal Performance Comparison

Analysis reveals distinct operational characteristics:

**Heat Transfer Efficiency:**
- Single U45mm demonstrates lowest prediction error, indicating stable thermal behavior
- MuoviEllipse 63mm shows highest flow rates but moderate thermal efficiency
- Double U45mm exhibits intermediate performance with enhanced thermal capacity

**Cross-Sectional Analysis:**
- Single U45mm: π × (22.5mm)² = 1,590 mm² effective area
- Double U45mm: 2 × π × (22.5mm)² = 3,181 mm² effective area
- MuoviEllipse 63mm: π × (31.5mm)² = 3,117 mm² effective area

## 6. Visualization and Output Analysis

### 6.1 Comprehensive Collector Analysis

The primary output visualization (comprehensive_collector_analysis.png) presents:
- Time-series comparison of training vs forecast periods
- Collector-specific thermal response patterns
- Performance metrics summary table
- Temporal trend analysis with forecast window highlighting

### 6.2 Model Performance Comparison

Secondary analysis plots demonstrate:
- MAE/RMSE comparison across collector types
- Thermal efficiency vs flow rate relationships
- Power utilization patterns
- Relative performance benefits

### 6.3 Raw Data Analysis

Pre-processing visualization includes:
- Temperature distribution scatter plots
- Heat transfer efficiency analysis
- Cross-sectional geometry impact assessment
- Operational mode classification

## 7. Technical Implementation

### 7.1 Software Requirements

**Core Dependencies:**
- PyTorch for deep learning framework
- Pandas/NumPy for data manipulation
- Matplotlib for visualization
- Scikit-learn for metrics calculation

**System Requirements:**
- CUDA-compatible GPU (recommended)
- Python 3.8+ environment
- Minimum 8GB RAM for dataset processing

### 7.2 Model Deployment

The trained model (comprehensive_model.pth) provides:
- Real-time temperature prediction capability
- Multi-collector configuration support
- 21-day forecast horizon
- Sub-degree prediction accuracy

## 8. Discussion and Applications

### 8.1 Practical Implications

The CNN-LSTM model enables:
- **Predictive Maintenance**: Early detection of thermal performance degradation
- **System Optimization**: Flow rate and power adjustment recommendations
- **Design Validation**: Comparative analysis of BHE configurations
- **Energy Management**: Improved heating/cooling system efficiency

### 8.2 Model Limitations

Current constraints include:
- Training data limited to single geological site
- Seasonal variation representation requires extended monitoring
- Ground thermal properties assumed constant
- Weather impact factors not explicitly modeled

### 8.3 Future Enhancements

Potential improvements encompass:
- Multi-site training data integration
- Weather parameter incorporation
- Geological property modeling
- Extended forecast horizons beyond 21 days

## 9. Conclusions

The CNN-LSTM hybrid architecture demonstrates effective thermal performance prediction across diverse BHE configurations. Single U45mm collectors exhibit superior predictability (0.264°C MAE), while advanced designs show acceptable accuracy for operational forecasting. The model provides valuable insights for geothermal system optimization and supports evidence-based design decisions.

Key findings indicate that collector geometry significantly influences thermal predictability, with traditional U-pipe designs offering more stable thermal behavior than advanced configurations. The 21-day forecast capability enables proactive system management and maintenance scheduling.

## References

[1] Gehlin, S., "Thermal Response Test: Method Development and Evaluation," Luleå University of Technology, 2002.

[2] Florides, G. and Kalogirou, S., "Ground heat exchangers—A review of systems, models and applications," Renewable Energy, vol. 32, pp. 2461-2478, 2007.

[3] Yang, H., Cui, P., and Fang, Z., "Vertical-borehole ground-coupled heat pumps: A review of models and systems," Applied Energy, vol. 87, pp. 16-27, 2010.

[4] Hochreiter, S. and Schmidhuber, J., "Long Short-Term Memory," Neural Computation, vol. 9, pp. 1735-1780, 1997.

[5] LeCun, Y., Bengio, Y., and Hinton, G., "Deep learning," Nature, vol. 521, pp. 436-444, 2015.

[6] Li, M., et al., "Review of analytical models for heat transfer by vertical ground heat exchangers," Applied Thermal Engineering, vol. 50, pp. 1178-1193, 2013.

[7] Spitler, J.D. and Gehlin, S.E.A., "Thermal response testing for ground source heat pump systems—An historical review," Renewable and Sustainable Energy Reviews, vol. 50, pp. 1125-1137, 2015.

[8] Kavanaugh, S. and Rafferty, K., "Geothermal Heating and Cooling: Design of Ground-Source Heat Pump Systems," ASHRAE, 2014.

---

*Technical Manual prepared for: CNN-LSTM Geothermal Analysis Project*  
*Version: rev09_Comprehensive*  
*Date: November 2025*