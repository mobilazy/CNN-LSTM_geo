
Sensor Malfunction Recovery

Sensor failures in geothermal systems create monitoring gaps that affect control and performance assessment. At the University of Stavanger facility, four return temperature sensors (RT512, RT513, RT514, RT515) monitor parallel boreholes connected to energy meter OE403. On September 10, 2025, RT514 failed, recording -49.0°C while the other sensors continued reporting valid data. The degradation started on September 8, 2025, with RT514 readings deviating from the healthy sensors. The malfunction was detected using a threshold below -10°C. The four boreholes share the same configuration and supply conditions, which creates thermal correlations useful for reconstruction.

Model Application and Data Preparation

The CNN-LSTM model (epoch 40, validation loss 0.0251°C) was applied without retraining. It uses supply temperature, flow rate, power, and BHE type encoding as input features, all obtained from OE403. Data cleaning followed the same 8-step pipeline as training: timestamp validation, numeric conversion, outlier removal, stuck sensor detection, median filtering, gap interpolation, physics validation, and missing value removal. This reduced the raw records to 66,380 valid measurements. Flow rate and power were divided by four for per-well values. Feature normalization used training statistics. The model processes 48-step input sequences to predict the next return temperature value.

Prediction and Validation

Model predictions were validated on the pre-fault period and extended for 8 days after the malfunction. The predictions tracked actual RT514 behavior, capturing seasonal and daily thermal variations. Predicted RT514 values remained within the operational envelope of the other three sensors. The reconstruction was stable throughout the forecast window.

Figure 23 shows the predictions for all four sensors, including the recovered RT514, compared to actual readings.

Baseline Comparison

A baseline method using the spatial average of the three healthy sensors was included for comparison. Both methods performed similarly: CNN-LSTM achieved MAE of 0.586°C and RMSE of 1.195°C, while the baseline had MAE of 0.603°C and RMSE of 1.132°C. The CNN-LSTM model provided a small improvement over the baseline, showing it learned the spatial correlation structure but did not add significant value beyond simple averaging for this configuration.

Figure 24 compares the CNN-LSTM model to the baseline, highlighting the difference in reconstruction accuracy.

Operational Implications and Limitations

The CNN-LSTM model captured thermal response relationships within the shared configuration. This enables continued monitoring and control using estimated values. Early detection of sensor degradation could support predictive maintenance. Limitations include dependency on OE403 data quality and the cleaning pipeline. The model provides point predictions without uncertainty estimates. Accuracy depends on healthy sensors and consistent configuration. The 8-day forecast is practical for continuous prediction; longer-term accuracy would require updates from healthy sensors. Results may not transfer to different configurations.
