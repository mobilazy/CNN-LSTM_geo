import pandas as pdimport pandas as pd

import Traindata_geothermal_HybridCNNLSTM_rev10_Fixed as rev10import numpy as np



sensor_df = pd.read_csv('input/DoubleU45_Treturn.csv')# Load sensor data

sensor_df.columns = [col.strip() for col in sensor_df.columns]sensor_df = pd.read_csv('input/DoubleU45_Treturn.csv')

sensor_df['Timestamp'] = pd.to_datetime(sensor_df['Timestamp'], format='%d.%m.%Y %H:%M', errors='coerce')sensor_df.columns = [col.strip().replace('', '') for col in sensor_df.columns]

sensor_df = sensor_df.sort_values('Timestamp').dropna(subset=['Timestamp']).reset_index(drop=True)sensor_df['Timestamp'] = pd.to_datetime(sensor_df['Timestamp'], format='%d.%m.%Y %H:%M', errors='coerce')

sensor_df = sensor_df.sort_values('Timestamp').dropna(subset=['Timestamp']).reset_index(drop=True)

oe403_df = rev10.load_double_u45mm_research_data()print(f'Sensor data: {len(sensor_df)} records')

print(f'Date range: {sensor_df[\"Timestamp\"].min()} to {sensor_df[\"Timestamp\"].max()}')

merged = sensor_df.merge(oe403_df[['Timestamp', 'supply_temp', 'flow_rate', 'power_kw']], on='Timestamp', how='left')

# Load OE403 via rev10

print(f'Sensor data: {len(sensor_df)} records')import sys

print(f'OE403 cleaned: {len(oe403_df)} records')import Traindata_geothermal_HybridCNNLSTM_rev10_Fixed as rev10

print(f'After merge: {len(merged)} records')oe403_df = rev10.load_double_u45mm_research_data()

supply_na = merged['supply_temp'].isna().sum()print(f'\nOE403 cleaned: {len(oe403_df)} records')

print(f'Missing supply_temp: {supply_na} records ({supply_na/len(merged)*100:.1f}%)')print(f'Date range: {oe403_df[\"Timestamp\"].min()} to {oe403_df[\"Timestamp\"].max()}')



fault_time = pd.Timestamp('2025-09-10 12:00')# Merge

prefault = merged[merged['Timestamp'] < fault_time]merged = sensor_df.merge(oe403_df[['Timestamp', 'supply_temp', 'flow_rate', 'power_kw']], on='Timestamp', how='left')

valid = (~prefault[['supply_temp', 'flow_rate', 'power_kw']].isna().any(axis=1)).sum()print(f'\nAfter merge: {len(merged)} records')

print(f'Pre-fault records: {len(prefault)}')print(f'Missing supply_temp: {merged[\"supply_temp\"].isna().sum()} ({merged[\"supply_temp\"].isna().sum()/len(merged)*100:.1f}%)')

print(f'Pre-fault with valid features: {valid} ({valid/len(prefault)*100:.1f}%)')print(f'Missing flow_rate: {merged[\"flow_rate\"].isna().sum()} ({merged[\"flow_rate\"].isna().sum()/len(merged)*100:.1f}%)')

print(f'Missing power_kw: {merged[\"power_kw\"].isna().sum()} ({merged[\"power_kw\"].isna().sum()/len(merged)*100:.1f}%)')

# Check pre-fault period
fault_time = pd.Timestamp('2025-09-10 12:00')
prefault = merged[merged['Timestamp'] < fault_time]
print(f'\nPre-fault records: {len(prefault)}')
print(f'Pre-fault with valid features: {(~prefault[[\"supply_temp\", \"flow_rate\", \"power_kw\"]].isna().any(axis=1)).sum()}')
