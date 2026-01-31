"""
Mock Data Generator for GT Operation and CEMS Data
Generates realistic synthetic data for XGBoost model development.

Data Configuration:
- Training: 1 month (~720 hourly samples)
- Testing: 1 week (~168 hourly samples)
- Total: ~888 samples
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Seed for reproducibility
np.random.seed(42)

def generate_gt_operation_data(start_time: datetime, num_hours: int) -> pd.DataFrame:
    """
    Generate Gas Turbine operation data.
    
    Features (typical base load ranges):
    - Fuel Gas Flow (kg/s): 8-12 kg/s
    - GCV Position (%): 70-95%
    - IGV Position (deg): 80-88 degrees
    - Stop/Speed Ratio Valve Position (%): 90-100%
    - GT Exhaust Temp (°C): 520-580°C
    - Exhaust Duct Temp (°C): 480-540°C
    - Wheel Space Temp Avg (°C): 400-450°C
    - Compressor Inlet Temp (°C): 25-35°C (tropical climate)
    - Compressor Discharge Pressure (bar): 14-18 bar
    - Generator Power (MW): 100-150 MW (typical CCGT unit)
    """
    timestamps = [start_time + timedelta(hours=i) for i in range(num_hours)]
    
    # Base load operation with slight variations
    base_power = 130  # MW
    power_variation = np.random.normal(0, 5, num_hours)
    generator_power = np.clip(base_power + power_variation, 100, 150)
    
    # Fuel flow correlates with power
    fuel_flow = 8 + (generator_power - 100) * 0.08 + np.random.normal(0, 0.3, num_hours)
    
    # GCV position correlates with fuel flow
    gcv_position = 70 + (fuel_flow - 8) * 6 + np.random.normal(0, 1, num_hours)
    gcv_position = np.clip(gcv_position, 70, 95)
    
    # IGV position (relatively stable in base load)
    igv_position = 84 + np.random.normal(0, 1.5, num_hours)
    igv_position = np.clip(igv_position, 80, 88)
    
    # Stop/Speed Ratio Valve (very stable in base load)
    ssrv_position = 95 + np.random.normal(0, 1.5, num_hours)
    ssrv_position = np.clip(ssrv_position, 90, 100)
    
    # GT Exhaust Temp correlates with power and fuel
    gt_exhaust_temp = 520 + (generator_power - 100) * 1.2 + np.random.normal(0, 5, num_hours)
    gt_exhaust_temp = np.clip(gt_exhaust_temp, 520, 580)
    
    # Exhaust Duct Temp (slightly lower than GT exhaust)
    exhaust_duct_temp = gt_exhaust_temp - 40 + np.random.normal(0, 5, num_hours)
    
    # Wheel Space Temp (Stage 1-3 average)
    wheel_space_temp = 400 + (gt_exhaust_temp - 520) * 0.5 + np.random.normal(0, 5, num_hours)
    
    # Compressor Inlet Temp (ambient, diurnal variation)
    hour_of_day = np.array([t.hour for t in timestamps])
    ambient_base = 28 + 5 * np.sin((hour_of_day - 6) * np.pi / 12)  # Peak at 14:00
    compressor_inlet_temp = ambient_base + np.random.normal(0, 1, num_hours)
    
    # Compressor Discharge Pressure
    compressor_discharge_pressure = 14 + (generator_power - 100) * 0.08 + np.random.normal(0, 0.3, num_hours)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'fuel_gas_flow_kg_s': fuel_flow,
        'gcv_position_pct': gcv_position,
        'igv_position_deg': igv_position,
        'ssrv_position_pct': ssrv_position,
        'gt_exhaust_temp_c': gt_exhaust_temp,
        'exhaust_duct_temp_c': exhaust_duct_temp,
        'wheel_space_temp_avg_c': wheel_space_temp,
        'compressor_inlet_temp_c': compressor_inlet_temp,
        'compressor_discharge_pressure_bar': compressor_discharge_pressure,
        'generator_power_mw': generator_power
    })
    
    return df


def generate_cems_data(gt_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate CEMS emission data correlated with GT operation.
    
    Emissions (typical ranges for natural gas GT):
    - NOx (mg/Nm³): 15-50 (depends on combustion temp)
    - SOx (mg/Nm³): 1-5 (low for natural gas)
    - CO (mg/Nm³): 5-30 (incomplete combustion indicator)
    - PM (mg/Nm³): 1-10 (particulate matter)
    """
    n = len(gt_data)
    
    # NOx: strongly correlated with exhaust temp (thermal NOx)
    nox_base = (gt_data['gt_exhaust_temp_c'] - 520) * 0.5 + 20
    nox = nox_base + np.random.normal(0, 3, n)
    nox = np.clip(nox, 15, 50)
    
    # SOx: low for natural gas, slight correlation with fuel flow
    sox = 2 + (gt_data['fuel_gas_flow_kg_s'] - 8) * 0.3 + np.random.normal(0, 0.5, n)
    sox = np.clip(sox, 1, 5)
    
    # CO: inversely related to efficiency (higher at off-design)
    # Higher at lower loads or when combustion is less optimal
    co_base = 30 - (gt_data['generator_power_mw'] - 100) * 0.3
    co = co_base + np.random.normal(0, 3, n)
    co = np.clip(co, 5, 30)
    
    # PM: correlated with fuel flow and exhaust temp
    pm = 3 + (gt_data['fuel_gas_flow_kg_s'] - 8) * 0.8 + np.random.normal(0, 1, n)
    pm = np.clip(pm, 1, 10)
    
    # Add slight timestamp offset to simulate CEMS delay (1-3 minutes)
    # For hourly data, we just use same timestamp
    df = pd.DataFrame({
        'timestamp': gt_data['timestamp'],
        'nox_mg_nm3': nox,
        'sox_mg_nm3': sox,
        'co_mg_nm3': co,
        'pm_mg_nm3': pm
    })
    
    return df


def introduce_missing_values(df: pd.DataFrame, missing_rate: float = 0.02) -> pd.DataFrame:
    """Introduce random missing values to simulate real-world data quality issues."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if col != 'timestamp':
            mask = np.random.random(len(df_copy)) < missing_rate
            df_copy.loc[mask, col] = np.nan
    return df_copy


def main():
    # Configuration
    training_hours = 24 * 30  # 1 month = 720 hours
    testing_hours = 24 * 7   # 1 week = 168 hours
    total_hours = training_hours + testing_hours  # 888 hours
    
    start_time = datetime(2025, 12, 1, 0, 0, 0)  # Start date
    
    print("=" * 60)
    print("Mock Data Generator for GT Emissions Model")
    print("=" * 60)
    print(f"Training period: {training_hours} hours (1 month)")
    print(f"Testing period: {testing_hours} hours (1 week)")
    print(f"Total samples: {total_hours}")
    print()
    
    # Generate data
    print("Generating GT operation data...")
    gt_data = generate_gt_operation_data(start_time, total_hours)
    
    print("Generating CEMS emission data...")
    cems_data = generate_cems_data(gt_data)
    
    # Introduce some missing values
    print("Introducing realistic missing values (2%)...")
    gt_data = introduce_missing_values(gt_data, 0.02)
    cems_data = introduce_missing_values(cems_data, 0.02)
    
    # Create data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    gt_path = os.path.join(data_dir, 'gt_operation.csv')
    cems_path = os.path.join(data_dir, 'cems_data.csv')
    
    gt_data.to_csv(gt_path, index=False)
    cems_data.to_csv(cems_path, index=False)
    
    print()
    print("Data saved successfully!")
    print(f"  GT Operation Data: {gt_path}")
    print(f"  CEMS Data: {cems_path}")
    print()
    
    # Summary statistics
    print("=" * 60)
    print("GT Operation Data Summary:")
    print("=" * 60)
    print(gt_data.describe().round(2))
    print()
    print("=" * 60)
    print("CEMS Data Summary:")
    print("=" * 60)
    print(cems_data.describe().round(2))


if __name__ == "__main__":
    main()
