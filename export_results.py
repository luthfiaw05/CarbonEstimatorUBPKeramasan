"""
Export Results to Excel
Generates a comprehensive Excel report with multiple sheets.

Run with: python export_results.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import os
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Configuration
FEATURE_COLS = [
    'fuel_gas_flow_kg_s', 'gcv_position_pct', 'igv_position_deg',
    'ssrv_position_pct', 'gt_exhaust_temp_c', 'exhaust_duct_temp_c',
    'wheel_space_temp_avg_c', 'compressor_inlet_temp_c',
    'compressor_discharge_pressure_bar', 'generator_power_mw'
]

TARGET_COLS = ['nox_mg_nm3', 'sox_mg_nm3', 'co_mg_nm3', 'pm_mg_nm3']
EMISSION_NAMES = {'nox_mg_nm3': 'NOx', 'sox_mg_nm3': 'SOx', 'co_mg_nm3': 'CO', 'pm_mg_nm3': 'PM'}


def load_data():
    """Load merged data."""
    gt_path = os.path.join(DATA_DIR, 'gt_operation.csv')
    cems_path = os.path.join(DATA_DIR, 'cems_data.csv')
    
    gt_data = pd.read_csv(gt_path, parse_dates=['timestamp'])
    cems_data = pd.read_csv(cems_path, parse_dates=['timestamp'])
    
    merged = pd.merge(gt_data, cems_data, on='timestamp', how='left')
    return merged


def load_models():
    """Load trained models."""
    models = {}
    for emission, name in EMISSION_NAMES.items():
        model_path = os.path.join(OUTPUT_DIR, f'model_{name.lower()}.json')
        if os.path.exists(model_path):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            models[emission] = model
    return models


def create_summary_sheet(data, models):
    """Create summary metrics sheet."""
    summary = []
    
    # Data info
    summary.append({'Metric': 'Total Samples', 'Value': len(data)})
    summary.append({'Metric': 'Date Range Start', 'Value': str(data['timestamp'].min())})
    summary.append({'Metric': 'Date Range End', 'Value': str(data['timestamp'].max())})
    summary.append({'Metric': '', 'Value': ''})
    
    # GT Operation stats
    if 'generator_power_mw' in data.columns:
        summary.append({'Metric': 'Avg Generator Power (MW)', 'Value': f"{data['generator_power_mw'].mean():.2f}"})
    if 'fuel_gas_flow_kg_s' in data.columns:
        summary.append({'Metric': 'Avg Fuel Flow (kg/s)', 'Value': f"{data['fuel_gas_flow_kg_s'].mean():.2f}"})
    if 'gt_exhaust_temp_c' in data.columns:
        summary.append({'Metric': 'Avg GT Exhaust Temp (°C)', 'Value': f"{data['gt_exhaust_temp_c'].mean():.2f}"})
    
    summary.append({'Metric': '', 'Value': ''})
    
    # Emissions stats
    for col in TARGET_COLS:
        if col in data.columns:
            name = EMISSION_NAMES[col]
            summary.append({'Metric': f'Avg {name} (mg/Nm³)', 'Value': f"{data[col].mean():.2f}"})
    
    summary.append({'Metric': '', 'Value': ''})
    
    # Note about model metrics
    summary.append({'Metric': 'Model Info', 'Value': f'{len(models)} models available'})
    summary.append({'Metric': 'Note', 'Value': 'Run train_emissions_model.py for detailed metrics'})
    
    return pd.DataFrame(summary)


def create_timeseries_sheet(data, models):
    """Create full time series (actual values only, predictions require full pipeline)."""
    df = data.copy()
    # Note: Model predictions skipped as they require full feature engineering
    return df


def create_feature_importance_sheet(models):
    """Create feature importance sheet."""
    if not models:
        return pd.DataFrame()
    
    importance_data = []
    
    for emission, model in models.items():
        name = EMISSION_NAMES[emission]
        imp = model.feature_importances_
        
        for i, col in enumerate(FEATURE_COLS):
            importance_data.append({
                'Feature': col,
                'Emission': name,
                'Importance': imp[i]
            })
    
    return pd.DataFrame(importance_data)


def create_carbon_footprint_sheet(data):
    """Create carbon footprint calculations sheet."""
    CO2_PER_KG_FUEL = 2.75
    GWP_NOX = 298
    GWP_CO = 1.9
    GWP_PM = 100
    FLUE_GAS_FLOW = 500
    
    df = data.copy()
    
    # Ensure required columns exist
    required_cols = ['fuel_gas_flow_kg_s', 'nox_mg_nm3', 'co_mg_nm3', 'pm_mg_nm3']
    for col in required_cols:
        if col not in df.columns:
            print(f"  WARNING: Missing {col} for carbon calc. Filling with 0/NaN.")
            df[col] = 0
            
    # Calculate hourly values
    df['Direct_CO2_kg'] = df['fuel_gas_flow_kg_s'] * 3600 * CO2_PER_KG_FUEL
    
    # Emissions to kg
    total_nm3 = FLUE_GAS_FLOW * 3600
    df['NOx_kg'] = df['nox_mg_nm3'] * total_nm3 / 1e6
    df['CO_kg'] = df['co_mg_nm3'] * total_nm3 / 1e6
    df['PM_kg'] = df['pm_mg_nm3'] * total_nm3 / 1e6
    
    # CO2 equivalents
    df['NOx_CO2e_kg'] = df['NOx_kg'] * GWP_NOX
    df['CO_CO2e_kg'] = df['CO_kg'] * GWP_CO
    df['PM_CO2e_kg'] = df['PM_kg'] * GWP_PM
    
    df['Total_CO2e_kg'] = df['Direct_CO2_kg'] + df['NOx_CO2e_kg'] + df['CO_CO2e_kg'] + df['PM_CO2e_kg']
    
    # Select relevant columns safely
    out_cols = ['timestamp', 'fuel_gas_flow_kg_s', 'generator_power_mw',
                 'Direct_CO2_kg', 'NOx_CO2e_kg', 'CO_CO2e_kg', 'PM_CO2e_kg', 'Total_CO2e_kg']
    
    # Only keep columns that exist
    final_cols = [c for c in out_cols if c in df.columns]
    
    return df[final_cols]


def main():
    print("=" * 60)
    print("EXPORTING RESULTS TO EXCEL")
    print("=" * 60)
    
    # Load data and models
    print("\nLoading data...")
    data = load_data()
    print(f"  Loaded {len(data)} samples")
    
    print("\nLoading models...")
    models = load_models()
    print(f"  Loaded {len(models)} models")
    
    # Create sheets
    print("\nCreating sheets...")
    
    summary_df = create_summary_sheet(data, models)
    print("  ✓ Summary")
    
    timeseries_df = create_timeseries_sheet(data, models)
    print("  ✓ Time Series")
    
    importance_df = create_feature_importance_sheet(models)
    print("  ✓ Feature Importance")
    
    carbon_df = create_carbon_footprint_sheet(data)
    print("  ✓ Carbon Footprint")
    
    # Write to Excel
    output_path = os.path.join(OUTPUT_DIR, 'emissions_study_results.xlsx')
    
    print(f"\nWriting to Excel: {output_path}")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        timeseries_df.to_excel(writer, sheet_name='Time_Series', index=False)
        importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
        carbon_df.to_excel(writer, sheet_name='Carbon_Footprint', index=False)
    
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\nFile saved: {output_path}")


if __name__ == "__main__":
    main()
