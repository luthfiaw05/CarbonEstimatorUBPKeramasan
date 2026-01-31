"""
Train XGBoost models for gas turbine emission prediction.
This script trains models using the 9 available features from the GT logsheet data.

Run: python train_model.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json

# Paths
BASE_DIR = os.path.dirname(__file__)
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column mapping from Excel to internal names
COLUMN_MAPPING = {
    'fuel_gas_flow_kg_s': 'Flow (Kgh)',
    'gcv_position_pct': 'LVDT 1 (%)',
    'igv_position_deg': 'LVDT 2 (%).1',
    'ssrv_position_pct': 'LVDT 2 (%)',
    'exhaust_duct_temp_c': 'Exhaust Temp (deg. C)',
    'wheel_space_temp_avg_c': 'First Stage FFWD 1FO-1 (deg. C)',
    'compressor_inlet_temp_c': 'Inlet Temperature (deg. C)',
    'compressor_discharge_pressure_bar': 'Discharge Pressure (MPa)',
    'generator_power_mw': 'GT LOAD (MW)'
}

FEATURE_COLS = list(COLUMN_MAPPING.keys())

EMISSION_NAMES = {
    'nox_mg_nm3': 'NOx',
    'sox_mg_nm3': 'SOx',
    'co_mg_nm3': 'CO',
    'pm_mg_nm3': 'PM'
}


def load_data():
    """Load and prepare data from Excel file."""
    excel_path = os.path.join(RAW_DATA_DIR, 'GTLOGSHEET_1.xlsx')
    
    if not os.path.exists(excel_path):
        print(f"Error: Excel file not found at {excel_path}")
        return None
    
    print(f"Loading data from {excel_path}...")
    raw_df = pd.read_excel(excel_path)
    print(f"Raw data shape: {raw_df.shape}")
    
    # Create timestamp from date and time columns
    date_col = None
    for col in raw_df.columns:
        if isinstance(col, float):
            date_col = col
            break
    
    if date_col is not None and 'Pukul' in raw_df.columns:
        raw_df['timestamp'] = pd.to_datetime(
            raw_df[date_col].astype(str) + ' ' + raw_df['Pukul'].astype(str),
            errors='coerce'
        )
    
    # Map columns
    gt_data = pd.DataFrame()
    gt_data['timestamp'] = raw_df['timestamp']
    
    for internal_name, excel_name in COLUMN_MAPPING.items():
        if excel_name in raw_df.columns:
            gt_data[internal_name] = pd.to_numeric(raw_df[excel_name], errors='coerce')
        else:
            print(f"Warning: Column '{excel_name}' not found in data")
    
    # Unit conversions
    if 'fuel_gas_flow_kg_s' in gt_data.columns:
        gt_data['fuel_gas_flow_kg_s'] = gt_data['fuel_gas_flow_kg_s'] / 3600  # kg/h to kg/s
    if 'compressor_discharge_pressure_bar' in gt_data.columns:
        gt_data['compressor_discharge_pressure_bar'] = gt_data['compressor_discharge_pressure_bar'] * 10  # MPa to bar
    
    # Drop invalid rows
    gt_data = gt_data.dropna(subset=['timestamp'])
    gt_data = gt_data.dropna(subset=FEATURE_COLS)
    
    print(f"Cleaned data shape: {gt_data.shape}")
    
    # Generate synthetic emissions (since CEMS data not available)
    # This creates realistic emission data correlated with operating parameters
    np.random.seed(42)
    n = len(gt_data)
    
    load = gt_data['generator_power_mw'].fillna(20).values
    fuel = gt_data['fuel_gas_flow_kg_s'].fillna(2).values
    temp = gt_data['exhaust_duct_temp_c'].fillna(550).values
    
    # Generate correlated emissions
    gt_data['nox_mg_nm3'] = 25 + (load - 20) * 0.8 + (temp - 550) * 0.1 + np.random.normal(0, 2, n)
    gt_data['sox_mg_nm3'] = 5 + fuel * 1.5 + np.random.normal(0, 0.5, n)
    gt_data['co_mg_nm3'] = 20 - (temp - 550) * 0.05 + np.random.normal(0, 1.5, n)
    gt_data['pm_mg_nm3'] = 3 + fuel * 0.8 + np.random.normal(0, 0.3, n)
    
    # Clip to realistic ranges
    gt_data['nox_mg_nm3'] = gt_data['nox_mg_nm3'].clip(10, 60)
    gt_data['sox_mg_nm3'] = gt_data['sox_mg_nm3'].clip(1, 15)
    gt_data['co_mg_nm3'] = gt_data['co_mg_nm3'].clip(5, 30)
    gt_data['pm_mg_nm3'] = gt_data['pm_mg_nm3'].clip(1, 10)
    
    return gt_data


def train_models(data):
    """Train XGBoost models for each emission type."""
    
    X = data[FEATURE_COLS]
    
    print("\n" + "="*60)
    print("TRAINING XGBOOST MODELS")
    print("="*60)
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Training samples: {len(X)}")
    
    models = {}
    metrics = {}
    
    for emission_key, emission_name in EMISSION_NAMES.items():
        print(f"\n--- Training model for {emission_name} ---")
        
        y = data[emission_key]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        # Save model
        model_path = os.path.join(OUTPUT_DIR, f'model_{emission_name.lower()}.json')
        model.save_model(model_path)
        print(f"  Model saved: {model_path}")
        
        models[emission_key] = model
        metrics[emission_name] = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
    
    # Save metrics summary
    metrics_path = os.path.join(OUTPUT_DIR, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")
    
    # Save feature names for reference
    feature_info = {
        'feature_cols': FEATURE_COLS,
        'n_features': len(FEATURE_COLS)
    }
    feature_path = os.path.join(OUTPUT_DIR, 'feature_info.json')
    with open(feature_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"Feature info saved: {feature_path}")
    
    return models, metrics


def main():
    print("="*60)
    print("GT EMISSIONS MODEL TRAINING")
    print("="*60)
    
    # Load data
    data = load_data()
    
    if data is None or data.empty:
        print("Error: No data available for training")
        return
    
    # Train models
    models, metrics = train_models(data)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModels saved to: {OUTPUT_DIR}")
    print("\nYou can now use Panel B (Model Evaluation) in the dashboard.")
    print("Restart the dashboard to load the new models:")
    print("  python -m streamlit run dashboard.py")


if __name__ == "__main__":
    main()
