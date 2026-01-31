"""
XGBoost Regression Model for Gas Turbine Emissions Prediction
and Carbon Footprint Estimation - IMPROVED VERSION

Improvements:
- Cyclical time features (hour sin/cos)
- Lag features (t-1, t-2, t-3)
- Rolling statistics (3-hour averages)
- Hyperparameter tuning with RandomizedSearchCV
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
TRAINING_HOURS = 24 * 30  # 720 hours
TESTING_HOURS = 24 * 7    # 168 hours

# Original feature columns (GT Operation parameters)
BASE_FEATURE_COLS = [
    'fuel_gas_flow_kg_s',
    'gcv_position_pct',
    'igv_position_deg',
    'ssrv_position_pct',
    'gt_exhaust_temp_c',
    'exhaust_duct_temp_c',
    'wheel_space_temp_avg_c',
    'compressor_inlet_temp_c',
    'compressor_discharge_pressure_bar',
    'generator_power_mw'
]

# Target columns (CEMS emissions)
TARGET_COLS = [
    'nox_mg_nm3',
    'sox_mg_nm3',
    'co_mg_nm3',
    'pm_mg_nm3'
]

# Emission display names
EMISSION_NAMES = {
    'nox_mg_nm3': 'NOx',
    'sox_mg_nm3': 'SOx',
    'co_mg_nm3': 'CO',
    'pm_mg_nm3': 'PM'
}

# Enable/Disable hyperparameter tuning (set to True for better results, slower)
ENABLE_TUNING = True


def load_and_merge_data():
    """Load GT operation and CEMS data, merge on timestamp."""
    print("Loading data...")
    
    gt_path = os.path.join(DATA_DIR, 'gt_operation.csv')
    cems_path = os.path.join(DATA_DIR, 'cems_data.csv')
    
    gt_data = pd.read_csv(gt_path, parse_dates=['timestamp'])
    cems_data = pd.read_csv(cems_path, parse_dates=['timestamp'])
    
    print(f"  GT Operation: {len(gt_data)} samples")
    print(f"  CEMS Data: {len(cems_data)} samples")
    
    # Merge on timestamp
    merged = pd.merge(gt_data, cems_data, on='timestamp', how='inner')
    print(f"  Merged: {len(merged)} samples")
    
    return merged


def add_time_features(df):
    """Add cyclical time features from timestamp."""
    print("\nAdding time features...")
    
    df = df.copy()
    
    # Extract hour of day
    hour = df['timestamp'].dt.hour
    
    # Cyclical encoding (captures daily patterns)
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Day of week (0=Monday, 6=Sunday)
    day_of_week = df['timestamp'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    print(f"  Added: hour_sin, hour_cos, day_sin, day_cos")
    
    return df


def add_lag_features(df, lag_cols, lags=[1, 2, 3]):
    """Add lagged versions of specified columns."""
    print("\nAdding lag features...")
    
    df = df.copy()
    new_cols = []
    
    for col in lag_cols:
        for lag in lags:
            new_col = f'{col}_lag{lag}'
            df[new_col] = df[col].shift(lag)
            new_cols.append(new_col)
    
    print(f"  Added {len(new_cols)} lag features for {len(lag_cols)} columns")
    
    return df


def add_rolling_features(df, rolling_cols, window=3):
    """Add rolling mean and std for specified columns."""
    print("\nAdding rolling features...")
    
    df = df.copy()
    new_cols = []
    
    for col in rolling_cols:
        # Rolling mean
        mean_col = f'{col}_roll{window}_mean'
        df[mean_col] = df[col].rolling(window=window, min_periods=1).mean()
        new_cols.append(mean_col)
        
        # Rolling std
        std_col = f'{col}_roll{window}_std'
        df[std_col] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
        new_cols.append(std_col)
    
    print(f"  Added {len(new_cols)} rolling features (window={window}h)")
    
    return df


def preprocess_data(df):
    """Handle missing values and prepare features/targets."""
    print("\nPreprocessing data...")
    
    # Report missing values
    missing = df.isnull().sum()
    if missing.any():
        print("  Missing values found:")
        for col, count in missing[missing > 0].items():
            if count > 0:
                print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # Forward fill for time series continuity
    df_clean = df.copy()
    df_clean = df_clean.ffill()
    
    # Backward fill for any remaining NaNs at start
    df_clean = df_clean.bfill()
    
    # Verify no missing values remain
    remaining_missing = df_clean.isnull().sum().sum()
    print(f"  After handling: {remaining_missing} missing values")
    
    return df_clean


def engineer_features(df):
    """Apply all feature engineering steps."""
    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING")
    print("=" * 50)
    
    # 1. Add time features
    df = add_time_features(df)
    
    # 2. Add lag features for key columns
    lag_cols = ['fuel_gas_flow_kg_s', 'gt_exhaust_temp_c', 'generator_power_mw']
    df = add_lag_features(df, lag_cols, lags=[1, 2, 3])
    
    # 3. Add rolling features
    rolling_cols = ['fuel_gas_flow_kg_s', 'gt_exhaust_temp_c', 'compressor_inlet_temp_c']
    df = add_rolling_features(df, rolling_cols, window=3)
    
    return df


def get_feature_columns(df):
    """Get all feature columns (original + engineered)."""
    # Start with base features
    feature_cols = BASE_FEATURE_COLS.copy()
    
    # Add time features
    feature_cols.extend(['hour_sin', 'hour_cos', 'day_sin', 'day_cos'])
    
    # Add lag features
    for col in ['fuel_gas_flow_kg_s', 'gt_exhaust_temp_c', 'generator_power_mw']:
        for lag in [1, 2, 3]:
            feature_cols.append(f'{col}_lag{lag}')
    
    # Add rolling features
    for col in ['fuel_gas_flow_kg_s', 'gt_exhaust_temp_c', 'compressor_inlet_temp_c']:
        feature_cols.append(f'{col}_roll3_mean')
        feature_cols.append(f'{col}_roll3_std')
    
    # Filter to only existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    return feature_cols


def split_data(df, feature_cols):
    """Time-based split into training and testing sets."""
    print("\nSplitting data (time-based, no shuffling)...")
    
    # Remove first few rows affected by lag/rolling
    df = df.iloc[3:].reset_index(drop=True)
    
    train_df = df.iloc[:TRAINING_HOURS].copy()
    test_df = df.iloc[TRAINING_HOURS:TRAINING_HOURS + TESTING_HOURS].copy()
    
    print(f"  Training set: {len(train_df)} samples")
    print(f"  Testing set: {len(test_df)} samples")
    print(f"  Total features: {len(feature_cols)}")
    
    return train_df, test_df


def scale_features(train_df, test_df, feature_cols):
    """Standardize features using training set statistics."""
    print("\nScaling features...")
    
    scaler = StandardScaler()
    
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_train = train_df[TARGET_COLS].values
    y_test = test_df[TARGET_COLS].values
    
    print(f"  Features scaled (mean=0, std=1)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def tune_hyperparameters(X_train, y_train_single, emission_name):
    """Use RandomizedSearchCV to find optimal hyperparameters."""
    print(f"    Tuning hyperparameters for {emission_name}...")
    
    # Parameter grid
    param_dist = {
        'n_estimators': [100, 150, 200, 250],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 2.0]
    }
    
    # Base model
    base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Randomized search
    search = RandomizedSearchCV(
        base_model,
        param_dist,
        n_iter=20,
        cv=tscv,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    search.fit(X_train, y_train_single)
    
    print(f"      Best params: max_depth={search.best_params_['max_depth']}, "
          f"lr={search.best_params_['learning_rate']:.2f}, "
          f"n_est={search.best_params_['n_estimators']}")
    
    return search.best_estimator_


def train_xgboost_models(X_train, y_train, tune=True):
    """Train XGBoost regressors with optional hyperparameter tuning."""
    print("\n" + "=" * 50)
    print("TRAINING XGBOOST MODELS")
    print("=" * 50)
    
    if tune:
        print("  Mode: Hyperparameter Tuning (RandomizedSearchCV)")
    else:
        print("  Mode: Default Parameters")
    
    models = {}
    
    for i, target in enumerate(TARGET_COLS):
        emission_name = EMISSION_NAMES[target]
        print(f"\n  Training model for {emission_name}...")
        
        if tune:
            model = tune_hyperparameters(X_train, y_train[:, i], emission_name)
        else:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train[:, i])
        
        models[target] = model
    
    print("\n  All models trained successfully!")
    return models


def evaluate_models(models, X_test, y_test):
    """Evaluate models and compute RMSE/MAE."""
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    predictions = {}
    metrics = {}
    
    for i, target in enumerate(TARGET_COLS):
        emission_name = EMISSION_NAMES[target]
        model = models[target]
        
        y_pred = model.predict(X_test)
        y_true = y_test[:, i]
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        predictions[target] = y_pred
        metrics[target] = {'rmse': rmse, 'mae': mae, 'r2': r2}
        
        # Color code R² for visibility
        if r2 >= 0.7:
            status = "✓ GOOD"
        elif r2 >= 0.5:
            status = "~ MODERATE"
        else:
            status = "✗ NEEDS IMPROVEMENT"
        
        print(f"\n{emission_name}: {status}")
        print(f"  RMSE: {rmse:.4f} mg/Nm³")
        print(f"  MAE:  {mae:.4f} mg/Nm³")
        print(f"  R²:   {r2:.4f}")
    
    return predictions, metrics


def extract_feature_importance(models, feature_cols):
    """Extract and rank feature importance for each model."""
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    importance_df = pd.DataFrame(index=feature_cols)
    
    for target in TARGET_COLS:
        emission_name = EMISSION_NAMES[target]
        model = models[target]
        importance = model.feature_importances_
        importance_df[emission_name] = importance
    
    # Print ranked importance for each emission
    for target in TARGET_COLS:
        emission_name = EMISSION_NAMES[target]
        print(f"\n{emission_name} - Top 5 Features:")
        sorted_importance = importance_df[emission_name].sort_values(ascending=False)
        for j, (feature, score) in enumerate(sorted_importance.head(5).items()):
            print(f"  {j+1}. {feature}: {score:.4f}")
    
    return importance_df


def estimate_carbon_footprint(df, predictions):
    """Estimate CO2 equivalent emissions."""
    print("\n" + "=" * 50)
    print("CARBON FOOTPRINT ESTIMATION")
    print("=" * 50)
    
    CO2_PER_KG_FUEL = 2.75
    GWP_NOX = 298
    GWP_CO = 1.9
    GWP_PM = 100
    
    # Adjust for removed rows
    test_start = TRAINING_HOURS + 3
    test_end = test_start + TESTING_HOURS
    test_df = df.iloc[test_start:test_end].copy()
    
    fuel_flow_kg_s = test_df['fuel_gas_flow_kg_s'].values
    hours = len(test_df)
    total_fuel_kg = np.sum(fuel_flow_kg_s) * 3600
    direct_co2_kg = total_fuel_kg * CO2_PER_KG_FUEL
    
    FLUE_GAS_FLOW_NM3_S = 500
    
    def emission_to_kg(emission_mg_nm3):
        total_nm3 = FLUE_GAS_FLOW_NM3_S * 3600 * hours
        total_mg = np.sum(emission_mg_nm3) * total_nm3 / len(emission_mg_nm3)
        return total_mg / 1e6
    
    nox_kg = emission_to_kg(predictions['nox_mg_nm3'])
    co_kg = emission_to_kg(predictions['co_mg_nm3'])
    pm_kg = emission_to_kg(predictions['pm_mg_nm3'])
    
    nox_co2e = nox_kg * GWP_NOX
    co_co2e = co_kg * GWP_CO
    pm_co2e = pm_kg * GWP_PM
    
    total_co2e_kg = direct_co2_kg + nox_co2e + co_co2e + pm_co2e
    total_co2e_tonnes = total_co2e_kg / 1000
    
    print(f"\nTest Period: {hours} hours")
    print(f"Total Fuel: {total_fuel_kg:,.0f} kg")
    print(f"\nBreakdown:")
    print(f"  Direct CO2: {direct_co2_kg:,.0f} kg ({direct_co2_kg/total_co2e_kg*100:.1f}%)")
    print(f"  NOx→CO2e:   {nox_co2e:,.0f} kg ({nox_co2e/total_co2e_kg*100:.1f}%)")
    print(f"  CO→CO2e:    {co_co2e:,.0f} kg ({co_co2e/total_co2e_kg*100:.1f}%)")
    print(f"  PM→CO2e:    {pm_co2e:,.0f} kg ({pm_co2e/total_co2e_kg*100:.1f}%)")
    print(f"\n  TOTAL: {total_co2e_tonnes:,.2f} tonnes CO2e")
    
    return {
        'total_fuel_kg': total_fuel_kg,
        'direct_co2_kg': direct_co2_kg,
        'nox_co2e_kg': nox_co2e,
        'co_co2e_kg': co_co2e,
        'pm_co2e_kg': pm_co2e,
        'total_co2e_tonnes': total_co2e_tonnes
    }


def plot_results(test_df, y_test, predictions, metrics, importance_df, carbon_data, feature_cols):
    """Generate visualization plots."""
    print("\nGenerating plots...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Prediction vs Actual (4 plots)
    for i, target in enumerate(TARGET_COLS):
        ax = fig.add_subplot(3, 4, i + 1)
        emission_name = EMISSION_NAMES[target]
        
        y_true = y_test[:, i]
        y_pred = predictions[target]
        
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2, label='Perfect')
        
        ax.set_xlabel(f'Actual {emission_name}')
        ax.set_ylabel(f'Predicted {emission_name}')
        ax.set_title(f'{emission_name}: R² = {metrics[target]["r2"]:.3f}')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # 2. Time series (4 plots)
    time_idx = range(len(y_test))
    for i, target in enumerate(TARGET_COLS):
        ax = fig.add_subplot(3, 4, i + 5)
        emission_name = EMISSION_NAMES[target]
        
        ax.plot(time_idx, y_test[:, i], 'b-', alpha=0.7, label='Actual', linewidth=1)
        ax.plot(time_idx, predictions[target], 'r-', alpha=0.7, label='Predicted', linewidth=1)
        
        ax.set_xlabel('Hour')
        ax.set_ylabel(f'{emission_name} (mg/Nm³)')
        ax.set_title(f'{emission_name} Time Series')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Top 10 Features for NOx (most important model)
    ax = fig.add_subplot(3, 4, 9)
    nox_importance = importance_df['NOx'].sort_values(ascending=True).tail(10)
    colors = ['#3498db' if 'lag' in f or 'roll' in f or 'sin' in f or 'cos' in f 
              else '#e74c3c' for f in nox_importance.index]
    ax.barh(range(len(nox_importance)), nox_importance.values, color=colors)
    ax.set_yticks(range(len(nox_importance)))
    ax.set_yticklabels([f.replace('_', '\n') for f in nox_importance.index], fontsize=7)
    ax.set_xlabel('Importance')
    ax.set_title('NOx: Top 10 Features\n(Blue=Engineered, Red=Original)')
    
    # 4. Error metrics
    ax = fig.add_subplot(3, 4, 10)
    x = range(len(TARGET_COLS))
    rmse_vals = [metrics[t]['rmse'] for t in TARGET_COLS]
    mae_vals = [metrics[t]['mae'] for t in TARGET_COLS]
    width = 0.35
    
    ax.bar([i - width/2 for i in x], rmse_vals, width, label='RMSE', color='steelblue')
    ax.bar([i + width/2 for i in x], mae_vals, width, label='MAE', color='coral')
    
    ax.set_ylabel('Error (mg/Nm³)')
    ax.set_title('Model Error Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels([EMISSION_NAMES[t] for t in TARGET_COLS])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Carbon Footprint
    ax = fig.add_subplot(3, 4, 11)
    labels = ['Direct CO2', 'NOx', 'CO', 'PM']
    sizes = [carbon_data['direct_co2_kg'], carbon_data['nox_co2e_kg'],
             carbon_data['co_co2e_kg'], carbon_data['pm_co2e_kg']]
    colors = ['#ff6b6b', '#ffa502', '#2ed573', '#3742fa']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Carbon Footprint\n({carbon_data["total_co2e_tonnes"]:.1f} tonnes)')
    
    # 6. R² comparison
    ax = fig.add_subplot(3, 4, 12)
    r2_vals = [metrics[t]['r2'] for t in TARGET_COLS]
    colors = ['#2ecc71' if r > 0.7 else '#f39c12' if r > 0.5 else '#e74c3c' for r in r2_vals]
    ax.bar([EMISSION_NAMES[t] for t in TARGET_COLS], r2_vals, color=colors)
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate')
    ax.set_ylabel('R² Score')
    ax.set_title('Model Accuracy')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'evaluation_plots.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    
    plt.close()


def save_models(models):
    """Save trained models."""
    print("\nSaving models...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for target in TARGET_COLS:
        emission_name = EMISSION_NAMES[target].lower()
        model_path = os.path.join(OUTPUT_DIR, f'model_{emission_name}.json')
        models[target].save_model(model_path)
        print(f"  Saved: {model_path}")


def main():
    print("=" * 60)
    print("XGBoost GT Emissions Model - IMPROVED VERSION")
    print("=" * 60)
    print("Improvements: Time features, Lag features, Hyperparameter tuning")
    print("=" * 60)
    
    # 1. Load data
    df = load_and_merge_data()
    
    # 2. Preprocess
    df_clean = preprocess_data(df)
    
    # 3. Feature engineering
    df_engineered = engineer_features(df_clean)
    
    # 4. Get all feature columns
    feature_cols = get_feature_columns(df_engineered)
    
    # 5. Handle any remaining NaN from lag/rolling
    df_engineered = df_engineered.ffill().bfill()
    
    # 6. Split
    train_df, test_df = split_data(df_engineered, feature_cols)
    
    # 7. Scale features
    X_train, X_test, y_train, y_test, scaler = scale_features(train_df, test_df, feature_cols)
    
    # 8. Train with tuning
    models = train_xgboost_models(X_train, y_train, tune=ENABLE_TUNING)
    
    # 9. Evaluate
    predictions, metrics = evaluate_models(models, X_test, y_test)
    
    # 10. Feature importance
    importance_df = extract_feature_importance(models, feature_cols)
    
    # 11. Carbon footprint
    carbon_data = estimate_carbon_footprint(df_engineered, predictions)
    
    # 12. Plots
    plot_results(test_df, y_test, predictions, metrics, importance_df, carbon_data, feature_cols)
    
    # 13. Save models
    save_models(models)
    
    print("\n" + "=" * 60)
    print("IMPROVED PIPELINE COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
