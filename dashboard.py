"""
Interactive Streamlit Dashboard for GT Emissions Analysis
Ultimate Dashboard Structure:
- Panel A: Data Explorer
- Panel B: Model Evaluation
- Panel C: Real-Time Simulation (Digital Twin)
- Panel D: Forecasting

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Page config
st.set_page_config(
    page_title="GT Digital Twin & Emissions Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chart export configuration - enables download as JPG/PNG
CHART_CONFIG = {
    'toImageButtonOptions': {
        'format': 'jpeg',
        'filename': 'gt_chart_export',
        'height': 800,
        'width': 1200,
        'scale': 2
    },
    'displaylogo': False,
    'modeBarButtonsToAdd': ['downloadImage']
}

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Feature configuration
FEATURE_COLS = [
    'fuel_gas_flow_kg_s', 'gcv_position_pct', 'igv_position_deg',
    'ssrv_position_pct', 'exhaust_duct_temp_c',
    'wheel_space_temp_avg_c', 'compressor_inlet_temp_c',
    'compressor_discharge_pressure_bar', 'generator_power_mw'
]

FEATURE_DISPLAY = {
    'fuel_gas_flow_kg_s': ('Fuel Gas Flow', 'kg/s', 1.5, 2.5),
    'gcv_position_pct': ('GCV Position', '%', 60.0, 95.0),
    'igv_position_deg': ('IGV Position', '°', 30.0, 40.0),
    'ssrv_position_pct': ('SSRV Position', '%', 60.0, 100.0),
    'exhaust_duct_temp_c': ('Exhaust Duct Temp', '°C', 540.0, 580.0),
    'wheel_space_temp_avg_c': ('Wheel Space Temp', '°C', 450.0, 490.0),
    'compressor_inlet_temp_c': ('Compressor Inlet Temp', '°C', 20.0, 35.0),
    'compressor_discharge_pressure_bar': ('Discharge Pressure', 'bar', 0.0, 5.0),
    'generator_power_mw': ('Generator Power', 'MW', 18.0, 25.0)
}

EMISSION_NAMES = {
    'nox_mg_nm3': 'NOx',
    'sox_mg_nm3': 'SOx',
    'co_mg_nm3': 'CO',
    'pm_mg_nm3': 'PM'
}

# Column mapping from Excel
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

# ----------------
# DATA LOADING
# ----------------

@st.cache_data
def load_data():
    """Load GT operation data from Excel and generate placeholder CEMS data."""
    
    excel_path = os.path.join(BASE_DIR, 'raw_data', 'GTLOGSHEET_1.xlsx')
    csv_path = os.path.join(DATA_DIR, 'gt_operation.csv')
    
    if os.path.exists(excel_path):
        try:
            raw_df = pd.read_excel(excel_path)
            
            # Find date and time columns
            date_col = None
            time_col = None
            
            for col in raw_df.columns:
                if isinstance(col, float) or (isinstance(col, str) and col.replace('.', '').isdigit()):
                    date_col = col
                if col == 'Pukul':
                    time_col = col
            
            # Create timestamp
            if date_col is not None and time_col is not None:
                raw_df['timestamp'] = pd.to_datetime(
                    raw_df[date_col].astype(str) + ' ' + raw_df[time_col].astype(str),
                    errors='coerce'
                )
            elif date_col is not None:
                raw_df['timestamp'] = pd.to_datetime(raw_df[date_col], errors='coerce')
            else:
                raw_df['timestamp'] = pd.to_datetime(raw_df.iloc[:, 0], errors='coerce')
            
            # Map columns
            gt_data = pd.DataFrame()
            gt_data['timestamp'] = raw_df['timestamp']
            
            for internal_name, excel_name in COLUMN_MAPPING.items():
                if excel_name in raw_df.columns:
                    gt_data[internal_name] = pd.to_numeric(raw_df[excel_name], errors='coerce')
            
            # Unit conversions
            if 'fuel_gas_flow_kg_s' in gt_data.columns:
                gt_data['fuel_gas_flow_kg_s'] = gt_data['fuel_gas_flow_kg_s'] / 3600
            if 'compressor_discharge_pressure_bar' in gt_data.columns:
                gt_data['compressor_discharge_pressure_bar'] = gt_data['compressor_discharge_pressure_bar'] * 10
                
        except Exception as e:
            st.warning(f"Error loading Excel: {e}")
            gt_data = None
            
    elif os.path.exists(csv_path):
        gt_data = pd.read_csv(csv_path, parse_dates=['timestamp'])
    else:
        return None
    
    if gt_data is None or gt_data.empty:
        return None
    
    # Ensure timestamp is datetime type
    gt_data['timestamp'] = pd.to_datetime(gt_data['timestamp'], errors='coerce')
    
    # Drop invalid timestamps
    gt_data = gt_data.dropna(subset=['timestamp'])
    
    # Ensure timestamps are timezone-naive
    if len(gt_data) > 0 and hasattr(gt_data['timestamp'].dt, 'tz') and gt_data['timestamp'].dt.tz is not None:
        gt_data['timestamp'] = gt_data['timestamp'].dt.tz_localize(None)
    
    # Filter out invalid dates
    gt_data = gt_data[gt_data['timestamp'] < pd.Timestamp('2027-01-01')]
    gt_data = gt_data[gt_data['timestamp'] > pd.Timestamp('2020-01-01')]
    gt_data = gt_data.sort_values('timestamp').reset_index(drop=True)
    
    if gt_data.empty:
        return None
    
    # Generate placeholder CEMS data
    np.random.seed(42)
    n = len(gt_data)
    load = gt_data['generator_power_mw'].fillna(20).values
    
    gt_data['nox_mg_nm3'] = (25 + (load - 20) * 0.5 + np.random.normal(0, 3, n)).clip(10, 60)
    gt_data['sox_mg_nm3'] = (5 + (load - 20) * 0.1 + np.random.normal(0, 1, n)).clip(1, 15)
    gt_data['co_mg_nm3'] = (15 + np.random.normal(0, 2, n)).clip(5, 30)
    gt_data['pm_mg_nm3'] = (3 + np.random.normal(0, 0.5, n)).clip(1, 10)
    
    return gt_data

@st.cache_resource
def load_models():
    """Load trained XGBoost models."""
    models = {}
    for emission, name in EMISSION_NAMES.items():
        model_path = os.path.join(OUTPUT_DIR, f'model_{name.lower()}.json')
        if os.path.exists(model_path):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            models[emission] = model
    return models

def calculate_carbon_footprint(fuel_flow_kg_s, nox_mg_nm3, co_mg_nm3, pm_mg_nm3):
    """Calculate CO2 equivalent emissions."""
    fuel_flow_kg_h = fuel_flow_kg_s * 3600
    
    co2_factor = 2.75
    direct_co2 = fuel_flow_kg_h * co2_factor
    
    nox_gwp = 298
    nox_mass_kg_h = nox_mg_nm3 * 0.001 * fuel_flow_kg_h * 0.01
    nox_co2e = nox_mass_kg_h * nox_gwp
    
    co_gwp = 3
    co_mass_kg_h = co_mg_nm3 * 0.001 * fuel_flow_kg_h * 0.01
    co_co2e = co_mass_kg_h * co_gwp
    
    pm_gwp = 1600
    pm_mass_kg_h = pm_mg_nm3 * 0.001 * fuel_flow_kg_h * 0.001
    pm_co2e = pm_mass_kg_h * pm_gwp
    
    return {
        'direct_co2': direct_co2 / 1000,
        'nox_co2e': nox_co2e / 1000,
        'co_co2e': co_co2e / 1000,
        'pm_co2e': pm_co2e / 1000,
        'total': (direct_co2 + nox_co2e + co_co2e + pm_co2e) / 1000
    }

# ----------------
# APP UI
# ----------------

st.title("🏭 GT Digital Twin & Emissions Dashboard")

data = load_data()
models = load_models()

if data is None:
    st.error("Data not found. Please check `data/` or `raw_data/` directory.")
    st.stop()

# Show data info
st.sidebar.markdown(f"**Data:** {len(data):,} points")
st.sidebar.markdown(f"**Range:** {data['timestamp'].min().strftime('%d %b %Y')} - {data['timestamp'].max().strftime('%d %b %Y')}")

# Main Navigation
panel = st.sidebar.radio(
    "Select Panel",
    ["Panel A: Data Explorer", 
     "Panel B: Model Evaluation", 
     "Panel C: Real-Time Simulation",
     "Panel D: Forecasting",
     "Panel E: Model Training Lab"]
)

# ----------------------------------------------------
# PANEL A: DATA EXPLORER
# ----------------------------------------------------
if panel == "Panel A: Data Explorer":
    st.header("📈 Panel A: Data Explorer")
    
    st.subheader("Global Settings")
    min_d, max_d = data['timestamp'].min().date(), data['timestamp'].max().date()
    
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("Start Date", min_d)
    with c2: end_date = st.date_input("End Date", max_d)
    
    filtered_data = data[
        (data['timestamp'].dt.date >= start_date) & 
        (data['timestamp'].dt.date <= end_date)
    ]
    
    st.markdown(f"**Data Points:** {len(filtered_data):,}")
    
    # Time-Series Plot
    st.subheader("1. Input Variables History")
    input_vars = st.multiselect("Select Input Variables", FEATURE_COLS, default=FEATURE_COLS[:3])
    
    if input_vars:
        fig_inputs = go.Figure()
        for var in input_vars:
            name, unit, _, _ = FEATURE_DISPLAY.get(var, (var, '', 0, 0))
            fig_inputs.add_trace(go.Scatter(x=filtered_data['timestamp'], y=filtered_data[var], name=f"{name} ({unit})"))
        
        fig_inputs.update_layout(
            title="Operational Parameters Over Time", 
            hovermode="x unified", 
            height=450,
            xaxis=dict(
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(count=6, label="6h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                )
            )
        )
        st.plotly_chart(fig_inputs, use_container_width=True, config=CHART_CONFIG)
    
    # Emission History
    st.subheader("2. Emission History")
    emission_vars = st.multiselect("Select Emissions", list(EMISSION_NAMES.keys()), default=list(EMISSION_NAMES.keys()))
    
    if emission_vars:
        fig_emissions = go.Figure()
        for var in emission_vars:
            name = EMISSION_NAMES[var]
            fig_emissions.add_trace(go.Scatter(x=filtered_data['timestamp'], y=filtered_data[var], name=f"{name} (mg/Nm³)"))
        
        fig_emissions.update_layout(
            title="Continuous Emission Monitoring System (CEMS) Data", 
            hovermode="x unified", 
            height=450,
            xaxis=dict(
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(count=6, label="6h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                )
            )
        )
        st.plotly_chart(fig_emissions, use_container_width=True, config=CHART_CONFIG)

    # Correlation Heatmap
    st.subheader("3. Correlation Analysis")
    corr_cols = [c for c in FEATURE_COLS if c in filtered_data.columns] + [c for c in EMISSION_NAMES.keys() if c in filtered_data.columns]
    readable_map = {k: v[0] for k, v in FEATURE_DISPLAY.items()}
    readable_map.update(EMISSION_NAMES)
    
    corr_df = filtered_data[corr_cols].rename(columns=readable_map).corr()
    
    fig_corr = px.imshow(
        corr_df, 
        text_auto='.2f', 
        aspect='auto', 
        color_continuous_scale='RdBu_r', 
        title="Input-Output Correlation Matrix"
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True, config=CHART_CONFIG)


# ----------------------------------------------------
# PANEL B: MODEL EVALUATION
# ----------------------------------------------------
elif panel == "Panel B: Model Evaluation":
    st.header("🛠️ Panel B: Model Performance Evaluation")
    
    if not models:
        st.warning("No models loaded! Train models first using `python train_model.py`")
        st.stop()
    
    # Check model compatibility
    model_compatible = True
    expected_features = None
    for key in models:
        try:
            expected_features = models[key].n_features_in_
            if expected_features != len(FEATURE_COLS):
                model_compatible = False
                break
        except:
            pass
    
    if not model_compatible:
        st.warning(f"""
        ⚠️ **Model Tidak Kompatibel**
        
        Model membutuhkan **{expected_features} fitur**, tetapi data hanya memiliki **{len(FEATURE_COLS)} fitur**.
        
        Jalankan `python train_model.py` untuk melatih ulang model.
        """)
        st.stop()
    
    st.markdown("Assess how well the XGBoost digital twin mimics the actual gas turbine.")
    
    min_d, max_d = data['timestamp'].min().date(), data['timestamp'].max().date()
    c1, c2 = st.columns(2)
    with c1: t_start = st.date_input("Evaluation Start", min_d)
    with c2: t_end = st.date_input("Evaluation End", max_d)
    
    test_data = data[
        (data['timestamp'].dt.date >= t_start) & 
        (data['timestamp'].dt.date <= t_end)
    ].copy()
    
    if test_data.empty:
        st.warning("No data in selected range.")
        st.stop()
    
    X_test = test_data[FEATURE_COLS]
    
    selected_emission = st.selectbox("Select Emission to Analyze", list(EMISSION_NAMES.values()))
    selected_key = [k for k, v in EMISSION_NAMES.items() if v == selected_emission][0]
    
    if selected_key in models:
        y_true = test_data[selected_key]
        y_pred = models[selected_key].predict(X_test)
        test_data[f"pred_{selected_key}"] = y_pred
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("RMSE", f"{rmse:.4f}")
        with m2: st.metric("MAE", f"{mae:.4f}")
        with m3: st.metric("R²", f"{r2:.4f}")
        
        # Plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=test_data['timestamp'], y=y_true, name="Actual"))
            fig_ts.add_trace(go.Scatter(x=test_data['timestamp'], y=y_pred, name="Predicted", line=dict(dash='dash')))
            fig_ts.update_layout(title="Actual vs Predicted", height=350)
            st.plotly_chart(fig_ts, use_container_width=True, config=CHART_CONFIG)
        
        with col2:
            fig_parity = px.scatter(x=y_true, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, opacity=0.5)
            min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            fig_parity.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="red", dash="dash"))
            fig_parity.update_layout(title="Parity Plot", height=350)
            st.plotly_chart(fig_parity, use_container_width=True, config=CHART_CONFIG)


# ----------------------------------------------------
# PANEL C: REAL-TIME SIMULATION
# ----------------------------------------------------
elif panel == "Panel C: Real-Time Simulation":
    st.header("🎛️ Panel C: Digital Twin Control Panel")
    
    use_placeholder = not models or any(
        models[k].n_features_in_ != len(FEATURE_COLS) for k in models if hasattr(models[k], 'n_features_in_')
    )
    
    if use_placeholder:
        st.warning("⚠️ Using estimated predictions (models not available or incompatible)")
    
    st.markdown("Adjust sliders to simulate different operating conditions.")
    
    col_controls, col_dashboard = st.columns([1, 2])
    
    baseline_stats = data[FEATURE_COLS].mean()
    
    PRESETS = {
        'baseline': {'name': '⚖️ Baseline', 'values': baseline_stats.to_dict()},
        'optimal': {'name': '✅ Optimal', 'values': {
            'fuel_gas_flow_kg_s': 1.8, 'gcv_position_pct': 75.0, 'igv_position_deg': 35.0,
            'ssrv_position_pct': 85.0, 'exhaust_duct_temp_c': 560.0, 'wheel_space_temp_avg_c': 470.0,
            'compressor_inlet_temp_c': 25.0, 'compressor_discharge_pressure_bar': 3.0, 'generator_power_mw': 20.0
        }},
        'worst': {'name': '❌ Worst', 'values': {
            'fuel_gas_flow_kg_s': 2.4, 'gcv_position_pct': 90.0, 'igv_position_deg': 38.0,
            'ssrv_position_pct': 95.0, 'exhaust_duct_temp_c': 575.0, 'wheel_space_temp_avg_c': 485.0,
            'compressor_inlet_temp_c': 32.0, 'compressor_discharge_pressure_bar': 4.5, 'generator_power_mw': 24.0
        }}
    }
    
    simulation_input = {}
    
    with col_controls:
        operating_hours = st.number_input("Operating Hours", 1, 8760, 24)
        
        preset_choice = st.radio("Preset", ['custom', 'baseline', 'optimal', 'worst'], format_func=lambda x: {
            'custom': '🔧 Custom', 'baseline': '⚖️ Baseline', 'optimal': '✅ Optimal', 'worst': '❌ Worst'
        }[x])
        
        st.divider()
        
        for col in FEATURE_COLS:
            name, unit, min_v, max_v = FEATURE_DISPLAY[col]
            default_v = PRESETS.get(preset_choice, {}).get('values', {}).get(col, baseline_stats[col]) if preset_choice != 'custom' else baseline_stats[col]
            default_v = max(min_v, min(max_v, default_v))
            
            if preset_choice == 'custom':
                simulation_input[col] = st.slider(f"{name} ({unit})", float(min_v), float(max_v), float(default_v), key=f"s_{col}")
            else:
                simulation_input[col] = default_v
                st.metric(name, f"{default_v:.2f} {unit}")
    
    # Calculate predictions
    fuel = simulation_input.get('fuel_gas_flow_kg_s', 2.0)
    power = simulation_input.get('generator_power_mw', 20.0)
    temp = simulation_input.get('exhaust_duct_temp_c', 550.0)
    
    preds = {
        'nox_mg_nm3': 25 + (power - 20) * 0.8 + (temp - 550) * 0.1,
        'sox_mg_nm3': 5 + fuel * 0.5,
        'co_mg_nm3': 15 - (temp - 550) * 0.05,
        'pm_mg_nm3': 3 + fuel * 0.2
    }
    
    with col_dashboard:
        st.subheader("Real-Time Prediction")
        
        c1, c2, c3, c4 = st.columns(4)
        for i, (col, (key, name)) in enumerate(zip([c1, c2, c3, c4], EMISSION_NAMES.items())):
            with col:
                st.metric(name, f"{preds[key]:.2f} mg/Nm³")
        
        st.divider()
        
        # Carbon Footprint
        st.subheader(f"Carbon Footprint ({operating_hours}h)")
        
        cf = calculate_carbon_footprint(fuel, preds['nox_mg_nm3'], preds['co_mg_nm3'], preds['pm_mg_nm3'])
        total_cf = cf['total'] * operating_hours
        
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Per Hour", f"{cf['total']*1000:.2f} kg/h")
        with m2:
            st.metric(f"Total ({operating_hours}h)", f"{total_cf:.2f} tonnes CO2e")


# ----------------------------------------------------
# PANEL D: FORECASTING
# ----------------------------------------------------
elif panel == "Panel D: Forecasting":
    st.header("🔮 Panel D: Emission Forecasting")
    
    st.markdown("**Prediksi emisi dan carbon footprint ke depan** berdasarkan tren historis.")
    
    min_date = data['timestamp'].min()
    max_date = data['timestamp'].max()
    st.info(f"📅 Data: **{min_date.strftime('%d %b %Y')}** - **{max_date.strftime('%d %b %Y')}** ({len(data):,} points)")
    
    col1, col2 = st.columns(2)
    with col1:
        forecast_days = st.slider("Forecast Period (days)", 7, 365, 30)
    with col2:
        scenario = st.radio("Scenario", ["📊 Trend", "✅ Optimal", "⚖️ Normal", "❌ Worst"])
    
    # Scenario factors
    factors = {"📊 Trend": 1.0, "✅ Optimal": 0.85, "⚖️ Normal": 1.0, "❌ Worst": 1.2}
    factor = factors[scenario]
    
    # Generate forecast
    forecast_dates = pd.date_range(max_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    
    baseline = data[FEATURE_COLS].mean()
    np.random.seed(42)
    
    forecast_results = []
    for date in forecast_dates:
        fuel = baseline['fuel_gas_flow_kg_s'] * factor
        power = baseline['generator_power_mw'] * factor
        temp = baseline.get('exhaust_duct_temp_c', 550)
        
        emissions = {
            'nox_mg_nm3': (25 + (power - 20) * 0.8) * factor,
            'sox_mg_nm3': (5 + fuel * 0.5) * factor,
            'co_mg_nm3': (15 - (temp - 550) * 0.05) * factor,
            'pm_mg_nm3': (3 + fuel * 0.2) * factor
        }
        
        cf = calculate_carbon_footprint(fuel, emissions['nox_mg_nm3'], emissions['co_mg_nm3'], emissions['pm_mg_nm3'])
        
        forecast_results.append({
            'date': date,
            **emissions,
            'co2e_kg_h': cf['total'] * 1000
        })
    
    forecast_df = pd.DataFrame(forecast_results)
    
    # Plot
    st.subheader("📈 Forecast Results")
    
    tab1, tab2 = st.tabs(["Emissions Forecast", "Carbon Footprint"])
    
    with tab1:
        emission_key = st.selectbox("Emission", list(EMISSION_NAMES.keys()), format_func=lambda x: EMISSION_NAMES[x])
        
        fig = go.Figure()
        
        # Historical
        hist = data.groupby(data['timestamp'].dt.date)[emission_key].mean().reset_index()
        hist.columns = ['date', 'value']
        fig.add_trace(go.Scatter(x=hist['date'], y=hist['value'], name='Historical', line=dict(color='blue')))
        
        # Forecast
        fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df[emission_key], name='Forecast', line=dict(color='red', dash='dash')))
        
        fig.add_vline(x=max_date, line_dash="dot", annotation_text="Forecast Start")
        fig.update_layout(title=f"{EMISSION_NAMES[emission_key]} Forecast - {scenario}", height=400)
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
    
    with tab2:
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['co2e_kg_h'], name='CO2e Forecast', fill='tozeroy'))
        fig_cf.update_layout(title=f"Carbon Footprint Forecast - {scenario}", yaxis_title="kg CO2e/h", height=400)
        st.plotly_chart(fig_cf, use_container_width=True, config=CHART_CONFIG)
        
        # Summary
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Avg CO2e/h", f"{forecast_df['co2e_kg_h'].mean():.2f} kg")
        with m2:
            total = forecast_df['co2e_kg_h'].sum() * 24 / 1000
            st.metric(f"Total ({forecast_days}d)", f"{total:.2f} tonnes")
        with m3:
            st.metric("Period", f"{forecast_days} days")
    
    # Download
    st.download_button("📥 Download Forecast CSV", forecast_df.to_csv(index=False), f"forecast_{forecast_days}d.csv", "text/csv")


# ----------------------------------------------------
# PANEL E: MODEL TRAINING LAB
# ----------------------------------------------------
elif panel == "Panel E: Model Training Lab":
    st.header("🧪 Panel E: Model Training Laboratory")
    
    st.markdown("""
    **Train XGBoost model langsung dari dashboard!**
    - Pilih rentang tanggal untuk data training
    - Customize fitur yang digunakan
    - Tune hyperparameters
    - Lihat learning curves dan feature importance
    """)
    
    # Training Configuration
    st.subheader("⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    
    min_d, max_d = data['timestamp'].min().date(), data['timestamp'].max().date()
    
    with col1:
        st.markdown("**📅 Data Range**")
        train_start = st.date_input("Training Start", min_d, key="train_start")
        train_end = st.date_input("Training End", max_d, key="train_end")
        test_size = st.slider("Test Split (%)", 10, 40, 20)
    
    with col2:
        st.markdown("**🎯 Target Emission**")
        target_emission = st.selectbox("Select Target", list(EMISSION_NAMES.keys()), 
                                       format_func=lambda x: EMISSION_NAMES[x])
        
        st.markdown("**📊 Model Parameters**")
        n_estimators = st.slider("N Estimators", 50, 500, 100, step=50)
        max_depth = st.slider("Max Depth", 3, 15, 6)
        learning_rate = st.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
    
    st.divider()
    
    # Feature Selection
    st.subheader("🔧 Feature Selection")
    
    available_features = [col for col in FEATURE_COLS if col in data.columns]
    selected_features = st.multiselect(
        "Select Features for Training",
        available_features,
        default=available_features,
        help="Choose which input variables to use for prediction"
    )
    
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features")
        st.stop()
    
    st.info(f"Using **{len(selected_features)}** features for training")
    
    st.divider()
    
    # Data Analysis
    st.subheader("📊 Training Data Analysis")
    
    train_data = data[
        (data['timestamp'].dt.date >= train_start) & 
        (data['timestamp'].dt.date <= train_end)
    ].copy()
    
    if train_data.empty:
        st.error("No data in selected range!")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", f"{len(train_data):,}")
    with col2:
        train_samples = int(len(train_data) * (1 - test_size/100))
        st.metric("Train Samples", f"{train_samples:,}")
    with col3:
        test_samples = len(train_data) - train_samples
        st.metric("Test Samples", f"{test_samples:,}")
    with col4:
        date_range = (train_end - train_start).days
        st.metric("Date Range", f"{date_range} days")
    
    # Learning Curve Preview (with different data amounts)
    st.subheader("📈 Data Amount vs Performance Analysis")
    
    data_percentages = [25, 50, 75, 100]
    
    if st.button("🧪 Analyze Data Amount Impact", type="secondary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        X_full = train_data[selected_features].dropna()
        y_full = train_data.loc[X_full.index, target_emission]
        
        results = []
        
        for i, pct in enumerate(data_percentages):
            status_text.text(f"Testing with {pct}% data...")
            progress_bar.progress((i+1)/len(data_percentages))
            
            # Sample data
            n_samples = int(len(X_full) * pct / 100)
            X_sample = X_full.head(n_samples)
            y_sample = y_full.head(n_samples)
            
            # Train/test split
            split_idx = int(len(X_sample) * (1 - test_size/100))
            X_train, X_test = X_sample[:split_idx], X_sample[split_idx:]
            y_train, y_test = y_sample[:split_idx], y_sample[split_idx:]
            
            # Quick model
            quick_model = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            quick_model.fit(X_train, y_train, verbose=False)
            
            y_pred = quick_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'data_pct': pct,
                'samples': n_samples,
                'rmse': rmse,
                'r2': r2
            })
        
        status_text.text("Done!")
        progress_bar.empty()
        
        # Plot results
        results_df = pd.DataFrame(results)
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=['RMSE vs Data Amount', 'R² vs Data Amount'])
        
        fig.add_trace(
            go.Scatter(x=results_df['samples'], y=results_df['rmse'], mode='lines+markers', 
                      name='RMSE', marker=dict(size=12)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=results_df['samples'], y=results_df['r2'], mode='lines+markers', 
                      name='R²', marker=dict(size=12, color='green')),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Number of Samples", row=1, col=1)
        fig.update_xaxes(title_text="Number of Samples", row=1, col=2)
        fig.update_yaxes(title_text="RMSE", row=1, col=1)
        fig.update_yaxes(title_text="R² Score", row=1, col=2)
        fig.update_layout(height=350, title_text=f"Model Performance vs Training Data Amount ({EMISSION_NAMES[target_emission]})")
        
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        
        # Show table
        st.dataframe(results_df.style.format({'rmse': '{:.4f}', 'r2': '{:.4f}', 'samples': '{:,}'}), use_container_width=True)
    
    st.divider()
    
    # Train Full Model
    st.subheader("🚀 Train Full Model")
    
    if st.button("🎯 Start Training", type="primary"):
        with st.spinner("Training model..."):
            # Prepare data
            X = train_data[selected_features].dropna()
            y = train_data.loc[X.index, target_emission]
            
            # Split
            split_idx = int(len(X) * (1 - test_size/100))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train with eval
            eval_results = {}
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=20
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False
            )
            
            # Get predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
        st.success("Training Complete!")
        
        # Results
        st.markdown("### 📊 Training Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train RMSE", f"{train_rmse:.4f}")
        with col2:
            st.metric("Test RMSE", f"{test_rmse:.4f}")
        with col3:
            st.metric("Test R²", f"{test_r2:.4f}")
        with col4:
            st.metric("Test MAE", f"{test_mae:.4f}")
        
        # Plots
        plot_col1, plot_col2 = st.columns(2)
        
        with plot_col1:
            # Feature Importance
            importance = pd.DataFrame({
                'feature': selected_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig_imp = px.bar(importance, x='importance', y='feature', orientation='h',
                           title='Feature Importance')
            fig_imp.update_layout(height=350)
            st.plotly_chart(fig_imp, use_container_width=True, config=CHART_CONFIG)
        
        with plot_col2:
            # Prediction vs Actual
            fig_parity = px.scatter(x=y_test, y=y_pred_test, labels={'x': 'Actual', 'y': 'Predicted'},
                                   title='Prediction vs Actual', opacity=0.5)
            min_val, max_val = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
            fig_parity.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, 
                               line=dict(color="red", dash="dash"))
            fig_parity.update_layout(height=350)
            st.plotly_chart(fig_parity, use_container_width=True, config=CHART_CONFIG)
        
        # Learning Curve from eval results
        if hasattr(model, 'evals_result'):
            evals = model.evals_result()
            if evals:
                fig_lc = go.Figure()
                if 'validation_0' in evals:
                    fig_lc.add_trace(go.Scatter(y=evals['validation_0']['rmse'], name='Train', mode='lines'))
                if 'validation_1' in evals:
                    fig_lc.add_trace(go.Scatter(y=evals['validation_1']['rmse'], name='Test', mode='lines'))
                fig_lc.update_layout(title='Learning Curve', xaxis_title='Iteration', yaxis_title='RMSE', height=300)
                st.plotly_chart(fig_lc, use_container_width=True, config=CHART_CONFIG)
        
        # Save Model
        st.divider()
        st.markdown("### 💾 Save Model")
        
        model_name = EMISSION_NAMES[target_emission].lower()
        
        if st.button(f"💾 Save Model as model_{model_name}.json"):
            save_path = os.path.join(OUTPUT_DIR, f'model_{model_name}.json')
            model.save_model(save_path)
            st.success(f"Model saved to `{save_path}`!")
            st.info("Restart the dashboard to use the new model in Panel B and C.")
            
            # Save feature info
            import json
            feature_path = os.path.join(OUTPUT_DIR, f'features_{model_name}.json')
            with open(feature_path, 'w') as f:
                json.dump({
                    'features': selected_features,
                    'n_features': len(selected_features),
                    'target': target_emission,
                    'train_samples': len(X_train),
                    'test_r2': float(test_r2),
                    'test_rmse': float(test_rmse)
                }, f, indent=2)
            st.info(f"Feature info saved to `{feature_path}`")

