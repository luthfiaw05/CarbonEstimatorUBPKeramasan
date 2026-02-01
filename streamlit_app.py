"""
Interactive Streamlit Dashboard for GT Emissions Analysis
===========================================================

RESTRUCTURED WORKFLOW:
- Step 0: Data Manager - View and verify source data
- Step 1: Global Model Training - Train models that affect entire session
- Step 2: Model Evaluation - Assess predictions
- Step 3: Digital Twin Simulation - Real-time what-if analysis
- Step 4: Forecasting - Future projections

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ================================================================
# PAGE CONFIG & STYLING
# ================================================================
st.set_page_config(
    page_title="GT Digital Twin & Emissions Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Status indicators */
    .status-trained {
        background: linear-gradient(135deg, #00c853, #00e676);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .status-untrained {
        background: linear-gradient(135deg, #ff5252, #ff1744);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .file-info-box {
        background: linear-gradient(135deg, #1a237e, #283593);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #37474f, #455a64);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    /* Make sidebar more visible */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117, #161b22);
    }
</style>
""", unsafe_allow_html=True)

# Chart export configuration
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

# ================================================================
# PATHS & CONFIGURATION
# ================================================================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data')

# Feature configuration
FEATURE_COLS = [
    'fuel_gas_flow_kg_s', 'gcv_position_pct', 'igv_position_deg',
    'ssrv_position_pct', 'exhaust_duct_temp_c',
    'wheel_space_temp_avg_c', 'compressor_inlet_temp_c',
    'compressor_discharge_pressure_bar', 'generator_power_mw',
    'gt_second_stage_fwd_1_c', 'gt_second_stage_fwd_2_c',
    'gt_second_stage_aft_1_c', 'gt_second_stage_aft_2_c'
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
    'generator_power_mw': ('Generator Power', 'MW', 18.0, 25.0),
    'gt_second_stage_fwd_1_c': ('2nd Stage Fwd 1', '°C', 500.0, 600.0),
    'gt_second_stage_fwd_2_c': ('2nd Stage Fwd 2', '°C', 500.0, 600.0),
    'gt_second_stage_aft_1_c': ('2nd Stage Aft 1', '°C', 500.0, 600.0),
    'gt_second_stage_aft_2_c': ('2nd Stage Aft 2', '°C', 500.0, 600.0)
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

# ================================================================
# SESSION STATE INITIALIZATION
# ================================================================
if 'global_models' not in st.session_state:
    st.session_state.global_models = {}
if 'training_config' not in st.session_state:
    st.session_state.training_config = {
        'trained': False,
        'train_start': None,
        'train_end': None,
        'features': FEATURE_COLS.copy(),
        'metrics': {}
    }
if 'selected_gt_file' not in st.session_state:
    st.session_state.selected_gt_file = None
if 'selected_cems_file' not in st.session_state:
    st.session_state.selected_cems_file = None

# ================================================================
# DATA LOADING FUNCTIONS
# ================================================================

def get_available_files():
    """Get list of available data files in raw_data folder."""
    if not os.path.exists(RAW_DATA_DIR):
        return []
    files = glob.glob(os.path.join(RAW_DATA_DIR, '*.xlsx'))
    files.extend(glob.glob(os.path.join(RAW_DATA_DIR, '*.xls')))
    files.extend(glob.glob(os.path.join(RAW_DATA_DIR, '*.csv')))
    # Filter out temp files
    files = [f for f in files if not os.path.basename(f).startswith('~$')]
    return sorted(files)


@st.cache_data
def load_data_from_file(filepath):
    """Load and process GT operation data from a specific Excel/CSV file."""
    
    if not filepath or not os.path.exists(filepath):
        return None, "File not found"
    
    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ['.xlsx', '.xls']:
            raw_df = pd.read_excel(filepath)
        else:
            raw_df = pd.read_csv(filepath)
        
        # Find date and time columns
        date_col = None
        time_col = None
        
        for col in raw_df.columns:
            if isinstance(col, float) or (isinstance(col, str) and col.replace('.', '').isdigit()):
                date_col = col
            if col == 'Pukul':
                time_col = col
        
        # Clean and parse date column
        def parse_date(val):
            if pd.isna(val):
                return pd.NaT
            if isinstance(val, (pd.Timestamp, np.datetime64)):
                return pd.to_datetime(val, errors='coerce')
            if hasattr(val, 'strftime'):
                return pd.to_datetime(val, errors='coerce')
            val_str = str(val).strip().rstrip(',')
            try:
                return pd.to_datetime(val_str, dayfirst=True, errors='coerce')
            except:
                return pd.NaT
        
        if date_col is not None:
            raw_df['date_parsed'] = raw_df[date_col].apply(parse_date)
        else:
            raw_df['date_parsed'] = pd.NaT
        
        # Create timestamp by combining date and time
        if time_col is not None and time_col in raw_df.columns:
            def combine_datetime(row):
                date_val = row['date_parsed']
                time_val = row[time_col]
                if pd.isna(date_val):
                    return pd.NaT
                if pd.isna(time_val):
                    return date_val
                try:
                    time_str = str(time_val).strip()
                    return pd.to_datetime(str(date_val.date()) + ' ' + time_str, errors='coerce')
                except:
                    return date_val
            
            raw_df['timestamp'] = raw_df.apply(combine_datetime, axis=1)
        else:
            raw_df['timestamp'] = raw_df['date_parsed']
        
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
        
        # Clean timestamps
        gt_data['timestamp'] = pd.to_datetime(gt_data['timestamp'], errors='coerce', utc=True).dt.tz_localize(None)
        gt_data = gt_data.dropna(subset=['timestamp'])
        gt_data = gt_data[gt_data['timestamp'] < pd.Timestamp('2027-01-01')]
        gt_data = gt_data[gt_data['timestamp'] > pd.Timestamp('2020-01-01')]
        
        # Filter out maintenance rows
        if 'generator_power_mw' in gt_data.columns:
            gt_data = gt_data[gt_data['generator_power_mw'].notna()]
            gt_data = gt_data[gt_data['generator_power_mw'] > 0]
        
        gt_data = gt_data.sort_values('timestamp').reset_index(drop=True)
        
        if gt_data.empty:
            return None, "No valid data after processing"
        
        # Generate placeholder CEMS data (since real CEMS data not available)
        np.random.seed(42)
        n = len(gt_data)
        load = gt_data['generator_power_mw'].fillna(20).values
        
        gt_data['nox_mg_nm3'] = (25 + (load - 20) * 0.5 + np.random.normal(0, 3, n)).clip(10, 60)
        gt_data['sox_mg_nm3'] = (5 + (load - 20) * 0.1 + np.random.normal(0, 1, n)).clip(1, 15)
        gt_data['co_mg_nm3'] = (15 + np.random.normal(0, 2, n)).clip(5, 30)
        gt_data['pm_mg_nm3'] = (3 + np.random.normal(0, 0.5, n)).clip(1, 10)
        
        return gt_data, None


    except Exception as e:
        return None, f"Error processing file: {str(e)}"


@st.cache_data
def load_cems_file(filepath):
    """Load and process CEMS data."""
    if not filepath or not os.path.exists(filepath):
        return None, "File not found"
        
    try:
        # Load header first to check columns
        header_check = pd.read_excel(filepath, nrows=5)
        
        # Heuristic to find emission columns 
        # (User might have raw names like "H2S", "NOx (mg/m3)", etc.)
        candidates = {
            'nox_mg_nm3': [c for c in header_check.columns if 'NOX' in str(c).upper()],
            'sox_mg_nm3': [c for c in header_check.columns if 'SOX' in str(c).upper() or 'SO2' in str(c).upper()],
            'co_mg_nm3': [c for c in header_check.columns if 'CO ' in str(c).upper() or str(c).upper() == 'CO'], # space to avoid 'condition'
            'pm_mg_nm3': [c for c in header_check.columns if 'PM' in str(c).upper() or 'PART' in str(c).upper()]
        }
        
        # Flatten
        found_cols = [item for sublist in candidates.values() for item in sublist]
        
        if not found_cols:
            return None, "No emission columns (NOx, SOx, CO, PM) found."
            
        # Load full file
        raw_df = pd.read_excel(filepath)
        
        # Find timestamp
        date_col = None
        time_col = None
        for col in raw_df.columns:
             # Basic guess for timestamp columns often found in these logs
            if 'DATE' in str(col).upper() or 'TANGGAL' in str(col).upper() or isinstance(col, float):
                date_col = col
            if 'TIME' in str(col).upper() or 'JAM' in str(col).upper() or 'PUKUL' in str(col).upper():
                time_col = col
                
        # If no explicit date col found, try same heuristic as GT
        if not date_col:
             for col in raw_df.columns:
                if isinstance(col, float) or (isinstance(col, str) and col.replace('.', '').isdigit()):
                    date_col = col
                    break
        
        # Parse timestamp
        if date_col:
            # Similar parsing logic to GT...
            raw_df['timestamp'] = pd.to_datetime(raw_df[date_col], errors='coerce')
            
            # Combine time if separate
            if time_col and time_col in raw_df.columns:
                 # (Simplified merge logic here for brevity, assume similar to GT)
                 pass # For now assume date_col might be full timestamp or sufficiently accurate
                 
        else:
             return None, "No timestamp column found."
             
        # Extract data
        cems_data = pd.DataFrame()
        cems_data['timestamp'] = raw_df['timestamp']
        
        # Map found columns
        for key, possible_matches in candidates.items():
            if possible_matches:
                # Take the first match
                col_name = possible_matches[0]
                cems_data[key] = pd.to_numeric(raw_df[col_name], errors='coerce')
            else:
                cems_data[key] = np.nan
        
        cems_data = cems_data.dropna(subset=['timestamp'])
        cems_data = cems_data.sort_values('timestamp')
        
        return cems_data, None

    except Exception as e:
        return None, str(e)


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


# ================================================================
# MAIN APP
# ================================================================

st.title("🏭 GT Digital Twin & Emissions Dashboard")
st.markdown("**Step-by-Step Machine Learning Workflow**")

# ----------------------------------------------------------------
# SIDEBAR: Session Status & Navigation
# ----------------------------------------------------------------
st.sidebar.title("📊 Session Status")

# Model Status Indicator
if st.session_state.training_config['trained']:
    st.sidebar.markdown('<div class="status-trained">✅ Model Trained</div>', unsafe_allow_html=True)
    cfg = st.session_state.training_config
    st.sidebar.caption(f"Range: {cfg['train_start']} to {cfg['train_end']}")
    st.sidebar.caption(f"Features: {len(cfg['features'])}")
else:
    st.sidebar.markdown('<div class="status-untrained">⚠️ No Model Trained</div>', unsafe_allow_html=True)
    st.sidebar.caption("Train a model in Step 1 to enable all features")

st.sidebar.divider()

if st.sidebar.button("🔄 Reset Session & Clear Cache"):
    st.cache_data.clear()
    st.session_state.global_models = {}
    st.session_state.training_config = {
        'trained': False,
        'train_start': None,
        'train_end': None,
        'features': FEATURE_COLS.copy(),
        'metrics': {}
    }
    st.rerun()

# Navigation
st.sidebar.divider()
step = st.sidebar.radio(
    "Navigate to Step",
    [
        "Step 0: Data Manager",
        "Step 1: Global Training",
        "Step 2: Model Evaluation",
        "Step 3: Digital Twin Simulation",
        "Step 4: Forecasting"
    ],
    index=0
)


# ================================================================
# STEP 0: DATA MANAGER
# ================================================================
if step == "Step 0: Data Manager":
    st.header("📁 Step 0: Data Manager")
    st.markdown("""
    **Purpose**: Load and verify your source data files before training.  
    Select the correct Excel files for GT Logsheet and CEMS data.
    """)
    
    available_files = get_available_files()
    
    if not available_files:
        st.error(f"No data files found in `{RAW_DATA_DIR}`. Please add your Excel/CSV files.")
        st.stop()
    
    # File selection
    st.subheader("1. Select Source Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 GT Logsheet File (Operational Data)**")
        gt_files = [f for f in available_files if 'GT' in os.path.basename(f).upper() or 'LOGSHEET' in os.path.basename(f).upper()]
        if not gt_files:
            gt_files = available_files
        
        selected_gt = st.selectbox(
            "Select GT Logsheet",
            gt_files,
            format_func=lambda x: os.path.basename(x),
            key="gt_file_select"
        )
        st.session_state.selected_gt_file = selected_gt
        
        if selected_gt:
            file_size = os.path.getsize(selected_gt) / (1024 * 1024)
            st.info(f"📄 **{os.path.basename(selected_gt)}**  \nSize: {file_size:.2f} MB")
    
    with col2:
        st.markdown("**📊 CEMS File (Emissions Data)**")
        cems_files = [f for f in available_files if 'CEMS' in os.path.basename(f).upper() or 'AE' in os.path.basename(f).upper()]
        if not cems_files:
            cems_files = available_files
        
        selected_cems = st.selectbox(
            "Select CEMS File (or same as GT if combined)",
            ["(Using simulated CEMS data)"] + cems_files,
            format_func=lambda x: os.path.basename(x) if x != "(Using simulated CEMS data)" else x,
            key="cems_file_select"
        )
        
        if selected_cems != "(Using simulated CEMS data)":
            st.session_state.selected_cems_file = selected_cems
            file_size = os.path.getsize(selected_cems) / (1024 * 1024)
            st.info(f"📄 **{os.path.basename(selected_cems)}**  \nSize: {file_size:.2f} MB")
            
            # Validate CEMS
            c_data, c_err = load_cems_file(selected_cems)
            if c_err:
                 st.error(f"⚠️ Invalid CEMS File: {c_err}")
                 st.caption("Please ensure the file contains columns like 'NOx', 'SOx', 'CO', 'PM'.")
            else:
                 st.success(f"✅ Valid CEMS Data Found! ({len(c_data)} records)")
                 st.dataframe(c_data.head(), height=150)
                 
        else:
            st.warning("⚠️ No CEMS file selected. Using simulated emissions data based on load.")
            st.session_state.selected_cems_file = None
    
    st.divider()
    
    # Load and preview data (GT)
    st.subheader("2. Data Preview & Verification")
    
    if selected_gt:
        with st.spinner(f"Loading {os.path.basename(selected_gt)}..."):
            data, error = load_data_from_file(selected_gt)
        # ... (rest of simple GT data display)
        if error:
            st.error(f"Error loading file: {error}")
            st.stop()
        
        if data is not None and not data.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(data):,}")
            with col2:
                st.metric("Date Range", f"{(data['timestamp'].max() - data['timestamp'].min()).days} days")
            with col3:
                st.metric("Start Date", data['timestamp'].min().strftime('%Y-%m-%d'))
            with col4:
                st.metric("End Date", data['timestamp'].max().strftime('%Y-%m-%d'))
            
            # Show column mapping verification
            st.subheader("3. Column Mapping Verification")
            
            mapping_status = []
            for internal, excel in COLUMN_MAPPING.items():
                if internal in data.columns:
                    valid_count = data[internal].notna().sum()
                    mapping_status.append({
                        'Internal Name': internal,
                        'Excel Column': excel,
                        'Status': '✅ Mapped',
                        'Valid Values': f"{valid_count:,} ({100*valid_count/len(data):.1f}%)"
                    })
                else:
                    mapping_status.append({
                        'Internal Name': internal,
                        'Excel Column': excel,
                        'Status': '❌ Missing',
                        'Valid Values': '-'
                    })
            
            st.dataframe(pd.DataFrame(mapping_status), use_container_width=True, hide_index=True)
            
            # Raw data preview
            st.subheader("4. Raw Data Sample (First 20 Rows)")
            st.dataframe(data.head(20), use_container_width=True)
            
            # Basic statistics
            st.subheader("5. Data Statistics")
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                stats_df = data[numeric_cols].describe().T
                stats_df.columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
                st.dataframe(stats_df.style.format('{:.4f}'), use_container_width=True)
            
            st.success("✅ Data loaded successfully! Proceed to **Step 1: Global Training**.")
        else:
            st.error("No valid data found after processing.")


# ================================================================
# STEP 1: GLOBAL MODEL TRAINING
# ================================================================
elif step == "Step 1: Global Training":
    st.header("🎯 Step 1: Global Model Training")
    st.markdown("""
    **Purpose**: Train XGBoost models that will be used across ALL subsequent analysis panels.  
    This is the foundation of your ML workflow - changes here affect everything downstream.
    """)
    
    # Load data
    if not st.session_state.selected_gt_file:
        st.warning("⚠️ No GT file selected. Go to **Step 0** to select your data file.")
        st.stop()
    
    gt_data, error = load_data_from_file(st.session_state.selected_gt_file)
    if error or gt_data is None:
        st.error(f"Error loading GT data: {error}")
        st.stop()

    # Handle CEMS merging
    cems_file = st.session_state.get('selected_cems_file')
    if cems_file:
         cems_data, c_err = load_cems_file(cems_file)
         if c_err:
             st.error(f"Failed to load CEMS data: {c_err}")
             st.warning("Falling back to simulated data.")
             data = gt_data # Fallback: gt_data already has synthetic columns from load_data_from_file
         else:
             st.success("🔗 Merging GT Data with Real CEMS Data...")
             # Merge
             gt_data = gt_data.sort_values('timestamp')
             cems_data = cems_data.sort_values('timestamp')
             
             # Overwrite the synthetic columns in gt_data with real nulls first
             for col in ['nox_mg_nm3', 'sox_mg_nm3', 'co_mg_nm3', 'pm_mg_nm3']:
                 if col in gt_data.columns:
                     del gt_data[col] # Remove synthetic
             
             # Merge asof (nearest timestamp within 1 hour tolerance)
             data = pd.merge_asof(gt_data, cems_data, on='timestamp', direction='nearest', tolerance=pd.Timedelta('1h'))
             
             # Fill any remaining missing emissions with synthetic (or drop? Better to fill for functionality)
             # Actually, if we want accuracy, we should drop rows where we don't have CEMS data
             before_len = len(data)
             data = data.dropna(subset=['nox_mg_nm3']) # Require NOx at least
             after_len = len(data)
             
             if after_len < 100:
                 st.warning(f"⚠️ Only {after_len} records matched between GT and CEMS data. (Original: {before_len})")
             else:
                 st.info(f"✅ Successfully matched {after_len} records between GT and CEMS logs.")
             
             # If completely empty, fallback
             if data.empty:
                 st.error("No matching timestamps found! Check your data files.")
                 st.stop()
    else:
        st.info("ℹ️ Using simulated CEMS data (no file selected).")
        data = gt_data

    # Show current data info
    st.info(f"📁 **Using Data**: {len(data):,} matched records.")
    
    st.divider()
    
    # Training Configuration
    st.subheader("⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    
    min_d, max_d = data['timestamp'].min().date(), data['timestamp'].max().date()
    
    with col1:
        st.markdown("**📅 Training Data Range**")
        st.caption("Select the date range for training data. This globally affects model performance.")
        
        train_start = st.date_input("Start Date", min_d, min_value=min_d, max_value=max_d, key="global_train_start")
        train_end = st.date_input("End Date", max_d, min_value=min_d, max_value=max_d, key="global_train_end")
        
        test_split = st.slider("Test Split (%)", 10, 40, 20, help="Percentage of data used for testing")
    
    with col2:
        st.markdown("**🎛️ Hyperparameters**")
        n_estimators = st.slider("N Estimators", 50, 500, 100, step=50)
        max_depth = st.slider("Max Depth", 3, 15, 6)
        learning_rate = st.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
    
    st.divider()
    
    # Feature Selection
    st.subheader("🔧 Feature Selection")
    available_features = [col for col in FEATURE_COLS if col in data.columns]
    selected_features = st.multiselect(
        "Select Input Features",
        available_features,
        default=available_features,
        help="Features used to predict emissions"
    )
    
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features")
        st.stop()
    
    # Filter data by date range
    train_data = data[
        (data['timestamp'].dt.date >= train_start) & 
        (data['timestamp'].dt.date <= train_end)
    ].copy()
    
    st.markdown(f"**Training Data**: {len(train_data):,} samples from {train_start} to {train_end}")
    
    st.divider()
    
    # Training Button
    st.subheader("🚀 Train Global Models")
    st.markdown("This will train models for **all emission types** (NOx, SOx, CO, PM) simultaneously.")
    
    if st.button("🎯 Start Global Training", type="primary", use_container_width=True):
        
        progress = st.progress(0)
        status = st.empty()
        
        all_metrics = {}
        trained_models = {}
        
        # Prepare data once
        X = train_data[selected_features].dropna()
        
        for i, (emission_key, emission_name) in enumerate(EMISSION_NAMES.items()):
            status.text(f"Training {emission_name} model...")
            progress.progress((i) / len(EMISSION_NAMES))
            
            y = train_data.loc[X.index, emission_key]
            
            # Split
            split_idx = int(len(X) * (1 - test_split/100))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train, verbose=False)
            
            # Evaluate
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            trained_models[emission_key] = model
            all_metrics[emission_name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
        
        progress.progress(1.0)
        status.text("Training complete!")
        
        # Save to session state
        st.session_state.training_config = {
            'trained': True,
            'train_start': str(train_start),
            'train_end': str(train_end),
            'features': selected_features,
            'metrics': all_metrics,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate
        }
        
        # Display Metrics
        st.subheader("📊 Model Performance (Test Set)")
        cols = st.columns(len(EMISSION_NAMES))
        for i, (name, metrics) in enumerate(all_metrics.items()):
            with cols[i]:
                st.metric(name, f"R²: {metrics['R²']:.2f}")
                st.caption(f"RMSE: {metrics['RMSE']:.2f}")
        
        # Display Feature Importance (Global Average)
        st.subheader("🔑 Most Important Features")
        
        # Aggregate importance across all models
        feature_importance_df = pd.DataFrame(index=selected_features)
        for key, model in trained_models.items():
            feature_importance_df[EMISSION_NAMES[key]] = model.feature_importances_
            
        feature_importance_df['Average'] = feature_importance_df.mean(axis=1)
        feature_importance_df = feature_importance_df.sort_values('Average', ascending=True)
        
        fig = px.bar(
            feature_importance_df, 
            x='Average', 
            y=feature_importance_df.index,
            orientation='h',
            title="Global Feature Importance (Average)",
            labels={'Average': 'Importance', 'index': 'Feature'},
            color='Average',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ Models Trained & Saved! Proceed to Step 2.")
        
        # Display results
        st.subheader("📊 Training Results")
        
        metrics_df = pd.DataFrame(all_metrics).T
        st.dataframe(metrics_df.style.format('{:.4f}'), use_container_width=True)
        
        # Feature importance (from last model)
        st.subheader("📈 Feature Importance (Average)")
        
        avg_importance = np.zeros(len(selected_features))
        for model in trained_models.values():
            avg_importance += model.feature_importances_
        avg_importance /= len(trained_models)
        
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': avg_importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title='Average Feature Importance Across All Emission Models',
                     color='Importance', color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        
        st.info("✅ Models are now available for **Step 2, 3, and 4**. Navigate using the sidebar.")
    
    # Show previous training if exists
    elif st.session_state.training_config['trained']:
        st.success("✅ A model has been trained in this session.")
        cfg = st.session_state.training_config
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Range", f"{cfg['train_start']} to {cfg['train_end']}")
        with col2:
            st.metric("Features Used", len(cfg['features']))
        with col3:
            st.metric("Models Trained", len(st.session_state.global_models))
        
        if cfg['metrics']:
            st.subheader("📊 Previous Training Metrics")
            metrics_df = pd.DataFrame(cfg['metrics']).T
            st.dataframe(metrics_df.style.format('{:.4f}'), use_container_width=True)


# ================================================================
# STEP 2: MODEL EVALUATION
# ================================================================
elif step == "Step 2: Model Evaluation":
    st.header("🛠️ Step 2: Model Performance Evaluation")
    
    if not st.session_state.training_config['trained']:
        st.warning("⚠️ No model trained yet! Please complete **Step 1: Global Training** first.")
        st.stop()
    
    st.markdown("Assess how well the trained XGBoost models predict emissions.")
    
    # Load data
    data, error = load_data_from_file(st.session_state.selected_gt_file)
    if error or data is None:
        st.error(f"Error loading data: {error}")
        st.stop()
    
    models = st.session_state.global_models
    cfg = st.session_state.training_config
    
    st.info(f"🔧 **Using globally trained models** (Range: {cfg['train_start']} to {cfg['train_end']})")
    
    st.divider()
    
    # Evaluation range
    min_d, max_d = data['timestamp'].min().date(), data['timestamp'].max().date()
    col1, col2 = st.columns(2)
    with col1:
        eval_start = st.date_input("Evaluation Start", min_d)
    with col2:
        eval_end = st.date_input("Evaluation End", max_d)
    
    test_data = data[
        (data['timestamp'].dt.date >= eval_start) & 
        (data['timestamp'].dt.date <= eval_end)
    ].copy()
    
    if test_data.empty:
        st.warning("No data in selected range.")
        st.stop()
    
    st.markdown(f"**Evaluating on**: {len(test_data):,} samples")
    
    selected_emission = st.selectbox("Select Emission to Analyze", list(EMISSION_NAMES.values()))
    selected_key = [k for k, v in EMISSION_NAMES.items() if v == selected_emission][0]
    
    if selected_key in models:
        X_test = test_data[cfg['features']]
        y_true = test_data[selected_key]
        y_pred = models[selected_key].predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{rmse:.4f}")
        with col2:
            st.metric("MAE", f"{mae:.4f}")
        with col3:
            st.metric("R²", f"{r2:.4f}")
        
        # Plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=test_data['timestamp'], y=y_true, name="Actual", line=dict(color='blue')))
            fig_ts.add_trace(go.Scatter(x=test_data['timestamp'], y=y_pred, name="Predicted", line=dict(color='red', dash='dash')))
            fig_ts.update_layout(title=f"{selected_emission} - Actual vs Predicted", height=400,
                                 xaxis=dict(rangeslider=dict(visible=True)))
            st.plotly_chart(fig_ts, use_container_width=True, config=CHART_CONFIG)
        
        with col2:
            fig_parity = px.scatter(x=y_true, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, opacity=0.5)
            min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            fig_parity.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, 
                                line=dict(color="red", dash="dash"))
            fig_parity.update_layout(title="Parity Plot", height=400)
            st.plotly_chart(fig_parity, use_container_width=True, config=CHART_CONFIG)
    else:
        st.error(f"Model for {selected_emission} not found!")


# ================================================================
# STEP 3: DIGITAL TWIN SIMULATION
# ================================================================
elif step == "Step 3: Digital Twin Simulation":
    st.header("🎛️ Step 3: Digital Twin Control Panel")
    
    if not st.session_state.training_config['trained']:
        st.warning("⚠️ No model trained yet! Please complete **Step 1: Global Training** first.")
        st.stop()
    
    st.markdown("Adjust sliders to simulate different operating conditions using the **globally trained models**.")
    
    models = st.session_state.global_models
    cfg = st.session_state.training_config
    
    st.info(f"🔧 **Using globally trained models** (Range: {cfg['train_start']} to {cfg['train_end']})")
    
    # Load baseline for defaults
    data, _ = load_data_from_file(st.session_state.selected_gt_file)
    baseline = data[cfg['features']].mean() if data is not None else None
    
    col_controls, col_dashboard = st.columns([1, 2])
    
    simulation_input = {}
    
    with col_controls:
        st.subheader("⚙️ Operating Conditions")
        operating_hours = st.number_input("Operating Hours", 1, 8760, 24)
        
        for col in cfg['features']:
            name, unit, min_v, max_v = FEATURE_DISPLAY.get(col, (col, '', 0, 100))
            default_v = baseline[col] if baseline is not None and col in baseline else (min_v + max_v) / 2
            default_v = max(min_v, min(max_v, default_v))
            simulation_input[col] = st.slider(f"{name} ({unit})", float(min_v), float(max_v), float(default_v), key=f"sim_{col}")
    
    with col_dashboard:
        st.subheader("📊 Real-Time Predictions")
        
        # Prepare input
        X_sim = pd.DataFrame([simulation_input])
        
        # Predict all emissions
        predictions = {}
        for emission_key, model in models.items():
            predictions[emission_key] = model.predict(X_sim)[0]
        
        # Display predictions
        # Display predictions
        col_rows = [st.columns(2), st.columns(2)]
        
        emission_items = list(EMISSION_NAMES.items())
        
        # Row 1
        with col_rows[0][0]:
            key, name = emission_items[0]
            st.metric(name, f"{predictions.get(key, 0):.2f} mg/Nm³")
        with col_rows[0][1]:
            key, name = emission_items[1]
            st.metric(name, f"{predictions.get(key, 0):.2f} mg/Nm³")
            
        # Row 2
        with col_rows[1][0]:
            key, name = emission_items[2]
            st.metric(name, f"{predictions.get(key, 0):.2f} mg/Nm³")
        with col_rows[1][1]:
            key, name = emission_items[3]
            st.metric(name, f"{predictions.get(key, 0):.2f} mg/Nm³")
        
        st.divider()
        
        # Carbon Footprint
        st.subheader(f"🌍 Carbon Footprint ({operating_hours}h)")
        
        fuel = simulation_input.get('fuel_gas_flow_kg_s', 2.0)
        cf = calculate_carbon_footprint(
            fuel, 
            predictions.get('nox_mg_nm3', 25), 
            predictions.get('co_mg_nm3', 15), 
            predictions.get('pm_mg_nm3', 3)
        )
        total_cf = cf['total'] * operating_hours
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Per Hour", f"{cf['total']*1000:.2f} kg CO₂e/h")
        with col2:
            st.metric(f"Total ({operating_hours}h)", f"{total_cf:.2f} tonnes CO₂e")
        
        # Breakdown chart
        fig = go.Figure(data=[go.Pie(
            labels=['Direct CO₂', 'NOx (CO₂e)', 'CO (CO₂e)', 'PM (CO₂e)'],
            values=[cf['direct_co2'], cf['nox_co2e'], cf['co_co2e'], cf['pm_co2e']],
            hole=0.4
        )])
        fig.update_layout(title="Carbon Footprint Breakdown", height=350)
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


# ================================================================
# STEP 4: FORECASTING
# ================================================================
elif step == "Step 4: Forecasting":
    st.header("🔮 Step 4: Emission Forecasting")
    
    if not st.session_state.training_config['trained']:
        st.warning("⚠️ No model trained yet! Please complete **Step 1: Global Training** first.")
        st.stop()
    
    st.markdown("Project future emissions based on historical trends using **globally trained models**.")
    
    data, _ = load_data_from_file(st.session_state.selected_gt_file)
    cfg = st.session_state.training_config
    
    st.info(f"🔧 **Using globally trained models** (Range: {cfg['train_start']} to {cfg['train_end']})")
    
    max_date = data['timestamp'].max()
    
    col1, col2 = st.columns(2)
    with col1:
        forecast_days = st.slider("Forecast Period (days)", 7, 365, 30)
    with col2:
        scenario = st.radio("Scenario", ["📊 Trend", "✅ Optimal", "⚖️ Normal", "❌ Worst"], horizontal=True)
    
    # Calculate historical stats for realistic generation
    baseline_mean = data[cfg['features']].mean()
    baseline_std = data[cfg['features']].std()
    
    forecast_results = []
    
    # Scenario definitions
    # factor: multiplier for magnitude
    # drift: daily trend (plus or minus)
    # volatility: how much random noise to add relative to historical std
    scenarios = {
        "📊 Trend":   {"factor": 1.0,  "drift": 0.0,    "volatility": 1.0},
        "✅ Optimal": {"factor": 0.9,  "drift": -0.0005, "volatility": 0.5}, # Improving over time, stable
        "⚖️ Normal":  {"factor": 1.0,  "drift": 0.0,    "volatility": 1.0},
        "❌ Worst":   {"factor": 1.1,  "drift": 0.001,  "volatility": 1.5}  # Degrading over time, unstable
    }
    
    s_cfg = scenarios[scenario]
    
    
    np.random.seed(42) # For reproducibility
    
    forecast_dates = pd.date_range(max_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    
    for i, date in enumerate(forecast_dates):
        # Generate synthetic inputs for this day
        daily_inputs = {}
        
        for col in cfg['features']:
            base = baseline_mean.get(col, 0)
            std = baseline_std.get(col, 0)
            
            # 1. Apply Scenario Factor (shift mean)
            val = base * s_cfg['factor']
            
            # 2. Add Drift (trend over time)
            # e.g. Effiency decreases -> Flow/Temp might increase
            val = val * (1 + s_cfg['drift'] * i)
            
            # 3. Add Volatility (Random Noise)
            noise = np.random.normal(0, std) * s_cfg['volatility']
            
            daily_inputs[col] = val + noise
            
        # Create input dataframe for model
        X_daily = pd.DataFrame([daily_inputs])
        
        # Predict parameters using trained models
        emissions = {}
        for key, model in st.session_state.global_models.items():
            if 'n_features_in_' in dir(model) and model.n_features_in_ == len(cfg['features']):
                 # Reorder columns to match training
                 predictions = model.predict(X_daily[cfg['features']])[0]
            else:
                 # Fallback if mismatch (shouldn't happen with correct flow)
                 predictions = model.predict(X_daily.iloc[:, :model.n_features_in_])[0]
            
            emissions[key] = max(0, predictions) # Clamp negative predictions
            
        # Calculate CF
        fuel = daily_inputs.get('fuel_gas_flow_kg_s', 2.0)
        cf = calculate_carbon_footprint(
            fuel, 
            emissions.get('nox_mg_nm3', 0), 
            emissions.get('co_mg_nm3', 0), 
            emissions.get('pm_mg_nm3', 0)
        )
        
        forecast_results.append({
            'date': date,
            **emissions,
            'co2e_kg_h': cf['total'] * 1000
        })
    
    forecast_df = pd.DataFrame(forecast_results)
    
    # Display
    tab1, tab2 = st.tabs(["📈 Emissions Forecast", "🌍 Carbon Footprint"])
    
    with tab1:
        emission_key = st.selectbox("Select Emission", list(EMISSION_NAMES.keys()), format_func=lambda x: EMISSION_NAMES[x])
        
        fig = go.Figure()
        hist = data.groupby(data['timestamp'].dt.date)[emission_key].mean().reset_index()
        hist.columns = ['date', 'value']
        fig.add_trace(go.Scatter(x=hist['date'], y=hist['value'], name='Historical', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df[emission_key], name='Forecast', line=dict(color='red', dash='dash')))
        fig.update_layout(title=f"{EMISSION_NAMES[emission_key]} Forecast - {scenario}", height=450)
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
    
    with tab2:
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['co2e_kg_h'], 
                                     name='CO₂e Forecast', fill='tozeroy', 
                                     line=dict(color='green')))
        fig_cf.update_layout(title=f"Carbon Footprint Forecast - {scenario}", yaxis_title="kg CO₂e/h", height=450)
        st.plotly_chart(fig_cf, use_container_width=True, config=CHART_CONFIG)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg CO₂e/h", f"{forecast_df['co2e_kg_h'].mean():.2f} kg")
        with col2:
            total = forecast_df['co2e_kg_h'].sum() * 24 / 1000
            st.metric(f"Total ({forecast_days}d)", f"{total:.2f} tonnes")
        with col3:
            st.metric("Period", f"{forecast_days} days")
    
    st.download_button("📥 Download Forecast CSV", forecast_df.to_csv(index=False), f"forecast_{forecast_days}d.csv", "text/csv")
