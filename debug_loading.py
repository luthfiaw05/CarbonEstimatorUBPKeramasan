import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(BASE_DIR, 'raw_data', 'GTLOGSHEET_1.xlsx')

TARGET_COLS = {
    'generator_power_mw': 'GT LOAD (MW)',
    # Add other key columns if needed
}

print(f"Loading {excel_path}...")
try:
    raw_df = pd.read_excel(excel_path)
    print(f"Initial shape: {raw_df.shape}")
except Exception as e:
    print(f"Error loading: {e}")
    exit()

# 1. Date Parsing
print("\n--- Date Parsing ---")
date_col = None
time_col = None

for col in raw_df.columns:
    if isinstance(col, float) or (isinstance(col, str) and col.replace('.', '').isdigit()):
        date_col = col
    if col == 'Pukul':
        time_col = col

print(f"Date col: {date_col}, Time col: {time_col}")

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

raw_df['date_parsed'] = raw_df[date_col].apply(parse_date)
print(f"Parsed dates NaT count: {raw_df['date_parsed'].isna().sum()}")

# Combine
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

print(f"Timestamp NaT count: {raw_df['timestamp'].isna().sum()}")

# 2. Data Mapping & Numeric Conversion
print("\n--- Column Mapping & Conversion ---")
gt_data = pd.DataFrame()
gt_data['timestamp'] = raw_df['timestamp']

col_name = 'generator_power_mw'
excel_col = TARGET_COLS[col_name]

if excel_col in raw_df.columns:
    print(f"Column '{excel_col}' found.")
    # Show sample values before conversion
    print("Sample raw values (head):", raw_df[excel_col].head().tolist())
    print("Sample raw values (tail):", raw_df[excel_col].tail().tolist())
    
    gt_data[col_name] = pd.to_numeric(raw_df[excel_col], errors='coerce')
    print(f"Converted '{col_name}'. NaN count: {gt_data[col_name].isna().sum()}")
else:
    print(f"Column '{excel_col}' NOT found!")

# 3. Filtering Steps
print("\n--- Filtering Steps ---")
print(f"Start count: {len(gt_data)}")

# Drop invalid timestamps
gt_data = gt_data.dropna(subset=['timestamp'])
print(f"After dropping NaT timestamps: {len(gt_data)}")

# Ensure timestamps are timezone-naive
if len(gt_data) > 0:
    # Check if tz-aware
    sample_tz = gt_data['timestamp'].iloc[0].tzinfo if not gt_data.empty else None
    print(f"Sample timezone info: {sample_tz}")
    try:
        gt_data['timestamp'] = gt_data['timestamp'].dt.tz_localize(None)
        print("Converted to timezone-naive.")
    except Exception as e:
        print(f"Timezone conversion note: {e}")
        # Fallback for mixed types if necessary
        gt_data['timestamp'] = pd.to_datetime(gt_data['timestamp'], utc=True).dt.tz_localize(None)


# Date range filter
gt_data = gt_data[gt_data['timestamp'] < pd.Timestamp('2027-01-01')]
gt_data = gt_data[gt_data['timestamp'] > pd.Timestamp('2020-01-01')]
print(f"After date range filter (2020-2027): {len(gt_data)}")

# Overhaul filter (notna)
if col_name in gt_data.columns:
    before = len(gt_data)
    gt_data = gt_data[gt_data[col_name].notna()]
    print(f"After dropping NaNs in {col_name}: {len(gt_data)} (Dropped {before - len(gt_data)})")
    
    # Overhaul filter (>0)
    before = len(gt_data)
    gt_data = gt_data[gt_data[col_name] > 0]
    print(f"After filtering {col_name} > 0: {len(gt_data)} (Dropped {before - len(gt_data)})")

print("\nFinal count:", len(gt_data))
if len(gt_data) < 50:
    print("\nData remaining:")
    print(gt_data.head())
