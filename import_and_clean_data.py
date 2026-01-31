"""
Data Import and Cleaning Pipeline for GT Operation and CEMS Data

This script:
1. Loads raw data files from raw_data/ folder
2. Maps columns using column_mapping.json
3. Parses timestamps and resamples to hourly
4. Handles missing values and outliers
5. Aligns GT and CEMS datasets
6. Saves cleaned data to data/ folder

Usage:
    1. Place your raw CSV/Excel files in raw_data/ folder
    2. Edit column_mapping.json to match your column names
    3. Run: python import_and_clean_data.py
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(__file__)
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
CONFIG_FILE = os.path.join(BASE_DIR, 'column_mapping.json')


def load_config():
    """Load column mapping configuration."""
    print("Loading configuration...")
    
    if not os.path.exists(CONFIG_FILE):
        print(f"  ERROR: {CONFIG_FILE} not found!")
        print("  Please create the configuration file first.")
        return None
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("  Configuration loaded successfully!")
    return config


def find_data_files(pattern, data_type):
    """Find data files matching pattern in raw_data folder."""
    search_path = os.path.join(RAW_DATA_DIR, pattern)
    files = glob.glob(search_path)
    
    # Also check for Excel files
    excel_path = os.path.join(RAW_DATA_DIR, pattern.replace('.csv', '.xlsx'))
    excel_files = glob.glob(excel_path)
    files.extend(excel_files)
    
    # Remove duplicates and filter temp files
    files = sorted(list(set(files)))
    files = [f for f in files if not os.path.basename(f).startswith('~$')]
    
    if not files:
        print(f"  WARNING: No {data_type} files found matching '{pattern}'")
        print(f"           Please place your files in: {RAW_DATA_DIR}")
        return []
    
    print(f"  Found {len(files)} {data_type} file(s)")
    for f in files:
        print(f"    - {os.path.basename(f)}")
    
    return files


def load_raw_file(filepath):
    """Load a single raw data file (CSV or Excel)."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.csv':
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not read CSV with common encodings: {filepath}")
    
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
        # Ensure column names are strings (handle numeric headers like 6.14)
        df.columns = df.columns.astype(str)
        return df
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def map_columns(df, column_mapping, data_type, keep_extra=None):
    """Rename columns based on mapping configuration."""
    print(f"\n  Mapping columns for {data_type}...")
    
    # Create reverse mapping (raw -> standard)
    rename_map = {}
    missing_cols = []
    
    for standard_name, raw_name in column_mapping.items():
        if raw_name in df.columns:
            rename_map[raw_name] = standard_name
            print(f"    ✓ {raw_name} → {standard_name}")
        else:
            missing_cols.append((standard_name, raw_name))
    
    if missing_cols:
        print(f"\n  WARNING: Missing columns:")
        for std, raw in missing_cols:
            print(f"    ✗ {raw} (expected for {std})")
        print(f"\n  Available columns in file: {list(df.columns)}")
    
    # Rename found columns
    df = df.rename(columns=rename_map)
    
    # Keep only mapped columns + extra
    keep_cols = [col for col in column_mapping.keys() if col in df.columns]
    
    if keep_extra:
        for col in keep_extra:
            if col and col in df.columns and col not in keep_cols:
                keep_cols.append(col)
                print(f"    + Keeping extra column: {col}")
    
    df = df[keep_cols]
    
    return df


def parse_timestamps(df, timestamp_col='timestamp', config={}, fmt='auto'):
    """Parse timestamp column with various format detection."""
    print(f"\n  Parsing timestamps...")
    
    settings = config.get('settings', {})
    date_col = settings.get('date_column')
    time_col = settings.get('time_column')
    
    # Handle separate date/time columns
    if date_col and time_col and date_col in df.columns and time_col in df.columns:
        print(f"    Merging Date ('{date_col}') and Time ('{time_col}')...")
        try:
            # Convert separately first
            temp_date = pd.to_datetime(df[date_col], errors='coerce')
            temp_time = pd.to_datetime(df[time_col], errors='coerce').dt.time
            
            # Create merged timestamp series safely
            # Convert date/time to string, handling NaT/None
            date_str = temp_date.dt.date.astype(str)
            time_str = temp_time.astype(str)
            
            # Combine only where both are valid (not 'NaT' or 'nan')
            valid_mask = (date_str != 'NaT') & (date_str != 'nan') & (time_str != 'NaT') & (time_str != 'nan')
            
            merged_ts = pd.Series(pd.NaT, index=df.index)
            merged_ts[valid_mask] = pd.to_datetime(date_str[valid_mask] + ' ' + time_str[valid_mask], errors='coerce')
            
            # If timestamp column exists, fill missing values
            if timestamp_col in df.columns:
                original_count = df[timestamp_col].notna().sum()
                # Ensure existing is datetime
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                # Fill na
                df[timestamp_col] = df[timestamp_col].fillna(merged_ts)
                new_count = df[timestamp_col].notna().sum()
                print(f"    ✓ Filled {new_count - original_count} missing timestamps from merge")
            else:
                df[timestamp_col] = merged_ts
                print(f"    ✓ Created {merged_ts.notna().sum()} timestamps from merge")
            
            return df
        except Exception as e:
            print(f"    ERROR merging date/time: {e}")
            # print details to help debug
            import traceback
            traceback.print_exc()
            return df
    
    if timestamp_col not in df.columns:
        print(f"    ERROR: Timestamp column '{timestamp_col}' not found!")
        return df
    
    # Try automatic parsing first
    if fmt == 'auto':
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], infer_datetime_format=True)
            print(f"    ✓ Auto-parsed {len(df)} timestamps")
            return df
        except Exception:
            pass
    
    # Try common formats
    common_formats = [
        '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S',
        '%m/%d/%Y %H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
        '%d-%m-%Y %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%d/%m/%Y %H:%M',
        '%Y%m%d%H%M%S',
    ]
    
    for date_fmt in common_formats:
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=date_fmt)
            print(f"    ✓ Parsed with format: {date_fmt}")
            return df
        except Exception:
            continue
    
    print(f"    ERROR: Could not parse timestamps!")
    try:
        print(f"    Sample values: {df[timestamp_col].head(3).tolist()}")
    except:
        pass
    return df


def apply_conversions(df, conversions):
    """Apply unit conversions."""
    if not conversions:
        return df
        
    print(f"\n  Applying unit conversions...")
    for col, factor in conversions.items():
        if col in df.columns:
            try:
                # Ensure numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Handle division (start with /)
                if isinstance(factor, str) and factor.startswith('/'):
                    divisor = float(factor[1:])
                    df[col] = df[col] / divisor
                    print(f"    ✓ {col}: divided by {divisor}")
                # Handle multiplication (default)
                else:
                    mult = float(factor)
                    df[col] = df[col] * mult
                    print(f"    ✓ {col}: multiplied by {mult}")
            except Exception as e:
                print(f"    ERROR converting {col}: {e}")
    return df


def resample_to_hourly(df, freq='1h'):
    """Resample data to hourly frequency."""
    print(f"\n  Resampling to {freq} frequency...")
    
    if 'timestamp' not in df.columns:
        print("    ERROR: No timestamp column for resampling!")
        return df
    
    # Set timestamp as index
    df = df.set_index('timestamp')
    
    # Resample (mean for numeric columns)
    # Select only numeric columns to avoid TypeError
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("    WARNING: No numeric data to resample!")
        return df.iloc[0:0] # Return empty
        
    df_resampled = numeric_df.resample(freq).mean()
    
    # Reset index
    df_resampled = df_resampled.reset_index()
    
    print(f"    Original: {len(df)} rows → Resampled: {len(df_resampled)} rows")
    
    return df_resampled


def coerce_to_numeric(df, exclude_cols=['timestamp']):
    """Coerce all non-excluded columns to numeric."""
    print(f"\n  Coercing columns to numeric...")
    for col in df.columns:
        if col not in exclude_cols:
            try:
                # Check if column is already numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    continue
                    
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"    ✓ {col}: converted to numeric")
            except Exception as e:
                print(f"    ERROR converting {col} to numeric: {e}")
    return df


def handle_outliers(df, threshold=4, method='clip'):
    """Handle outliers using z-score method."""
    print(f"\n  Handling outliers (method={method}, threshold={threshold}σ)...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_found = 0
    
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        
        if std == 0:
            continue
        
        z_scores = np.abs((df[col] - mean) / std)
        outlier_mask = z_scores > threshold
        n_outliers = outlier_mask.sum()
        
        if n_outliers > 0:
            outliers_found += n_outliers
            
            if method == 'clip':
                lower = mean - threshold * std
                upper = mean + threshold * std
                df[col] = df[col].clip(lower, upper)
            elif method == 'nan':
                df.loc[outlier_mask, col] = np.nan
    
    if outliers_found > 0:
        print(f"    Handled {outliers_found} outliers")
    else:
        print(f"    No outliers detected")
    
    return df


def handle_missing_values(df):
    """Handle missing values with forward/backward fill."""
    print(f"\n  Handling missing values...")
    
    missing_before = df.isnull().sum().sum()
    
    if missing_before > 0:
        # Forward fill then backward fill
        df = df.ffill().bfill()
        
        missing_after = df.isnull().sum().sum()
        print(f"    Filled {missing_before - missing_after} missing values")
        
        if missing_after > 0:
            print(f"    WARNING: {missing_after} values still missing!")
    else:
        print(f"    No missing values found")
    
    return df


def align_datasets(gt_df, cems_df):
    """Align GT and CEMS datasets on timestamp."""
    print("\n" + "=" * 50)
    print("ALIGNING DATASETS")
    print("=" * 50)
    
    if 'timestamp' not in gt_df.columns or 'timestamp' not in cems_df.columns:
        print("  ERROR: Both datasets must have 'timestamp' column!")
        return gt_df, cems_df
    
    # Find common time range (For Report Only)
    gt_start = gt_df['timestamp'].min()
    gt_end = gt_df['timestamp'].max()
    cems_start = cems_df['timestamp'].min()
    cems_end = cems_df['timestamp'].max()
    
    print(f"\n  GT Range:   {gt_start} to {gt_end}")
    print(f"  CEMS Range: {cems_start} to {cems_end}")
    
    # Do NOT filter strictly. Keep all data.
    # The user might want to analyze GT data even if CEMS is missing.
    gt_aligned = gt_df
    cems_aligned = cems_df
    
    # But if one is completely empty, we might want to warn
    if len(gt_aligned) == 0:
        print("  WARNING: GT Data is empty!")
    if len(cems_aligned) == 0:
        print("  WARNING: CEMS Data is empty!")
    
    return gt_aligned, cems_aligned


def generate_data_report(gt_df, cems_df):
    """Generate a summary report of the cleaned data."""
    print("\n" + "=" * 50)
    print("DATA SUMMARY REPORT")
    print("=" * 50)
    
    print("\n  GT Operation Data:")
    print(f"    Samples: {len(gt_df)}")
    print(f"    Time Range: {gt_df['timestamp'].min()} to {gt_df['timestamp'].max()}")
    print(f"    Columns: {list(gt_df.columns)}")
    
    print("\n  CEMS Data:")
    print(f"    Samples: {len(cems_df)}")
    print(f"    Time Range: {cems_df['timestamp'].min()} to {cems_df['timestamp'].max()}")
    print(f"    Columns: {list(cems_df.columns)}")
    
    # Check for training/testing requirements
    total_hours = len(gt_df)
    training_hours = 24 * 30  # 720
    testing_hours = 24 * 7    # 168
    required_hours = training_hours + testing_hours
    
    print(f"\n  Model Requirements:")
    print(f"    Training period: {training_hours} hours (30 days)")
    print(f"    Testing period:  {testing_hours} hours (7 days)")
    print(f"    Total required:  {required_hours} hours")
    print(f"    Available:       {total_hours} hours")
    
    if total_hours >= required_hours:
        print(f"    ✓ Sufficient data available!")
    else:
        print(f"    ✗ Need {required_hours - total_hours} more hours of data!")


def process_gt_data(config):
    """Process GT operation data files."""
    print("\n" + "=" * 50)
    print("PROCESSING GT OPERATION DATA")
    print("=" * 50)
    
    settings = config.get('settings', {})
    gt_mapping = config.get('gt_operation', {})
    
    # Find GT files
    pattern = settings.get('gt_file_pattern', '*.csv')
    files = find_data_files(pattern, 'GT operation')
    
    if not files:
        return None
    
    # Load and concatenate all files
    all_dfs = []
    for filepath in files:
        print(f"\n  Loading: {os.path.basename(filepath)}")
        df = load_raw_file(filepath)
        print(f"    Rows: {len(df)}, Columns: {len(df.columns)}")
        all_dfs.append(df)
    
    # Combine if multiple files
    if len(all_dfs) > 1:
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n  Combined: {len(df)} total rows")
    else:
        df = all_dfs[0]
    
    # Map columns, keeping date/time for parsing
    extra_cols = [settings.get('date_column'), settings.get('time_column')]
    df = map_columns(df, gt_mapping, 'GT Operation', keep_extra=extra_cols)
    
    # Parse timestamps
    df = parse_timestamps(df, config=config, fmt=settings.get('timestamp_format', 'auto'))
    
    if 'timestamp' not in df.columns:
        print("    WARNING: 'timestamp' column not found after parsing. Assuming no valid CEMS data.")
        # Return empty df with expected columns
        expected_cols = ['timestamp'] + list(cems_mapping.keys())
        return pd.DataFrame(columns=expected_cols)
    
    # Apply conversions
    df = apply_conversions(df, config.get('unit_conversions', {}))
    
    # Coerce all columns to numeric (crucial for sensor data)
    df = coerce_to_numeric(df, exclude_cols=['timestamp'])
    
    # Resample
    df = resample_to_hourly(df, freq=settings.get('resample_freq', '1h'))
    
    # Handle outliers
    df = handle_outliers(df, 
                        threshold=settings.get('outlier_std_threshold', 4),
                        method=settings.get('outlier_handling', 'clip'))
    
    # Handle missing values
    df = handle_missing_values(df)
    
    return df


def process_cems_data(config):
    """Process CEMS data files."""
    print("\n" + "=" * 50)
    print("PROCESSING CEMS DATA")
    print("=" * 50)
    
    settings = config.get('settings', {})
    cems_mapping = config.get('cems', {})
    
    # Find CEMS files
    pattern = settings.get('cems_file_pattern', '*.csv')
    files = find_data_files(pattern, 'CEMS')
    
    if not files:
        return None
    
    # Load and concatenate all files
    all_dfs = []
    for filepath in files:
        print(f"\n  Loading: {os.path.basename(filepath)}")
        df = load_raw_file(filepath)
        print(f"    Rows: {len(df)}, Columns: {len(df.columns)}")
        all_dfs.append(df)
    
    # Combine if multiple files
    if len(all_dfs) > 1:
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n  Combined: {len(df)} total rows")
    else:
        df = all_dfs[0]
    
    # Map columns
    extra_cols = [settings.get('date_column'), settings.get('time_column')]
    df = map_columns(df, cems_mapping, 'CEMS', keep_extra=extra_cols)
    
    # Parse timestamps
    df = parse_timestamps(df, config=config, fmt=settings.get('timestamp_format', 'auto'))
    
    # Resample
    df = resample_to_hourly(df, freq=settings.get('resample_freq', '1h'))
    
    # Handle outliers
    df = handle_outliers(df, 
                        threshold=settings.get('outlier_std_threshold', 4),
                        method=settings.get('outlier_handling', 'clip'))
    
    # Handle missing values
    df = handle_missing_values(df)
    
    return df


def save_cleaned_data(gt_df, cems_df):
    """Save cleaned data to output directory."""
    print("\n" + "=" * 50)
    print("SAVING CLEANED DATA")
    print("=" * 50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    gt_path = os.path.join(OUTPUT_DIR, 'gt_operation.csv')
    cems_path = os.path.join(OUTPUT_DIR, 'cems_data.csv')
    
    gt_df.to_csv(gt_path, index=False)
    cems_df.to_csv(cems_path, index=False)
    
    print(f"\n  ✓ Saved: {gt_path}")
    print(f"  ✓ Saved: {cems_path}")


def main():
    print("=" * 60)
    print("DATA IMPORT AND CLEANING PIPELINE")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories if needed
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load configuration
    config = load_config()
    if config is None:
        return
    
    # Check for raw data files
    print("\n" + "=" * 50)
    print("CHECKING RAW DATA FILES")
    print("=" * 50)
    
    raw_files = os.listdir(RAW_DATA_DIR)
    data_files = [f for f in raw_files if f.endswith(('.csv', '.xlsx', '.xls'))]
    
    if not data_files:
        print(f"\n  No data files found in: {RAW_DATA_DIR}")
        print("\n  INSTRUCTIONS:")
        print("  1. Place your GT operation CSV/Excel files in raw_data/")
        print("  2. Place your CEMS data CSV/Excel files in raw_data/")
        print("  3. Edit column_mapping.json to match your column names")
        print("  4. Run this script again")
        print("\n  TIP: Use different file naming patterns to distinguish GT and CEMS data")
        print("       e.g., 'gt_*.csv' for GT files and 'cems_*.csv' for CEMS files")
        print("       Then update 'gt_file_pattern' and 'cems_file_pattern' in config")
        return
    
    print(f"\n  Found {len(data_files)} data file(s):")
    for f in data_files:
        print(f"    - {f}")
    
    # Process GT data
    gt_df = process_gt_data(config)
    
    # Process CEMS data
    cems_df = process_cems_data(config)
    
    # Handle missing CEMS data gracefully
    if cems_df is None:
        print("\n  WARNING: No CEMS data processed. Creating empty placeholder.")
        cems_mapping = config.get('cems', {})
        expected_cols = ['timestamp'] + list(cems_mapping.keys())
        cems_df = pd.DataFrame(columns=expected_cols)
        
    # Check if GT data is available
    if gt_df is None:
        print("\n  ERROR: Could not process GT data!")
        print("  Please check your files and column_mapping.json")
        return
    
    # Align datasets
    gt_aligned, cems_aligned = align_datasets(gt_df, cems_df)
    
    # Generate report
    generate_data_report(gt_aligned, cems_aligned)
    
    # Save cleaned data
    save_cleaned_data(gt_aligned, cems_aligned)
    
    print("\n" + "=" * 60)
    print("DATA IMPORT COMPLETE!")
    print("=" * 60)
    print("\n  Next step: Run 'python train_emissions_model.py' to train the model")


if __name__ == "__main__":
    main()
