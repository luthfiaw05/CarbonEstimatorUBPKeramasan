
import pandas as pd
import os

data_dir = r"c:\Users\ASUS\Downloads\KP_Analisis Proses Pembakaran Turbin Gas dan Estimasi Carbon Footprint dalam Konsisi Operasi Base Load Unit 1 UPPLTGU Keramasan PT PLN Indonesia Power\data"
gt_path = os.path.join(data_dir, 'gt_operation.csv')
cems_path = os.path.join(data_dir, 'cems_data.csv')

try:
    if os.path.exists(gt_path):
        print(f"Reading {gt_path}...")
        df = pd.read_csv(gt_path, nrows=1)
        print("GT Columns:", list(df.columns))
    else:
        print("GT file not found")
        
    if os.path.exists(cems_path):
        print(f"Reading {cems_path}...")
        df = pd.read_csv(cems_path, nrows=1)
        print("CEMS Columns:", list(df.columns))
    else:
        print("CEMS file not found")
        
except Exception as e:
    print(e)
