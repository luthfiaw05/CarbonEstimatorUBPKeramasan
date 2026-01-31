
import pandas as pd
import os

file_path = r"c:\Users\ASUS\Downloads\KP_Analisis Proses Pembakaran Turbin Gas dan Estimasi Carbon Footprint dalam Konsisi Operasi Base Load Unit 1 UPPLTGU Keramasan PT PLN Indonesia Power\raw_data\logsheet GT1.xlsx"

try:
    print(f"Reading {file_path}...")
    df = pd.read_excel(file_path, nrows=5)
    cols = list(df.columns)
    print(f"Total columns: {len(cols)}")
    print("First 20 columns:", cols[:20])
    
    # Keyword search
    keywords = ['date', 'time', 'tanggal', 'jam', 'waktu', 'fuel', 'gas', 'flow', 'load', 'mw', 'power', 'temp', 'exhaust', 'igv', 'press']
    print("\nPotential matches:")
    for col in cols:
        col_lower = str(col).lower()
        if any(k in col_lower for k in keywords):
            print(f"  {col}")
            
except Exception as e:
    print(f"Error: {e}")
