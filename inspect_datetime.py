
import pandas as pd

file_path = r"c:\Users\ASUS\Downloads\KP_Analisis Proses Pembakaran Turbin Gas dan Estimasi Carbon Footprint dalam Konsisi Operasi Base Load Unit 1 UPPLTGU Keramasan PT PLN Indonesia Power\raw_data\logsheet GT1.xlsx"

try:
    df = pd.read_excel(file_path, header=0, nrows=10)
    print("Col 0 (Date) Name:", df.columns[0])
    print("Col 1 (Time) Name:", df.columns[1])
    
    print("\nValues:")
    print(df.iloc[:, [0, 1]].head())
except Exception as e:
    print(e)
