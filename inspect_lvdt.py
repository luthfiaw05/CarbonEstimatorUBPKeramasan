
import pandas as pd

file_path = r"c:\Users\ASUS\Downloads\KP_Analisis Proses Pembakaran Turbin Gas dan Estimasi Carbon Footprint dalam Konsisi Operasi Base Load Unit 1 UPPLTGU Keramasan PT PLN Indonesia Power\raw_data\logsheet GT1.xlsx"

try:
    df = pd.read_excel(file_path, header=0, nrows=20, usecols=['LVDT 1 (%)', 'LVDT 2 (%)', 'LVDT 1 (%).1', 'LVDT 2 (%).1', 'GT LOAD (MW)'])
    print(df.head(20))
except Exception as e:
    print(e)
