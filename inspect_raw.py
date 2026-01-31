
import pandas as pd

file_path = r"c:\Users\ASUS\Downloads\KP_Analisis Proses Pembakaran Turbin Gas dan Estimasi Carbon Footprint dalam Konsisi Operasi Base Load Unit 1 UPPLTGU Keramasan PT PLN Indonesia Power\raw_data\logsheet GT1.xlsx"

try:
    df = pd.read_excel(file_path, header=None, nrows=10)
    print("Top 10 rows:")
    print(df.head(10))
except Exception as e:
    print(e)
