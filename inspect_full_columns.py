
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

file_path = r"c:\Users\ASUS\Downloads\KP_Analisis Proses Pembakaran Turbin Gas dan Estimasi Carbon Footprint dalam Konsisi Operasi Base Load Unit 1 UPPLTGU Keramasan PT PLN Indonesia Power\raw_data\logsheet GT1.xlsx"

try:
    df = pd.read_excel(file_path, nrows=2) # Read header
    with open('columns.txt', 'w') as f:
        for i, col in enumerate(df.columns):
            f.write(f"{i}: {col}\n")
    print("Columns written to columns.txt")
    print("-------------------")
    print("First row values:")
    print(df.iloc[0].to_dict())
except Exception as e:
    print(e)
