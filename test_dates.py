import pandas as pd
import numpy as np

df = pd.read_excel('raw_data/GTLOGSHEET_1.xlsx')
date_col = 6.14

def parse_date(val):
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(val, errors='coerce')
    if hasattr(val, 'strftime'):
        return pd.to_datetime(val, errors='coerce')
    str_val = str(val).strip().rstrip(',')
    try:
        return pd.to_datetime(str_val, dayfirst=True, errors='coerce')
    except:
        return pd.NaT

dates = df[date_col].apply(parse_date)
valid = dates.dropna()
valid = valid[(valid > pd.Timestamp('2020-01-01')) & (valid < pd.Timestamp('2027-01-01'))]

print('Total rows:', len(df))
print('Valid dates:', len(valid))
print('Date range:', valid.min(), 'to', valid.max())
