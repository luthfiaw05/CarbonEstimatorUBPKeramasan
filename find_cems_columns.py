
import pandas as pd
import os
import glob

RAW_DIR = 'raw_data'
KEYWORDS = ['NOX', 'SOX', 'SO2', 'CO', 'EMISI', 'CEMS', 'PARTICULATE', 'PM', 'MG/NM3']

print(f"Scanning files in {RAW_DIR} for keywords: {KEYWORDS}...\n")

files = glob.glob(os.path.join(RAW_DIR, '*.xlsx')) + glob.glob(os.path.join(RAW_DIR, '*.xls'))

found_any = False

for f in files:
    try:
        print(f"Checking {os.path.basename(f)}...")
        # Read only header
        df = pd.read_excel(f, nrows=0)
        cols = [str(c).upper() for c in df.columns]
        
        matches = []
        for col in cols:
            if any(k in col for k in KEYWORDS):
                matches.append(col)
        
        if matches:
            print(f"  ✅ FOUND potential CEMS columns:")
            for m in matches:
                print(f"    - {m}")
            found_any = True
        else:
            print("  ❌ No obvious emission columns found.")
            
        print("-" * 30)
            
    except Exception as e:
        print(f"  ⚠️ Error reading {f}: {e}")

if not found_any:
    print("\nWARNING: No CEMS columns found in any file!")
