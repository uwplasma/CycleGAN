# parquet_to_csv.py
import pandas as pd
import sys
import os

parquet_path = sys.argv[1] if len(sys.argv) > 1 else "stel_results.parquet"
csv_path = os.path.splitext(parquet_path)[0] + "_restored.csv"

print(f"Reading Parquet from: {parquet_path}")
df = pd.read_parquet(parquet_path)

print(f"Writing CSV to: {csv_path}")
df.to_csv(csv_path, index=False)

print("Done.")
