# csv_to_parquet.py
import pandas as pd
import sys
import os

csv_path = sys.argv[1] if len(sys.argv) > 1 else "stel_results.csv"
parquet_path = os.path.splitext(csv_path)[0] + ".parquet"

print(f"Reading CSV from: {csv_path}")
df = pd.read_csv(csv_path)

# Save as compressed Parquet
print(f"Writing compressed Parquet to: {parquet_path}")
df.to_parquet(parquet_path, index=False, compression='zstd')  # or use 'snappy'

print("Done.")
