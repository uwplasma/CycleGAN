import argparse
import pandas as pd

def main(args):
    
    data = pd.read_csv(args.input_data)
    good_data = pd.read_csv(args.good_data)

    
    merged = pd.merge(data, good_data, on=data.columns.tolist(), how='left', indicator=True)
    diff_df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    
    bad_stels = diff_df.sample(n=args.n, random_state=42)

    
    bad_stels.to_csv(args.output_path, index=False)
    print(f"Saved {args.n} sampled bad configurations to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract N bad configurations not in good_data.")
    parser.add_argument("--input_data", type=str, required=True, help="Path to full dataset CSV (e.g. dataset.csv)")
    parser.add_argument("--good_data", type=str, required=True, help="Path to known good data CSV (e.g. GStels.csv)")
    parser.add_argument("--n", type=int, required=True, help="Number of bad samples to extract")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save bad sample CSV (e.g. bad_data.csv)")
    
    args = parser.parse_args()
    main(args)
