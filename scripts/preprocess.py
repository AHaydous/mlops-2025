import argparse
import pandas as pd
import os
import sys

# Add src to path
sys.path.append('src')
from mlops_2025.preprocessing.preprocessor import TitanicPreprocessor

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Titanic Data Preprocessing")
    p.add_argument("--input", required=True, help="Input raw CSV file")
    p.add_argument("--output", required=True, help="Output processed CSV file")
    return p

def main():
    args = build_parser().parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Read raw data
    print(f"Reading data from {args.input}")
    df = pd.read_csv(args.input)
    print(f"Original shape: {df.shape}")
    
    # USE CLASS instead of direct logic
    preprocessor = TitanicPreprocessor()
    processed_df = preprocessor.process(df)
    
    print(f"Processed shape: {processed_df.shape}")
    print("Missing values after processing:")
    print(processed_df.isnull().sum())
    
    # Save processed data
    processed_df.to_csv(args.output, index=False)
    print(f"Processed data saved to {args.output}")

if __name__ == "__main__":
    main()