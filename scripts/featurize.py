import argparse
import pandas as pd
import os
import sys

# Add src to path
sys.path.append('src')
from mlops_2025.features.features_computer import TitanicFeaturesComputer

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Titanic Feature Engineering")
    p.add_argument("--input", required=True, help="Input processed CSV file")
    p.add_argument("--output", required=True, help="Output features CSV file")
    return p

def main():
    args = build_parser().parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Read processed data
    print(f"Reading processed data from {args.input}")
    df = pd.read_csv(args.input)
    print(f"Input shape: {df.shape}")
    
    # USE CLASS INSTEAD OF DIRECT LOGIC
    features_computer = TitanicFeaturesComputer()
    features_df = features_computer.compute(df)
    
    print(f"Improved features shape: {features_df.shape}")
    print("Improved features columns:", features_df.columns.tolist())
    
    # Save features
    features_df.to_csv(args.output, index=False)
    print(f"Improved features saved to {args.output}")

if __name__ == "__main__":
    main()