import argparse
import pandas as pd
import os

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Titanic Feature Engineering - IMPROVED FEATURES")
    p.add_argument("--input", required=True, help="Input processed CSV file")
    p.add_argument("--output", required=True, help="Output features CSV file")
    return p

def family_size_improved(number):
    """CHANGED FEATURE: Different binning strategy"""
    if number == 1:
        return "Alone"
    elif number == 2:
        return "Couple"
    elif 3 <= number <= 4:
        return "Small_Family"
    else:
        return "Large_Family"

def main():
    args = build_parser().parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Read processed data
    print(f"Reading processed data from {args.input}")
    df = pd.read_csv(args.input)
    print(f"Input shape: {df.shape}")
    
    # KEEP the Title feature from Step 4.2
    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    df['Title'] = df['Title'].replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # CHANGED FEATURE (Step 4.5): Different Family_size binning
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    df['Family_size'] = df['Family_size'].apply(family_size_improved)
    
    # Drop unnecessary columns
    columns_to_drop = ['Name', 'SibSp', 'Parch', 'Ticket']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    
    print(f"Improved features shape: {df.shape}")
    print("Improved features columns:", df.columns.tolist())
    
    # Save features
    df.to_csv(args.output, index=False)
    print(f"Improved features saved to {args.output}")

if __name__ == "__main__":
    main()