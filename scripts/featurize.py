import argparse
import pandas as pd
import os

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Titanic Feature Engineering")
    p.add_argument("--input", required=True, help="Input processed CSV file")
    p.add_argument("--output", required=True, help="Output features CSV file")
    return p

def family_size(number):
    if number == 1:
        return "Alone"
    elif number > 1 and number < 5:
        return "Small"
    else:
        return "Large"

def main():
    args = build_parser().parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Read processed data
    print(f"Reading processed data from {args.input}")
    df = pd.read_csv(args.input)
    print(f"Input shape: {df.shape}")
    
    # Feature engineering from notebook
    
    # 1. Extract title from name (from notebook)
    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    
    # Simplify titles (from notebook)
    df['Title'] = df['Title'].replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # 2. Create family size feature (from notebook)
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    df['Family_size'] = df['Family_size'].apply(family_size)
    
    # 3. Drop unnecessary columns as in notebook
    columns_to_drop = ['Name', 'SibSp', 'Parch', 'Ticket']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    
    print(f"Features shape: {df.shape}")
    print("Final columns:", df.columns.tolist())
    
    # Save features (X and y together as required)
    df.to_csv(args.output, index=False)
    print(f"Features saved to {args.output}")

if __name__ == "__main__":
    main()