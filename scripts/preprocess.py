import argparse
import pandas as pd
import os

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Titanic Data Preprocessing")
    p.add_argument("--input", required=True, help="Input raw CSV file")
    p.add_argument("--output", required=True, help="Output processed CSV file")
    return p

def main():
    args = build_parser().parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Read raw data
    print(f"Reading data from {args.input}")
    df = pd.read_csv(args.input)
    print(f"Original shape: {df.shape}")
    
    # Basic preprocessing from notebook
    # Drop Cabin column (too many missing values)
    if 'Cabin' in df.columns:
        df = df.drop(columns=['Cabin'])
    
    # Handle missing values as in notebook (without inplace)
    if 'Embarked' in df.columns:
        df = df.assign(Embarked=df['Embarked'].fillna('S'))
    
    if 'Fare' in df.columns:
        df = df.assign(Fare=df['Fare'].fillna(df['Fare'].mean()))
    
    # Handle Age missing values using median by Sex and Pclass (from notebook)
    if 'Age' in df.columns:
        age_median_by_group = df.groupby(['Sex', 'Pclass'])['Age'].median()
        
        def fill_age(row):
            if pd.isna(row['Age']):
                return age_median_by_group[row['Sex'], row['Pclass']]
            return row['Age']
        
        df = df.assign(Age=df.apply(fill_age, axis=1))
        df = df.assign(Age=df['Age'].astype('int64'))
    
    print(f"Processed shape: {df.shape}")
    print("Missing values after processing:")
    print(df.isnull().sum())
    
    # Save processed data
    df.to_csv(args.output, index=False)
    print(f"Processed data saved to {args.output}")

if __name__ == "__main__":
    main()