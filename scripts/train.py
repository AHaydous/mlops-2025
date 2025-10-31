import argparse
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append('src')
from mlops_2025.models.logistic_model import LogisticModel

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Titanic Survival Model")
    p.add_argument("--input", required=True, help="Input features CSV file")
    p.add_argument("--model_output", required=True, help="Output model file")
    return p

def main():
    args = build_parser().parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    
    # Read features
    print(f"Reading features from {args.input}")
    df = pd.read_csv(args.input)
    print(f"Data shape: {df.shape}")
    
    if 'Survived' not in df.columns:
        print("Error: 'Survived' column not found in input data")
        return
    
    # Separate features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Train/test split (YOUR EXACT LOGIC)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # USE CLASS INSTEAD OF DIRECT LOGIC
    model = LogisticModel()
    model.train(X_train, y_train)
    
    # Calculate accuracy
    train_score = model.model.score(X_train, y_train)
    test_score = model.model.score(X_test, y_test)
    
    # Save model
    model.save(args.model_output)
    
    print(f"Model saved to {args.model_output}")
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Save test set for evaluation
    test_df = X_test.copy()
    test_df['Survived'] = y_test
    test_output_path = args.model_output.replace('.pkl', '_test_set.csv')
    test_df.to_csv(test_output_path, index=False)
    print(f"Test set saved to {test_output_path}")

if __name__ == "__main__":
    main()