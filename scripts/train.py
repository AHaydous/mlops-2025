import argparse
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

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
    
    # Drop PassengerId for training
    if 'PassengerId' in X.columns:
        X = X.drop('PassengerId', axis=1)
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Create model pipeline with LogisticRegression (simple baseline as required)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # Train model
    print("Training model...")
    model.fit(X, y)
    
    # Save model with pickle as required
    with open(args.model_output, 'wb') as f:
        pickle.dump(model, f)
    
    # Training metrics
    train_score = model.score(X, y)
    print(f"Model saved to {args.model_output}")
    print(f"Training accuracy: {train_score:.4f}")

if __name__ == "__main__":
    main()