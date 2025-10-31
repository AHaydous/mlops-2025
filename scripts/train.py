import argparse
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append('src')
from mlops_2025.models.logistic_model import LogisticModel
from mlops_2025.models.random_forest_model import RandomForestModel
from mlops_2025.models.xgboost_model import XGBoostModel

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Titanic Survival Model")
    p.add_argument("--input", required=True, help="Input features CSV file")
    p.add_argument("--model_output", required=True, help="Output model file")
    p.add_argument("--model_type", type=str, default="logistic", 
                   choices=["logistic", "random_forest", "xgboost"],
                   help="Type of model to train: logistic, random_forest, or xgboost")
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
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Choose model based on argument
    if args.model_type == "logistic":
        model = LogisticModel()
        print("Training Logistic Regression model...")
    elif args.model_type == "random_forest":
        model = RandomForestModel()
        print("Training Random Forest model...")
    elif args.model_type == "xgboost":
        model = XGBoostModel()
        print("Training XGBoost model...")
    
    # Train the model
    model.train(X_train, y_train)
    
    # Calculate accuracy
    train_score = model.model.score(X_train, y_train) if hasattr(model, 'model') else "N/A"
    test_predictions = model.predict(X_test)
    test_score = (test_predictions == y_test).mean()
    
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