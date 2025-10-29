import argparse
import pandas as pd
import pickle
import json
import os
from sklearn.metrics import accuracy_score, classification_report

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Titanic Model")
    p.add_argument("--model_input", required=True, help="Input model file")
    p.add_argument("--test_data", required=True, help="Test data CSV file") 
    p.add_argument("--metrics_output", required=True, help="Output metrics JSON file")
    return p

def main():
    args = build_parser().parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_input}")
    with open(args.model_input, 'rb') as f:
        model = pickle.load(f)
    
    # Load test data - NOW USING THE PROPER TEST SET
    print(f"Loading test data from {args.test_data}")
    test_df = pd.read_csv(args.test_data)
    
    if 'Survived' not in test_df.columns:
        print("Error: 'Survived' column not found in test data")
        return
    
    # Prepare features and target
    X_test = test_df.drop('Survived', axis=1)
    if 'PassengerId' in X_test.columns:
        X_test = X_test.drop('PassengerId', axis=1)
    y_test = test_df['Survived']
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save metrics to JSON
    metrics = {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    with open(args.metrics_output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {args.metrics_output}")

if __name__ == "__main__":
    main()