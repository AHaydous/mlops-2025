import argparse
import pandas as pd
import pickle
import os

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Make predictions with Titanic Model")
    p.add_argument("--model_input", required=True, help="Input model file")
    p.add_argument("--input_data", required=True, help="Input data CSV file for prediction")
    p.add_argument("--output", required=True, help="Output predictions CSV file")
    return p

def main():
    args = build_parser().parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_input}")
    with open(args.model_input, 'rb') as f:
        model = pickle.load(f)
    
    # Load data for prediction (no label as required)
    print(f"Loading data from {args.input_data}")
    df = pd.read_csv(args.input_data)
    
    # Store PassengerId if present for output
    passenger_ids = None
    if 'PassengerId' in df.columns:
        passenger_ids = df['PassengerId']
        X = df.drop('PassengerId', axis=1)
    else:
        X = df
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X)
    
    # Create output DataFrame
    result_df = pd.DataFrame()
    if passenger_ids is not None:
        result_df['PassengerId'] = passenger_ids
    result_df['Survived'] = predictions
    
    # Save predictions to CSV
    result_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(f"Predicted {len(predictions)} samples")

if __name__ == "__main__":
    main()