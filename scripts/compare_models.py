import json
import pandas as pd

def compare_models():
    """Compare all three models and display results."""
    models = {
        'logistic': 'Logistic Regression',
        'random_forest': 'Random Forest', 
        'xgboost': 'XGBoost'
    }
    
    print("=== Model Comparison ===")
    results = []
    
    for model_key, model_name in models.items():
        try:
            # Try metrics directory first (for production), then test_pipeline (for testing)
            metrics_paths = [
                f'metrics/{model_key}_metrics.json',
                f'test_pipeline/{model_key}_metrics.json'
            ]
            
            metrics = None
            for path in metrics_paths:
                try:
                    with open(path, 'r') as f:
                        metrics = json.load(f)
                    break
                except FileNotFoundError:
                    continue
            
            if metrics is None:
                print(f"\n{model_name}: Metrics file not found")
                continue
                
            accuracy = metrics['accuracy']
            precision_0 = metrics['classification_report']['0']['precision']
            recall_0 = metrics['classification_report']['0']['recall']
            precision_1 = metrics['classification_report']['1']['precision']
            recall_1 = metrics['classification_report']['1']['recall']
            
            results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision (0)': precision_0,
                'Recall (0)': recall_0,
                'Precision (1)': precision_1, 
                'Recall (1)': recall_1
            })
            
            print(f"\n{model_name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision (Class 0): {precision_0:.4f}")
            print(f"  Recall (Class 0): {recall_0:.4f}")
            print(f"  Precision (Class 1): {precision_1:.4f}")
            print(f"  Recall (Class 1): {recall_1:.4f}")
            
        except Exception as e:
            print(f"\n{model_name}: Error - {e}")
    
    # Find best model
    if results:
        best_model = max(results, key=lambda x: x['Accuracy'])
        print(f"\n🎯 Best Model: {best_model['Model']} with accuracy {best_model['Accuracy']:.4f}")
        
        # Save comparison table
        df = pd.DataFrame(results)
        df.to_csv('metrics/model_comparison.csv', index=False)
        print("Comparison saved to metrics/model_comparison.csv")

if __name__ == "__main__":
    compare_models()