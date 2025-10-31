#!/bin/bash

# Titanic ML Pipeline Runner
echo "Starting Titanic ML Pipeline..."

# Create directories
mkdir -p data/processed
mkdir -p data/features
mkdir -p models
mkdir -p metrics

# Step 1: Preprocessing
echo "=== Step 1: Preprocessing data ==="
uv run python scripts/preprocess.py \
    --input data/raw/train.csv \
    --output data/processed/train_processed.csv

# Check if preprocessing was successful
if [ ! -f "data/processed/train_processed.csv" ]; then
    echo "❌ Preprocessing failed!"
    exit 1
fi

# Step 2: Feature Engineering
echo "=== Step 2: Feature engineering ==="
uv run python scripts/featurize.py \
    --input data/processed/train_processed.csv \
    --output data/features/train_features.csv

# Step 3: Training all three models
echo "=== Step 3: Training models ==="
echo "Training Logistic Regression..."
uv run python scripts/train.py \
    --input data/features/train_features.csv \
    --model_output models/logistic_model.pkl \
    --model_type logistic

echo "Training Random Forest..."
uv run python scripts/train.py \
    --input data/features/train_features.csv \
    --model_output models/random_forest_model.pkl \
    --model_type random_forest

echo "Training XGBoost..."
uv run python scripts/train.py \
    --input data/features/train_features.csv \
    --model_output models/xgboost_model.pkl \
    --model_type xgboost

# Step 4: Evaluate all models
echo "=== Step 4: Evaluating models ==="
uv run python scripts/evaluate.py \
    --model_input models/logistic_model.pkl \
    --test_data models/logistic_model_test_set.csv \
    --metrics_output metrics/logistic_metrics.json

uv run python scripts/evaluate.py \
    --model_input models/random_forest_model.pkl \
    --test_data models/random_forest_model_test_set.csv \
    --metrics_output metrics/random_forest_metrics.json

uv run python scripts/evaluate.py \
    --model_input models/xgboost_model.pkl \
    --test_data models/xgboost_model_test_set.csv \
    --metrics_output metrics/xgboost_metrics.json

# Step 5: Compare models
echo "=== Step 5: Comparing models ==="
uv run python scripts/compare_models.py

echo "=== Pipeline completed! ==="
read -p "Pipeline finished — press Enter to close..."