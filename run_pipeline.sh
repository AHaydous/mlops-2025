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

# Step 3: Training
echo "=== Step 3: Training model ==="
uv run python scripts/train.py \
    --input data/features/train_features.csv \
    --model_output models/titanic_model.pkl

# Step 4: Evaluation
echo "=== Step 4: Evaluating model ==="
uv run python scripts/evaluate.py \
    --model_input models/titanic_model.pkl \
    --test_data data/features/train_features.csv \
    --metrics_output metrics/metrics.json

echo "=== Pipeline completed! ==="