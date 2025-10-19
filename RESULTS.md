## Branches Created:
- feature/preprocess-cli
- feature/featurize-cli 
- feature/train-cli
- feature/eval-cli
- feature/predict-cli
- feature/pipeline-runner

## Final Metrics:
- **Accuracy:** 83.28%
- **Model:** Saved to `models/titanic_model.pkl`
- **Detailed Metrics:** metrics/metrics.json

## Results From Git Bash:

User@DESKTOP-V121TQ0 MINGW64 ~/Desktop/mlops-2025 (feature/pipeline-runner)
$ ./run_pipeline.sh
Starting Titanic ML Pipeline...
=== Step 1: Preprocessing data ===
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0⠙ Preparing packages... (0/0)                                                ⠙ Preparing packages... (0/1)                                                   Building mlops-2025 @ file:///C:/Users/User/Desktop/mlops-2025
      Built mlops-2025 @ file:///C:/Users/User/Desktop/mlops-2025
⠙ Preparing packages... (0/1)                                                ⠙  (1/1)                                                                     Uninstalled 1 package in 21ms
░░░░░░░░░░░░░░░░░░░░ [0/0] Installing wheels...                              ░░░░░░░░░░░░░░░░░░░░ [0/1] Installing wheels...                              ░░░░░░░░░░░░░░░░░░░░ [0/1] mlops-2025==0.1.0 (from file:///C:/Users/User/Desk████████████████████ [1/1] mlops-2025==0.1.0 (from file:///C:/Users/User/DeskInstalled 1 package in 52ms
Reading data from data/raw/train.csv
Original shape: (891, 12)
Processed shape: (891, 11)
Missing values after processing:
PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age            0
SibSp          0
Parch          0
Ticket         0
Fare           0
Embarked       0
dtype: int64
Processed data saved to data/processed/train_processed.csv
=== Step 2: Feature engineering ===
Reading processed data from data/processed/train_processed.csv
Input shape: (891, 11)
Features shape: (891, 9)
Final columns: ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'Family_size']
Features saved to data/features/train_features.csv
=== Step 3: Training model ===
Reading features from data/features/train_features.csv
Data shape: (891, 9)
Categorical columns: ['Sex', 'Embarked', 'Title', 'Family_size']
Numerical columns: ['Pclass', 'Age', 'Fare']
Training model...
Model saved to models/titanic_model.pkl
Training accuracy: 0.8328
=== Step 4: Evaluating model ===
Loading model from models/titanic_model.pkl
Loading test data from data/features/train_features.csv
Making predictions...
Accuracy: 0.8328
Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.88      0.87       549
           1       0.80      0.75      0.78       342

    accuracy                           0.83       891
   macro avg       0.83      0.82      0.82       891
weighted avg       0.83      0.83      0.83       891

Metrics saved to metrics/metrics.json
=== Pipeline completed! ===