## Branches Created:
- feature/preprocess-cli
- feature/featurize-cli 
- feature/train-cli
- feature/eval-cli
- feature/predict-cli
- feature/pipeline-runner

## Results From Git Bash:

User@DESKTOP-V121TQ0 MINGW64 ~/Desktop/mlops-2025 (master)
$ chmod +x run_pipeline.sh

User@DESKTOP-V121TQ0 MINGW64 ~/Desktop/mlops-2025 (master)
$ ./run_pipeline.sh
Starting Titanic ML Pipeline...
=== Step 1: Preprocessing data ===
Reading data from data/raw/train.csv
Original shape: (891, 12)
Processed shape: (891, 11)
Missing values after processing:
PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age            0
SibSp          0
Parch          0
Ticket         0
Fare           0
Embarked       0
dtype: int64
Processed data saved to data/processed/train_processed.csv
=== Step 2: Feature engineering ===
Reading processed data from data/processed/train_processed.csv
Input shape: (891, 11)
Basic features shape: (891, 11)
Basic features columns: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']
Basic features saved to data/features/train_features.csv
=== Step 3: Training model ===
Reading features from data/features/train_features.csv
Data shape: (891, 11)
Training set: (712, 9)
Test set: (179, 9)
Categorical columns: ['Name', 'Sex', 'Ticket', 'Embarked']
Numerical columns: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
Training model...
Model saved to models/titanic_model.pkl
Training accuracy: 0.9256
Test accuracy: 0.8268
Test set saved to models/titanic_model_test_set.csv
=== Step 4: Evaluating model ===
Loading model from models/titanic_model.pkl
Loading test data from models/titanic_model_test_set.csv
Making predictions...
Accuracy: 0.8268
Classification Report:
              precision    recall  f1-score   support

           0       0.83      0.91      0.87       110
           1       0.83      0.70      0.76        69

    accuracy                           0.83       179
   macro avg       0.83      0.80      0.81       179
weighted avg       0.83      0.83      0.82       179

Metrics saved to metrics/metrics.json
=== Pipeline completed! ===
Pipeline finished — press Enter to close...

User@DESKTOP-V121TQ0 MINGW64 ~/Desktop/mlops-2025 (master)
$ ./run_pipeline.sh
Starting Titanic ML Pipeline...
=== Step 1: Preprocessing data ===
Reading data from data/raw/train.csv
Original shape: (891, 12)
Processed shape: (891, 11)
Missing values after processing:
PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age            0
SibSp          0
Parch          0
Ticket         0
Fare           0
Embarked       0
dtype: int64
Processed data saved to data/processed/train_processed.csv
=== Step 2: Feature engineering ===
Reading processed data from data/processed/train_processed.csv
Input shape: (891, 11)
Enhanced features shape: (891, 9)
Enhanced features columns: ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'Family_size']
Enhanced features saved to data/features/train_features.csv
=== Step 3: Training model ===
Reading features from data/features/train_features.csv
Data shape: (891, 9)
Training set: (712, 7)
Test set: (179, 7)
Categorical columns: ['Sex', 'Embarked', 'Title', 'Family_size']
Numerical columns: ['Pclass', 'Age', 'Fare']
Training model...
Model saved to models/titanic_model.pkl
Training accuracy: 0.8301
Test accuracy: 0.8436
Test set saved to models/titanic_model_test_set.csv
=== Step 4: Evaluating model ===
Loading model from models/titanic_model.pkl
Loading test data from models/titanic_model_test_set.csv
Making predictions...
Accuracy: 0.8436
Classification Report:
              precision    recall  f1-score   support

           0       0.86      0.89      0.88       110
           1       0.82      0.77      0.79        69

    accuracy                           0.84       179
   macro avg       0.84      0.83      0.83       179
weighted avg       0.84      0.84      0.84       179

Metrics saved to metrics/metrics.json
=== Pipeline completed! ===
Pipeline finished — press Enter to close...

User@DESKTOP-V121TQ0 MINGW64 ~/Desktop/mlops-2025 (master)
$ ./run_pipeline.sh
Starting Titanic ML Pipeline...
=== Step 1: Preprocessing data ===
Reading data from data/raw/train.csv
Original shape: (891, 12)
Processed shape: (891, 11)
Missing values after processing:
PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age            0
SibSp          0
Parch          0
Ticket         0
Fare           0
Embarked       0
dtype: int64
Processed data saved to data/processed/train_processed.csv
=== Step 2: Feature engineering ===
Reading processed data from data/processed/train_processed.csv
Input shape: (891, 11)
Improved features shape: (891, 9)
Improved features columns: ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'Family_size']
Improved features saved to data/features/train_features.csv
=== Step 3: Training model ===
Reading features from data/features/train_features.csv
Data shape: (891, 9)
Training set: (712, 7)
Test set: (179, 7)
Categorical columns: ['Sex', 'Embarked', 'Title', 'Family_size']
Numerical columns: ['Pclass', 'Age', 'Fare']
Training model...
Model saved to models/titanic_model.pkl
Training accuracy: 0.8329
Test accuracy: 0.8380
Test set saved to models/titanic_model_test_set.csv
=== Step 4: Evaluating model ===
Loading model from models/titanic_model.pkl
Loading test data from models/titanic_model_test_set.csv
Making predictions...
Accuracy: 0.8380
Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.89      0.87       110
           1       0.81      0.75      0.78        69

    accuracy                           0.84       179
   macro avg       0.83      0.82      0.83       179
weighted avg       0.84      0.84      0.84       179

Metrics saved to metrics/metrics.json
=== Pipeline completed! ===
Pipeline finished — press Enter to close...




Answers for Section 4:

1. Run the pipeline
Executed ./run_pipeline.sh - achieved 82.68% test accuracy

2. Add some features, rerun the pipeline
Added Title + Family_size features - improved to 84.36% test accuracy

3. What can we improve?
Several improvements could enhance the pipeline further:
	.Implement cross-validation for more robust model evaluation
	.Experiment with different machine learning algorithms like Random Forest or Gradient Boosting
	.Add hyperparameter tuning to optimize model performance
	.Create additional features like age bins or fare categories
	.Address potential class imbalance in the survival labels
	.Add more comprehensive logging and experiment tracking

5. Change another feature, rerun the pipeline
Modified Family_size binning - maintained 83.80% test accuracy