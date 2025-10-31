import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from .base_model import BaseModel

class LogisticModel(BaseModel):
    """Logistic Regression implementation using YOUR exact training logic."""
    
    def __init__(self):
        self.model = None
        self.categorical_cols = None
        self.numerical_cols = None
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        # YOUR EXACT LOGIC FROM train.py
        # Drop PassengerId for training
        if 'PassengerId' in X.columns:
            X = X.drop('PassengerId', axis=1)
        
        # Identify categorical and numerical columns (YOUR EXACT LOGIC)
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessing pipeline (YOUR EXACT LOGIC)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols)
            ])
        
        # Create model pipeline (YOUR EXACT LOGIC)
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # Train the model
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        # Drop PassengerId if present
        if 'PassengerId' in X.columns:
            X = X.drop('PassengerId', axis=1)
        return self.model.predict(X)
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)