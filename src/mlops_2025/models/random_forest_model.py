import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest implementation."""
    
    def __init__(self):
        self.model = None
        self.categorical_cols = None
        self.numerical_cols = None
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        # Drop PassengerId for training
        if 'PassengerId' in X.columns:
            X = X.drop('PassengerId', axis=1)
        
        # Identify categorical and numerical columns
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols)
            ])
        
        # Create Random Forest model
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Train the model
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        if 'PassengerId' in X.columns:
            X = X.drop('PassengerId', axis=1)
        return self.model.predict(X)
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)