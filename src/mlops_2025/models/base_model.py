from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    """Abstract base class for ML models."""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the model on given data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions on given data."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save the model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load the model from disk."""
        pass