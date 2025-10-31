import pandas as pd
from .base_features_computer import BaseFeaturesComputer

def family_size_improved(number):
    """CHANGED FEATURE: Different binning strategy - YOUR EXACT FUNCTION"""
    if number == 1:
        return "Alone"
    elif number == 2:
        return "Couple"
    elif 3 <= number <= 4:
        return "Small_Family"
    else:
        return "Large_Family"

class TitanicFeaturesComputer(BaseFeaturesComputer):
    """Titanic-specific feature engineering."""
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # COPY YOUR EXACT LOGIC FROM featurize.py
        features_df = df.copy()
        
        # Title feature extraction (YOUR EXACT LOGIC)
        features_df['Title'] = features_df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
        features_df['Title'] = features_df['Title'].replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        features_df['Title'] = features_df['Title'].replace('Mlle', 'Miss')
        features_df['Title'] = features_df['Title'].replace('Ms', 'Miss')
        features_df['Title'] = features_df['Title'].replace('Mme', 'Mrs')
        
        # Family_size feature (YOUR EXACT LOGIC)
        features_df['Family_size'] = features_df['SibSp'] + features_df['Parch'] + 1
        features_df['Family_size'] = features_df['Family_size'].apply(family_size_improved)
        
        # Drop unnecessary columns (YOUR EXACT LOGIC)
        columns_to_drop = ['Name', 'SibSp', 'Parch', 'Ticket']
        features_df.drop(columns=[col for col in columns_to_drop if col in features_df.columns], inplace=True)
        
        return features_df