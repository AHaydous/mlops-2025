import pandas as pd
from .base_preprocessor import BasePreprocessor

class TitanicPreprocessor(BasePreprocessor):
    """Titanic-specific preprocessor."""
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Copy EXACT logic from your preprocess.py
        processed_df = df.copy()
        
        # Drop Cabin column (too many missing values)
        if 'Cabin' in processed_df.columns:
            processed_df = processed_df.drop(columns=['Cabin'])
        
        # Handle missing values
        if 'Embarked' in processed_df.columns:
            processed_df = processed_df.assign(Embarked=processed_df['Embarked'].fillna('S'))
        
        if 'Fare' in processed_df.columns:
            processed_df = processed_df.assign(Fare=processed_df['Fare'].fillna(processed_df['Fare'].mean()))
        
        # Handle Age missing values using median by Sex and Pclass
        if 'Age' in processed_df.columns:
            age_median_by_group = processed_df.groupby(['Sex', 'Pclass'])['Age'].median()
            
            def fill_age(row):
                if pd.isna(row['Age']):
                    return age_median_by_group[row['Sex'], row['Pclass']]
                return row['Age']
            
            processed_df = processed_df.assign(Age=processed_df.apply(fill_age, axis=1))
            processed_df = processed_df.assign(Age=processed_df['Age'].astype('int64'))
        
        return processed_df