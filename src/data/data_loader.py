import pandas as pd
import numpy as np
from sklearn.datasets import load_boston, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
import os
import pickle

class DataLoader:
    def __init__(self, dataset_name='boston'):
        self.dataset_name = dataset_name
        self.data_dir = 'data/raw'

    def load_dataset(self):
        if self.dataset_name == 'boston':
            data = load_boston()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            self.problem_type = 'regression'

        elif self.dataset_name == 'wine':
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            self.problem_type = 'classification'

        elif self.dataset_name == 'breast_cancer':
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            self.problem_type = 'classification'

        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")

        return df

    def save_raw_data(self, df):
        os.makedirs(self.data_dir, exist_ok=True)
        filepath = os.path.join(self.data_dir, f'{self.dataset_name}.csv')
        df.to_csv(filepath, index=False)
        print(f"Raw data saved to {filepath}")

    def load_and_split(self, test_size=0.2, random_state=42):
        df = self.load_dataset()
        self.save_raw_data(df)

        X = df.drop('target', axis=1)
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if self.problem_type == 'classification' else None
        )

        return X_train, X_test, y_train, y_test, df

if __name__ == "__main__":
    loader = DataLoader('boston')
    X_train, X_test, y_train, y_test, df = loader.load_and_split()
    print(f"Dataset shape: {df.shape}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")