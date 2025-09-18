import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.decomposition import PCA
import pickle
import os

class FeaturePreprocessor:
    def __init__(self, problem_type='regression'):
        self.problem_type = problem_type
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        self.feature_selector = None
        self.pca = None
        self.fitted_scaler = None

    def scale_features(self, X_train, X_test, scaler_type='standard'):
        scaler = self.scalers[scaler_type]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.fitted_scaler = scaler

        return X_train_scaled, X_test_scaled

    def select_features(self, X_train, y_train, X_test, k=10):
        if self.problem_type == 'regression':
            selector = SelectKBest(score_func=f_regression, k=k)
        else:
            selector = SelectKBest(score_func=f_classif, k=k)

        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        self.feature_selector = selector

        return X_train_selected, X_test_selected

    def apply_pca(self, X_train, X_test, n_components=0.95):
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        self.pca = pca

        return X_train_pca, X_test_pca

    def full_preprocessing_pipeline(self, X_train, y_train, X_test,
                                  scaler_type='standard',
                                  feature_selection=True,
                                  apply_pca=False,
                                  k_features=10,
                                  pca_components=0.95):

        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()

        X_train_processed, X_test_processed = self.scale_features(
            X_train_processed, X_test_processed, scaler_type
        )

        if feature_selection:
            k_features = min(k_features, X_train_processed.shape[1])
            X_train_processed, X_test_processed = self.select_features(
                X_train_processed, y_train, X_test_processed, k_features
            )

        if apply_pca:
            X_train_processed, X_test_processed = self.apply_pca(
                X_train_processed, X_test_processed, pca_components
            )

        return X_train_processed, X_test_processed

    def save_preprocessors(self, filepath='models/preprocessors.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        preprocessors = {
            'scaler': self.fitted_scaler,
            'feature_selector': self.feature_selector,
            'pca': self.pca
        }
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessors, f)

    def get_feature_importance_scores(self):
        if self.feature_selector:
            return self.feature_selector.scores_
        return None