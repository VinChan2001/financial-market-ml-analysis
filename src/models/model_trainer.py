import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self, problem_type='regression'):
        self.problem_type = problem_type
        self.models = {}
        self.best_model = None
        self.model_results = {}

    def get_models(self):
        if self.problem_type == 'regression':
            return {
                'linear_regression': LinearRegression(),
                'ridge': Ridge(),
                'lasso': Lasso(),
                'random_forest': RandomForestRegressor(random_state=42),
                'gradient_boosting': GradientBoostingRegressor(random_state=42),
                'svr': SVR(),
                'knn': KNeighborsRegressor(),
                'mlp': MLPRegressor(random_state=42, max_iter=500)
            }
        else:
            return {
                'logistic_regression': LogisticRegression(random_state=42),
                'random_forest': RandomForestClassifier(random_state=42),
                'gradient_boosting': GradientBoostingClassifier(random_state=42),
                'svc': SVC(random_state=42),
                'knn': KNeighborsClassifier(),
                'mlp': MLPClassifier(random_state=42, max_iter=500)
            }

    def train_all_models(self, X_train, y_train, X_test, y_test):
        models = self.get_models()
        results = {}

        for name, model in models.items():
            print(f"Training {name}...")

            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            if self.problem_type == 'regression':
                train_score = r2_score(y_train, train_pred)
                test_score = r2_score(y_test, test_pred)
                mse = mean_squared_error(y_test, test_pred)
                rmse = np.sqrt(mse)

                results[name] = {
                    'model': model,
                    'train_r2': train_score,
                    'test_r2': test_score,
                    'mse': mse,
                    'rmse': rmse,
                    'train_predictions': train_pred,
                    'test_predictions': test_pred
                }
            else:
                train_score = accuracy_score(y_train, train_pred)
                test_score = accuracy_score(y_test, test_pred)

                results[name] = {
                    'model': model,
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'classification_report': classification_report(y_test, test_pred),
                    'confusion_matrix': confusion_matrix(y_test, test_pred),
                    'train_predictions': train_pred,
                    'test_predictions': test_pred
                }

        self.model_results = results
        return results

    def get_best_model(self):
        if not self.model_results:
            raise ValueError("No models trained yet. Run train_all_models first.")

        if self.problem_type == 'regression':
            best_name = max(self.model_results.keys(),
                          key=lambda x: self.model_results[x]['test_r2'])
            best_score = self.model_results[best_name]['test_r2']
            metric = 'R²'
        else:
            best_name = max(self.model_results.keys(),
                          key=lambda x: self.model_results[x]['test_accuracy'])
            best_score = self.model_results[best_name]['test_accuracy']
            metric = 'Accuracy'

        self.best_model = {
            'name': best_name,
            'model': self.model_results[best_name]['model'],
            'score': best_score,
            'metric': metric
        }

        return self.best_model

    def hyperparameter_tuning(self, X_train, y_train, model_name='random_forest'):
        models = self.get_models()
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found")

        if self.problem_type == 'regression':
            if model_name == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
                scoring = 'r2'
        else:
            if model_name == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
                scoring = 'accuracy'

        grid_search = GridSearchCV(
            models[model_name], param_grid, cv=5, scoring=scoring, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    def cross_validate_model(self, X, y, model_name, cv=5):
        models = self.get_models()
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found")

        scoring = 'r2' if self.problem_type == 'regression' else 'accuracy'
        scores = cross_val_score(models[model_name], X, y, cv=cv, scoring=scoring)

        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'all_scores': scores
        }

    def save_models(self, filepath='models/trained_models.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model_results, f)

    def generate_model_summary(self):
        if not self.model_results:
            return "No models trained yet."

        summary = []
        summary.append(f"Model Training Summary - {self.problem_type.title()}")
        summary.append("=" * 50)

        for name, results in self.model_results.items():
            summary.append(f"\n{name.upper()}:")
            if self.problem_type == 'regression':
                summary.append(f"  Train R²: {results['train_r2']:.4f}")
                summary.append(f"  Test R²: {results['test_r2']:.4f}")
                summary.append(f"  RMSE: {results['rmse']:.4f}")
            else:
                summary.append(f"  Train Accuracy: {results['train_accuracy']:.4f}")
                summary.append(f"  Test Accuracy: {results['test_accuracy']:.4f}")

        best_model = self.get_best_model()
        summary.append(f"\nBest Model: {best_model['name']}")
        summary.append(f"Best {best_model['metric']}: {best_model['score']:.4f}")

        return "\n".join(summary)