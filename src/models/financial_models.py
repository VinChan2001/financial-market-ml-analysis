import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FinancialModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_models = {}

    def get_price_prediction_models(self):
        """Models for predicting future stock prices (regression)"""
        return {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'lasso_regression': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf'),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }

    def get_direction_prediction_models(self):
        """Models for predicting price direction (classification)"""
        return {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svc': SVC(kernel='rbf', random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }

    def prepare_financial_features(self, df):
        """Prepare features for financial prediction"""
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'SMA_30', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
            'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width',
            'Price_Change', 'Price_Change_5d', 'Price_Change_10d',
            'Volatility_10d', 'Volatility_30d', 'Volume_Ratio',
            'High_Low_Pct', 'Close_Position'
        ]

        # Add fundamental analysis features if available
        fundamental_features = [col for col in df.columns if col.startswith('Fund_')]
        feature_columns.extend(fundamental_features)

        # Select only available features and ensure they are numeric
        available_features = []
        for col in feature_columns:
            if col in df.columns:
                # Check if column is numeric
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    available_features.append(col)

        X = df[available_features]

        # Remove rows with NaN values
        X = X.dropna()

        return X, available_features

    def time_series_split_data(self, X, y, test_size=0.2):
        """Split data respecting time series nature"""
        split_index = int(len(X) * (1 - test_size))

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        return X_train, X_test, y_train, y_test

    def train_price_prediction_models(self, df, target_column='Future_Price'):
        """Train models to predict future stock prices"""
        print("Training Price Prediction Models...")

        X, feature_names = self.prepare_financial_features(df)
        y = df[target_column].loc[X.index]

        # Remove NaN targets
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]

        if len(X) < 50:
            print("Insufficient data for training")
            return {}

        # Time series split
        X_train, X_test, y_train, y_test = self.time_series_split_data(X, y)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = self.get_price_prediction_models()
        results = {}

        for name, model in models.items():
            try:
                print(f"  Training {name}...")
                model.fit(X_train_scaled, y_train)

                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)

                results[name] = {
                    'model': model,
                    'scaler': scaler,
                    'feature_names': feature_names,
                    'train_r2': r2_score(y_train, train_pred),
                    'test_r2': r2_score(y_test, test_pred),
                    'train_mse': mean_squared_error(y_train, train_pred),
                    'test_mse': mean_squared_error(y_test, test_pred),
                    'train_mae': mean_absolute_error(y_train, train_pred),
                    'test_mae': mean_absolute_error(y_test, test_pred),
                    'predictions': test_pred,
                    'actual': y_test.values
                }

            except Exception as e:
                print(f"  Error training {name}: {e}")

        self.results['price_prediction'] = results
        return results

    def train_direction_prediction_models(self, df, target_column='Price_Direction'):
        """Train models to predict price direction (up/down)"""
        print("Training Direction Prediction Models...")

        X, feature_names = self.prepare_financial_features(df)
        y = df[target_column].loc[X.index]

        # Remove NaN targets
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]

        if len(X) < 50:
            print("Insufficient data for training")
            return {}

        # Time series split
        X_train, X_test, y_train, y_test = self.time_series_split_data(X, y)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = self.get_direction_prediction_models()
        results = {}

        for name, model in models.items():
            try:
                print(f"  Training {name}...")
                model.fit(X_train_scaled, y_train)

                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)

                results[name] = {
                    'model': model,
                    'scaler': scaler,
                    'feature_names': feature_names,
                    'train_accuracy': accuracy_score(y_train, train_pred),
                    'test_accuracy': accuracy_score(y_test, test_pred),
                    'classification_report': classification_report(y_test, test_pred),
                    'predictions': test_pred,
                    'actual': y_test.values
                }

                # Add probability predictions if available
                if hasattr(model, 'predict_proba'):
                    test_proba = model.predict_proba(X_test_scaled)
                    results[name]['probabilities'] = test_proba

            except Exception as e:
                print(f"  Error training {name}: {e}")

        self.results['direction_prediction'] = results
        return results

    def get_best_models(self):
        """Get best performing models for each task"""
        best_models = {}

        if 'price_prediction' in self.results:
            best_price_model = max(
                self.results['price_prediction'].items(),
                key=lambda x: x[1]['test_r2']
            )
            best_models['price_prediction'] = {
                'name': best_price_model[0],
                'model': best_price_model[1]['model'],
                'performance': best_price_model[1]['test_r2'],
                'metric': 'R²'
            }

        if 'direction_prediction' in self.results:
            best_direction_model = max(
                self.results['direction_prediction'].items(),
                key=lambda x: x[1]['test_accuracy']
            )
            best_models['direction_prediction'] = {
                'name': best_direction_model[0],
                'model': best_direction_model[1]['model'],
                'performance': best_direction_model[1]['test_accuracy'],
                'metric': 'Accuracy'
            }

        self.best_models = best_models
        return best_models

    def generate_model_report(self):
        """Generate comprehensive model performance report"""
        report = []
        report.append("FINANCIAL ML MODEL PERFORMANCE REPORT")
        report.append("=" * 50)

        if 'price_prediction' in self.results:
            report.append("\nPRICE PREDICTION MODELS (Regression):")
            report.append("-" * 30)
            for name, result in self.results['price_prediction'].items():
                report.append(f"{name.upper()}:")
                report.append(f"  Test R²: {result['test_r2']:.4f}")
                report.append(f"  Test MSE: {result['test_mse']:.4f}")
                report.append(f"  Test MAE: {result['test_mae']:.4f}")

        if 'direction_prediction' in self.results:
            report.append("\nDIRECTION PREDICTION MODELS (Classification):")
            report.append("-" * 30)
            for name, result in self.results['direction_prediction'].items():
                report.append(f"{name.upper()}:")
                report.append(f"  Test Accuracy: {result['test_accuracy']:.4f}")

        if self.best_models:
            report.append("\nBEST PERFORMING MODELS:")
            report.append("-" * 30)
            for task, model_info in self.best_models.items():
                report.append(f"{task}: {model_info['name']} ({model_info['metric']}: {model_info['performance']:.4f})")

        return "\n".join(report)

    def hyperparameter_tuning(self, df, task='price_prediction', model_name='random_forest'):
        """Perform hyperparameter tuning for specified model"""
        print(f"Performing hyperparameter tuning for {model_name} on {task}...")

        X, _ = self.prepare_financial_features(df)

        if task == 'price_prediction':
            y = df['Future_Price'].loc[X.index]
            models = self.get_price_prediction_models()
            scoring = 'r2'
        else:
            y = df['Price_Direction'].loc[X.index]
            models = self.get_direction_prediction_models()
            scoring = 'accuracy'

        # Remove NaN values
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]

        if len(X) < 50:
            print("Insufficient data for hyperparameter tuning")
            return None, None, None

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Define parameter grids
        if model_name == 'random_forest':
            if task == 'price_prediction':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
        else:
            print(f"Parameter grid not defined for {model_name}")
            return None, None, None

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(
            models[model_name],
            param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=-1
        )

        grid_search.fit(X_scaled, y)

        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_