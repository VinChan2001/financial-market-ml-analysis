import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
import os

class ModelDeployment:
    def __init__(self, model_path='models/best_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.preprocessors = None
        self.metadata = {}

    def save_model_for_deployment(self, model, preprocessors, metadata, model_path=None):
        if model_path is None:
            model_path = self.model_path

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        deployment_package = {
            'model': model,
            'preprocessors': preprocessors,
            'metadata': metadata,
            'deployment_timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }

        with open(model_path, 'wb') as f:
            pickle.dump(deployment_package, f)

        print(f"Model deployment package saved to {model_path}")
        return model_path

    def load_model_for_inference(self, model_path=None):
        if model_path is None:
            model_path = self.model_path

        try:
            with open(model_path, 'rb') as f:
                deployment_package = pickle.load(f)

            self.model = deployment_package['model']
            self.preprocessors = deployment_package['preprocessors']
            self.metadata = deployment_package['metadata']

            print(f"Model loaded successfully from {model_path}")
            print(f"Model version: {deployment_package.get('version', 'Unknown')}")
            print(f"Deployment timestamp: {deployment_package.get('deployment_timestamp', 'Unknown')}")

            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def preprocess_input(self, X):
        if self.preprocessors is None:
            raise ValueError("Preprocessors not loaded. Load model first.")

        X_processed = X.copy()

        if self.preprocessors.get('scaler'):
            X_processed = self.preprocessors['scaler'].transform(X_processed)

        if self.preprocessors.get('feature_selector'):
            X_processed = self.preprocessors['feature_selector'].transform(X_processed)

        if self.preprocessors.get('pca'):
            X_processed = self.preprocessors['pca'].transform(X_processed)

        return X_processed

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not loaded. Load model first.")

        if isinstance(X, list):
            X = np.array(X).reshape(1, -1)
        elif isinstance(X, pd.DataFrame):
            X = X.values
        elif len(X.shape) == 1:
            X = X.reshape(1, -1)

        X_processed = self.preprocess_input(X)
        predictions = self.model.predict(X_processed)

        return predictions

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not loaded. Load model first.")

        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions.")

        if isinstance(X, list):
            X = np.array(X).reshape(1, -1)
        elif isinstance(X, pd.DataFrame):
            X = X.values
        elif len(X.shape) == 1:
            X = X.reshape(1, -1)

        X_processed = self.preprocess_input(X)
        probabilities = self.model.predict_proba(X_processed)

        return probabilities

    def batch_predict(self, X_batch):
        if self.model is None:
            raise ValueError("Model not loaded. Load model first.")

        predictions = []
        for i, x in enumerate(X_batch):
            try:
                pred = self.predict(x)
                predictions.append(pred[0] if len(pred) == 1 else pred)
            except Exception as e:
                print(f"Error predicting sample {i}: {str(e)}")
                predictions.append(None)

        return predictions

    def get_model_info(self):
        if self.model is None:
            return "Model not loaded"

        info = {
            'model_type': type(self.model).__name__,
            'metadata': self.metadata,
            'preprocessors': list(self.preprocessors.keys()) if self.preprocessors else [],
            'model_params': self.model.get_params() if hasattr(self.model, 'get_params') else {}
        }

        return info

    def simulate_api_endpoint(self, input_data):
        try:
            start_time = datetime.now()

            if isinstance(input_data, dict):
                X = np.array(list(input_data.values())).reshape(1, -1)
            else:
                X = input_data

            prediction = self.predict(X)

            end_time = datetime.now()
            inference_time = (end_time - start_time).total_seconds() * 1000

            response = {
                'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                'inference_time_ms': round(inference_time, 2),
                'timestamp': end_time.isoformat(),
                'model_version': self.metadata.get('version', '1.0.0'),
                'status': 'success'
            }

            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.predict_proba(X)
                    response['probabilities'] = probabilities.tolist()
                except:
                    pass

            return response

        except Exception as e:
            return {
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }

    def health_check(self):
        try:
            test_input = np.random.random((1, 10))
            _ = self.predict(test_input)
            return {
                'status': 'healthy',
                'model_loaded': True,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

class ModelMonitoring:
    def __init__(self):
        self.prediction_log = []
        self.performance_metrics = {}

    def log_prediction(self, input_data, prediction, actual=None):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input_hash': hash(str(input_data)),
            'prediction': prediction,
            'actual': actual,
            'drift_detected': False
        }
        self.prediction_log.append(log_entry)

    def detect_data_drift(self, new_data, reference_data, threshold=0.1):
        new_mean = np.mean(new_data, axis=0)
        ref_mean = np.mean(reference_data, axis=0)

        drift_score = np.mean(np.abs(new_mean - ref_mean) / (np.std(reference_data, axis=0) + 1e-8))

        return drift_score > threshold, drift_score

    def get_monitoring_report(self):
        if not self.prediction_log:
            return "No predictions logged yet."

        total_predictions = len(self.prediction_log)
        recent_predictions = [log for log in self.prediction_log
                            if (datetime.now() - datetime.fromisoformat(log['timestamp'])).days <= 7]

        report = {
            'total_predictions': total_predictions,
            'recent_predictions_7_days': len(recent_predictions),
            'avg_predictions_per_day': total_predictions / max(1,
                (datetime.now() - datetime.fromisoformat(self.prediction_log[0]['timestamp'])).days),
            'drift_alerts': sum(1 for log in self.prediction_log if log.get('drift_detected', False))
        }

        return report