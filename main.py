#!/usr/bin/env python3

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')

from data.data_loader import DataLoader
from features.preprocessing import FeaturePreprocessor
from models.model_trainer import ModelTrainer
from models.model_deployment import ModelDeployment, ModelMonitoring
from visualization.visualizer import MLVisualizer

def run_complete_ml_pipeline(dataset_name='boston'):
    print("=" * 60)
    print("          END-TO-END ML PROJECT PIPELINE")
    print("=" * 60)

    print("\n1. LOADING AND EXPLORING DATA")
    print("-" * 40)

    loader = DataLoader(dataset_name)
    X_train, X_test, y_train, y_test, df = loader.load_and_split()

    print(f"Dataset: {dataset_name}")
    print(f"Problem Type: {loader.problem_type}")
    print(f"Dataset Shape: {df.shape}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Test Samples: {X_test.shape[0]}")

    print("\n2. FEATURE PREPROCESSING")
    print("-" * 40)

    preprocessor = FeaturePreprocessor(loader.problem_type)
    X_train_processed, X_test_processed = preprocessor.full_preprocessing_pipeline(
        X_train, y_train, X_test,
        scaler_type='standard',
        feature_selection=True,
        k_features=min(10, X_train.shape[1])
    )

    print(f"Original Features: {X_train.shape[1]}")
    print(f"Processed Features: {X_train_processed.shape[1]}")
    print("Preprocessing: Scaling + Feature Selection")

    print("\n3. MODEL TRAINING AND EVALUATION")
    print("-" * 40)

    trainer = ModelTrainer(loader.problem_type)
    model_results = trainer.train_all_models(X_train_processed, y_train, X_test_processed, y_test)

    print(trainer.generate_model_summary())

    print("\n4. MODEL DEPLOYMENT SIMULATION")
    print("-" * 40)

    best_model_info = trainer.get_best_model()
    best_model = best_model_info['model']

    deployment = ModelDeployment()

    preprocessors = {
        'scaler': preprocessor.fitted_scaler,
        'feature_selector': preprocessor.feature_selector,
        'pca': preprocessor.pca
    }

    metadata = {
        'dataset': dataset_name,
        'problem_type': loader.problem_type,
        'best_model': best_model_info['name'],
        'performance': best_model_info['score'],
        'features': X_train.shape[1],
        'training_samples': X_train.shape[0]
    }

    model_path = deployment.save_model_for_deployment(best_model, preprocessors, metadata)

    deployment.load_model_for_inference(model_path)

    test_sample = X_test.iloc[0:1] if hasattr(X_test, 'iloc') else X_test[0:1]
    api_response = deployment.simulate_api_endpoint(test_sample)

    print("API Simulation Results:")
    print(f"  Status: {api_response['status']}")
    print(f"  Prediction: {api_response.get('prediction', 'N/A')}")
    print(f"  Inference Time: {api_response.get('inference_time_ms', 'N/A')} ms")

    health_status = deployment.health_check()
    print(f"  Health Check: {health_status['status']}")

    print("\n5. GENERATING VISUALIZATIONS")
    print("-" * 40)

    visualizer = MLVisualizer()
    visualizer.create_comprehensive_report(model_results, df, y_test, y_test, loader.problem_type)

    print("\n6. PROJECT SUMMARY")
    print("-" * 40)
    print(f"✓ Dataset: {dataset_name} ({loader.problem_type})")
    print(f"✓ Models Trained: {len(model_results)}")
    print(f"✓ Best Model: {best_model_info['name']}")
    print(f"✓ Best {best_model_info['metric']}: {best_model_info['score']:.4f}")
    print(f"✓ Model Deployed: {model_path}")
    print("✓ Visualizations: reports/figures/")
    print("✓ API Simulation: Functional")

    print("\n" + "=" * 60)
    print("         PROJECT PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return {
        'loader': loader,
        'preprocessor': preprocessor,
        'trainer': trainer,
        'deployment': deployment,
        'visualizer': visualizer,
        'results': model_results,
        'best_model': best_model_info
    }

def demonstrate_different_datasets():
    print("\n" + "=" * 80)
    print("           DEMONSTRATING MULTIPLE DATASETS")
    print("=" * 80)

    datasets = ['boston', 'wine', 'breast_cancer']
    all_results = {}

    for dataset in datasets:
        print(f"\n{'='*20} {dataset.upper()} DATASET {'='*20}")
        try:
            results = run_complete_ml_pipeline(dataset)
            all_results[dataset] = results
        except Exception as e:
            print(f"Error with {dataset} dataset: {str(e)}")

    print("\n" + "=" * 80)
    print("               MULTI-DATASET SUMMARY")
    print("=" * 80)

    for dataset, results in all_results.items():
        best_model = results['best_model']
        print(f"{dataset.upper():<15} | {best_model['name']:<20} | {best_model['metric']}: {best_model['score']:.4f}")

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == '--all-datasets':
        demonstrate_different_datasets()
    else:
        dataset = sys.argv[1] if len(sys.argv) > 1 else 'boston'
        run_complete_ml_pipeline(dataset)