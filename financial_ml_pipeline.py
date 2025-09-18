#!/usr/bin/env python3

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')

from data.financial_data_loader import FinancialDataLoader
from models.financial_models import FinancialModelTrainer
from analysis.economic_analyzer import EconomicAnalyzer
from visualization.financial_visualizer import FinancialVisualizer

def run_financial_ml_pipeline():
    print("=" * 80)
    print("              FINANCIAL MARKET ANALYSIS & PREDICTION SYSTEM")
    print("=" * 80)

    print("\nSTEP 1: EXTRACTING FINANCIAL DATA")
    print("-" * 50)

    # Initialize data loader
    loader = FinancialDataLoader()

    # Create comprehensive financial dataset
    print("Downloading real-time stock data from major companies...")
    df = loader.create_economic_dataset()

    if df.empty:
        print("Failed to create dataset. Please check your internet connection.")
        return

    print(f"Dataset created successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Companies: {df['Symbol'].nunique()}")
    print(f"   Date range: {len(df)} data points")
    print(f"   Features: {len([col for col in df.columns if col not in ['Symbol', 'Company']])}")

    # Get market indices for economic context
    print("\nDownloading market indices...")
    market_indices = loader.get_market_indices()

    print("\nSTEP 2: TRAINING PREDICTION MODELS")
    print("-" * 50)

    # Initialize model trainer
    trainer = FinancialModelTrainer()

    # Train stock price prediction models
    price_results = trainer.train_price_prediction_models(df)

    # Train price direction prediction models
    direction_results = trainer.train_direction_prediction_models(df)

    # Get best models
    best_models = trainer.get_best_models()

    print("\nMODEL PERFORMANCE SUMMARY:")
    if 'price_prediction' in best_models:
        print(f"   Best Price Predictor: {best_models['price_prediction']['name']}")
        print(f"      R² Score: {best_models['price_prediction']['performance']:.4f}")

    if 'direction_prediction' in best_models:
        print(f"   Best Direction Predictor: {best_models['direction_prediction']['name']}")
        print(f"      Accuracy: {best_models['direction_prediction']['performance']:.4f}")

    print("\nSTEP 3: ECONOMIC ANALYSIS")
    print("-" * 50)

    # Initialize economic analyzer
    analyzer = EconomicAnalyzer()

    # Generate comprehensive economic report
    economic_report = analyzer.generate_economic_report(df, market_indices)

    print("Analysis completed:")

    # Financial Health Analysis
    if 'financial_health' in economic_report:
        health_df = economic_report['financial_health']
        top_health = health_df.nlargest(3, 'Health_Score')
        print(f"   Top Financial Health:")
        for _, row in top_health.iterrows():
            print(f"      {row['Symbol']}: {row['Health_Score']:.1f}/100 (Rating: {row['Rating']})")

    # Sector Performance
    if 'sector_analysis' in economic_report:
        sector_data = economic_report['sector_analysis']
        best_sector = max(sector_data.items(), key=lambda x: x[1]['avg_return'])
        print(f"   Best Performing Sector: {best_sector[0]} ({best_sector[1]['avg_return']:.2f}%)")

    # Risk Assessment
    if 'risk_assessment' in economic_report and not economic_report['risk_assessment'].empty:
        risk_df = economic_report['risk_assessment']
        low_risk_count = len(risk_df[risk_df['Risk_Level'] == 'Low'])
        print(f"   Low Risk Companies: {low_risk_count}")

    print("\nSTEP 4: GENERATING VISUALIZATIONS")
    print("-" * 50)

    # Initialize visualizer
    visualizer = FinancialVisualizer()

    # Create comprehensive dashboard
    visualizer.create_financial_dashboard(economic_report, df)

    # Plot economic indicators
    visualizer.plot_economic_indicators(market_indices, 'reports/figures/economic_indicators.png')

    # Plot model predictions for best models
    if price_results:
        best_price_name = max(price_results.keys(), key=lambda x: price_results[x]['test_r2'])
        best_price_result = price_results[best_price_name]
        visualizer.plot_model_predictions(
            best_price_result['actual'],
            best_price_result['predictions'],
            best_price_name,
            'regression',
            'reports/figures/price_predictions.png'
        )

    if direction_results:
        best_direction_name = max(direction_results.keys(), key=lambda x: direction_results[x]['test_accuracy'])
        best_direction_result = direction_results[best_direction_name]
        visualizer.plot_model_predictions(
            best_direction_result['actual'],
            best_direction_result['predictions'],
            best_direction_name,
            'classification',
            'reports/figures/direction_predictions.png'
        )

    print("\nSTEP 5: INVESTMENT RECOMMENDATIONS")
    print("-" * 50)

    recommendations = analyzer.get_investment_recommendations(economic_report)
    print(recommendations)

    print("\nSTEP 6: MODEL DEPLOYMENT SIMULATION")
    print("-" * 50)

    # Save best models for deployment
    if best_models:
        import pickle
        os.makedirs('models', exist_ok=True)

        for task, model_info in best_models.items():
            model_path = f"models/best_{task}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model_info['model'],
                    'performance': model_info['performance'],
                    'task': task
                }, f)
            print(f"{task} model saved: {model_path}")

    print("\n" + "=" * 80)
    print("                     ANALYSIS COMPLETE!")
    print("=" * 80)

    print("\nPROJECT SUMMARY:")
    print(f"   Data Points Analyzed: {len(df):,}")
    print(f"   Companies Studied: {df['Symbol'].nunique()}")
    print(f"   ML Models Trained: {len(price_results) + len(direction_results)}")
    print(f"   Best Price Predictor: {best_models.get('price_prediction', {}).get('name', 'N/A')}")
    print(f"   Best Direction Predictor: {best_models.get('direction_prediction', {}).get('name', 'N/A')}")
    print(f"   Visualizations: reports/figures/")
    print(f"   Models Saved: models/")

    return {
        'dataset': df,
        'price_results': price_results,
        'direction_results': direction_results,
        'economic_report': economic_report,
        'best_models': best_models,
        'market_indices': market_indices
    }

def demonstrate_real_time_prediction():
    """Demonstrate real-time prediction capabilities"""
    print("\nREAL-TIME PREDICTION DEMO")
    print("-" * 40)

    try:
        import pickle

        # Load saved model
        with open('models/best_price_prediction_model.pkl', 'rb') as f:
            model_data = pickle.load(f)

        print(f"Loaded {model_data['task']} model")
        print(f"   Performance: {model_data['performance']:.4f}")

        # Get latest data for prediction
        loader = FinancialDataLoader()
        latest_data = loader.get_stock_data(['AAPL'], period='5d')

        if 'AAPL' in latest_data:
            print("Making prediction for AAPL...")
            print("   (In production, this would use real-time market data)")

        print("Real-time prediction simulation complete")

    except FileNotFoundError:
        print("No saved models found. Run main pipeline first.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demonstrate_real_time_prediction()
    else:
        results = run_financial_ml_pipeline()