# End-to-End Machine Learning Project

A comprehensive ML project demonstrating the complete pipeline from data loading to model deployment.

## Project Overview

This project showcases a professional end-to-end machine learning workflow including:

- **Data Pipeline**: Automated data loading with multiple dataset options
- **Feature Engineering**: Preprocessing, scaling, and feature selection
- **Model Training**: Multiple algorithms with comparison framework
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Deployment Simulation**: Model packaging and API endpoint simulation
- **Monitoring**: Basic model monitoring and drift detection

## Project Structure

```
project-paper-pitch/
├── data/
│   ├── raw/           # Raw datasets
│   ├── processed/     # Processed datasets
│   └── external/      # External data sources
├── src/
│   ├── data/          # Data loading and management
│   ├── features/      # Feature engineering and preprocessing
│   ├── models/        # Model training and deployment
│   └── visualization/ # Plotting and reporting
├── models/            # Saved trained models
├── notebooks/         # Jupyter notebooks for exploration
├── reports/
│   ├── figures/       # Generated plots and visualizations
│   └── presentations/ # Project presentations
├── config/            # Configuration files
├── main.py            # Main execution script
└── requirements.txt   # Python dependencies
```

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd project-paper-pitch

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# Run with Boston housing dataset (regression)
python main.py

# Run with Wine dataset (classification)
python main.py wine

# Run with Breast Cancer dataset (classification)
python main.py breast_cancer

# Run all datasets for comparison
python main.py --all-datasets
```

### 3. Explore with Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook notebooks/ml_project_demo.ipynb
```

## Available Datasets

1. **Boston Housing** (Regression)
   - Predicting house prices
   - 13 features, 506 samples
   - Metric: R² Score

2. **Wine Classification** (Classification)
   - Wine quality classification
   - 13 features, 178 samples
   - Metric: Accuracy

3. **Breast Cancer** (Classification)
   - Cancer diagnosis prediction
   - 30 features, 569 samples
   - Metric: Accuracy

## Machine Learning Models

The project trains and compares multiple algorithms:

### Regression Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression
- K-Nearest Neighbors
- Multi-layer Perceptron

### Classification Models
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine
- K-Nearest Neighbors
- Multi-layer Perceptron

## Key Features

### 1. Automated Preprocessing
- **Scaling**: Standard, MinMax, or Robust scaling
- **Feature Selection**: Statistical feature selection
- **Dimensionality Reduction**: PCA (optional)

### 2. Model Comparison
- Automated training of multiple models
- Performance comparison and ranking
- Cross-validation support
- Hyperparameter tuning

### 3. Comprehensive Visualizations
- Data distribution plots
- Model performance comparison
- Feature importance analysis
- Prediction vs actual plots
- Confusion matrices for classification

### 4. Deployment Simulation
- Model packaging for deployment
- API endpoint simulation
- Health checks
- Inference time monitoring

### 5. Model Monitoring
- Prediction logging
- Data drift detection
- Performance monitoring

## Usage Examples

### Basic Usage
```python
from src.data.data_loader import DataLoader
from src.models.model_trainer import ModelTrainer

# Load data
loader = DataLoader('boston')
X_train, X_test, y_train, y_test, df = loader.load_and_split()

# Train models
trainer = ModelTrainer(loader.problem_type)
results = trainer.train_all_models(X_train, y_train, X_test, y_test)

# Get best model
best_model = trainer.get_best_model()
```

### Deployment Simulation
```python
from src.models.model_deployment import ModelDeployment

deployment = ModelDeployment()
# ... (after model training)
api_response = deployment.simulate_api_endpoint(test_sample)
```

## Output Files

After running the pipeline, you'll find:

- **Models**: `models/` - Saved trained models and preprocessors
- **Visualizations**: `reports/figures/` - All generated plots
- **Data**: `data/raw/` - Raw datasets in CSV format

## Performance Metrics

The project automatically evaluates models using appropriate metrics:

- **Regression**: R² Score, MSE, RMSE
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Cross-validation**: 5-fold CV scores with standard deviation

## Customization

### Adding New Datasets
1. Extend `DataLoader` class in `src/data/data_loader.py`
2. Add dataset loading logic
3. Specify problem type (regression/classification)

### Adding New Models
1. Add model to `get_models()` method in `src/models/model_trainer.py`
2. Include any necessary hyperparameters
3. Models will be automatically evaluated

### Custom Preprocessing
Modify `FeaturePreprocessor` in `src/features/preprocessing.py`:
- Add new scaling methods
- Implement custom feature engineering
- Add feature selection techniques

## Project Highlights for Academic Presentation

1. **Professional Structure**: Industry-standard project organization
2. **Multiple Problem Types**: Both regression and classification examples
3. **Comprehensive Pipeline**: Data → Features → Models → Deployment
4. **Model Comparison**: Systematic evaluation of multiple algorithms
5. **Visualization**: Professional plots and analysis
6. **Deployment Ready**: Model packaging and API simulation
7. **Reproducible**: Consistent results with random seeds
8. **Scalable**: Easy to extend with new datasets and models

## Technical Skills Demonstrated

- **Python Programming**: Object-oriented design, modular code
- **Machine Learning**: Scikit-learn, model selection, evaluation
- **Data Science**: Pandas, NumPy, statistical analysis
- **Visualization**: Matplotlib, Seaborn, professional plots
- **Software Engineering**: Project structure, documentation, testing
- **MLOps**: Model deployment, monitoring, API design

This project demonstrates a complete understanding of the machine learning lifecycle and software engineering best practices.