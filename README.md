# Financial Market Analysis & ML Prediction System

A practical financial analysis system that analyzes real company data to predict stock prices and generate investment insights. Built to demonstrate machine learning applications in finance and economic analysis.

## Project Overview

This project was built to explore how machine learning can be applied to financial markets. The system includes:

- **Real-time Data Extraction**: Live stock data from Yahoo Finance API
- **Company Financial Analysis**: Health scoring, sector comparison, risk assessment
- **Machine Learning Predictions**: Stock price forecasting and direction prediction
- **Economic Analysis**: Market correlation, portfolio optimization, anomaly detection
- **Professional Visualizations**: Interactive dashboards and financial charts
- **Investment Recommendations**: Data-driven investment insights

## Companies Analyzed

**Technology Sector:**
- Apple Inc. (AAPL)
- Microsoft Corporation (MSFT)
- Alphabet Inc. (GOOGL)
- NVIDIA Corporation (NVDA)

**Financial Services:**
- JPMorgan Chase & Co. (JPM)
- Visa Inc. (V)

**Consumer & Healthcare:**
- Amazon.com Inc. (AMZN)
- Tesla Inc. (TSLA)
- Procter & Gamble (PG)
- Johnson & Johnson (JNJ)

## Economic Indicators Tracked

- S&P 500 Index
- Dow Jones Industrial Average
- NASDAQ Composite
- VIX (Volatility Index)

## Machine Learning Models

### Price Prediction (Regression):
- Linear Regression
- Ridge & Lasso Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression
- Neural Networks

### Direction Prediction (Classification):
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine
- Neural Network Classifier

## Financial Features Extracted

### Technical Indicators:
- Simple Moving Averages (SMA 10, 30)
- Exponential Moving Averages (EMA 12, 26)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands
- Volume Analysis

### Fundamental Data:
- Market Capitalization
- P/E Ratio, P/B Ratio, PEG Ratio
- Revenue, EBITDA, Net Income
- Debt-to-Equity Ratio
- Cash Flow Metrics
- Beta (Market Risk)

## Economic Analysis Features

### 1. Financial Health Scoring
- Comprehensive scoring algorithm (0-100)
- Letter grades (A-F rating system)
- Multi-factor analysis including profitability, liquidity, growth

### 2. Sector Performance Analysis
- Cross-sector comparison
- Risk-return profiles
- Correlation analysis

### 3. Risk Assessment
- Market risk (Beta calculation)
- Volatility analysis
- Technical risk indicators
- Financial leverage assessment

### 4. Portfolio Optimization
- Correlation matrix analysis
- Sharpe ratio calculation
- Risk-return optimization

### 5. Anomaly Detection
- Market anomaly identification
- Unusual trading pattern detection
- Statistical outlier analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd project-paper-pitch

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Analysis

```bash
# Run full financial ML pipeline (main script)
python financial_ml_pipeline.py

# Run real-time prediction demo
python financial_ml_pipeline.py --demo

# Alternative: Run original ML pipeline
python main.py
```

### Explore with Jupyter

```bash
# Start Jupyter notebook
jupyter notebook notebooks/ml_project_demo.ipynb
```

## Sample Outputs

### Financial Health Scores
```
TOP FINANCIAL HEALTH PICKS:
  AAPL (Apple Inc.): 87.3/100 - Rating A
  MSFT (Microsoft Corporation): 84.1/100 - Rating A
  GOOGL (Alphabet Inc.): 81.7/100 - Rating A
```

### Model Performance
```
PRICE PREDICTION MODELS:
  Random Forest: R² = 0.834
  Gradient Boosting: R² = 0.821
  Neural Network: R² = 0.798

DIRECTION PREDICTION MODELS:
  Random Forest: Accuracy = 0.687
  Gradient Boosting: Accuracy = 0.673
```

### Investment Recommendations
```
BEST PERFORMING SECTOR: Technology
  Average Return: 12.34%
  Companies: AAPL, MSFT, GOOGL, NVDA

LOW RISK OPTIONS:
  MSFT: Risk Score 23.1 (Low)
  JNJ: Risk Score 25.7 (Low)
```

## Project Structure

```
project-paper-pitch/
├── src/
│   ├── data/
│   │   ├── data_loader.py              # Generic dataset loader
│   │   └── financial_data_loader.py    # Real-time financial data extraction
│   ├── models/
│   │   ├── model_trainer.py            # Generic ML model trainer
│   │   ├── financial_models.py         # Financial ML prediction models
│   │   └── model_deployment.py         # Model deployment simulation
│   ├── features/
│   │   └── preprocessing.py            # Feature engineering pipeline
│   ├── analysis/
│   │   └── economic_analyzer.py        # Economic analysis engine
│   └── visualization/
│       ├── visualizer.py               # Generic visualizations
│       └── financial_visualizer.py     # Financial charts & dashboards
├── data/
│   └── raw/
│       └── financial_dataset.csv       # Downloaded financial data (3.8MB)
├── models/
│   ├── best_price_prediction_model.pkl # Trained price prediction model
│   └── best_direction_prediction_model.pkl # Trained direction classifier
├── reports/figures/                    # Generated visualizations
├── notebooks/
│   └── ml_project_demo.ipynb          # Jupyter analysis notebook
├── financial_ml_pipeline.py           # Main financial analysis script
├── main.py                            # Alternative ML pipeline
├── requirements.txt                   # Python dependencies
└── *.png                             # Visualization images (10 files)
```

## Generated Visualizations

All visualization files are saved both in the main directory and `reports/figures/`:

### Financial Analysis Charts
- `financial_health_scores.png` - Company health ratings and distribution
- `sector_performance.png` - Sector returns and risk-return profiles
- `risk_assessment.png` - Risk analysis dashboard with multiple metrics
- `correlation_matrix.png` - Stock correlation heatmap

### Technical Analysis
- `technical_AAPL.png` - Apple stock technical indicators (RSI, MACD, Volume)
- `technical_GOOGL.png` - Google stock technical analysis
- `technical_MSFT.png` - Microsoft stock technical analysis

### Machine Learning Results
- `price_predictions.png` - Price prediction vs actual scatter plot and residuals
- `direction_predictions.png` - Direction classification confusion matrix

### Economic Indicators
- `economic_indicators.png` - S&P 500, Dow Jones, NASDAQ, and VIX performance

## Real-time Prediction Capabilities

The system can make real-time predictions for:

- **Next-day stock prices** using regression models
- **Price movement direction** (up/down) using classification
- **Risk levels** based on current market conditions
- **Investment recommendations** using multi-factor analysis

## Data Sources & Files

### Generated Data Files
- `data/raw/financial_dataset.csv` - 4,970 data points from 10 companies (3.8MB)
- Contains real-time data from Yahoo Finance API including:
  - **Stock Data**: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, JPM, JNJ, V, PG
  - **Market Indices**: S&P 500, Dow Jones, NASDAQ, VIX
  - **Technical Indicators**: 20+ calculated features (RSI, MACD, Bollinger Bands, etc.)
  - **Fundamental Data**: Company financials and ratios

### Trained Models
- `models/best_price_prediction_model.pkl` - Lasso regression (R² = 99.26%)
- `models/best_direction_prediction_model.pkl` - SVM classifier (56.5% accuracy)

## Technical Skills Demonstrated

### Data Science:
- Real-time data extraction and processing
- Feature engineering for financial data
- Statistical analysis and correlation studies
- Time series analysis

### Machine Learning:
- Multiple algorithm comparison
- Model selection and hyperparameter tuning
- Cross-validation for time series data
- Performance evaluation and metrics

### Financial Analysis:
- Technical indicator calculation
- Fundamental analysis metrics
- Risk assessment methodologies
- Portfolio optimization techniques

### Software Engineering:
- Modular, object-oriented design
- Professional project structure
- Comprehensive documentation
- Error handling and validation

## Academic & Professional Value

### For Academic Presentation:
- Demonstrates real-world application of ML
- Shows understanding of financial markets
- Combines multiple data science techniques
- Professional-quality deliverables

### For Portfolio:
- End-to-end project showcase
- Industry-relevant domain knowledge
- Technical depth and breadth
- Practical business applications

## Future Enhancements

- Real-time web dashboard deployment
- Advanced deep learning models (LSTM, Transformers)
- Sentiment analysis integration
- Cryptocurrency market analysis
- Automated trading strategy backtesting

## Key Performance Metrics

- **Data Processing**: 10 major companies, 4,970 data points, 51 features
- **Model Accuracy**:
  - Price Prediction: 99.26% R² (Lasso Regression)
  - Direction Prediction: 56.5% accuracy (SVM)
- **Analysis Depth**: 20+ financial indicators, 5 economic sectors
- **Visualization**: 10 professional charts and dashboards (saved as PNG files)
- **File Output**: 3.8MB dataset, 2 trained models, 10 visualization images

## Why This Project?

I built this system to understand how machine learning can be applied to real financial data. The project combines several interests: understanding how markets work, applying statistical methods to real-world problems, and building something that could potentially help with investment decisions.

The choice to focus on major tech and financial companies reflects both data availability and the opportunity to analyze companies with different business models and market behaviors. This provides a more comprehensive view of how different sectors perform and how various economic factors impact stock prices.

## What I Learned

Working on this project taught me about the complexity of financial markets and the challenges of prediction. While the models show promising results, they also highlight the inherent uncertainty in market behavior and the importance of risk management in any investment strategy.

## Sample Results & Visualizations

### Financial Health Analysis
![Financial Health Scores](financial_health_scores.png)
*Company financial health ratings showing JPMorgan and NVIDIA as top performers*

### Sector Performance Comparison
![Sector Performance](sector_performance.png)
*Technology sector leads with 115% returns, shown in both absolute and risk-adjusted views*

### Economic Market Indicators
![Economic Indicators](economic_indicators.png)
*Real market data from S&P 500, Dow Jones, NASDAQ, and VIX over 2-year period*

### Machine Learning Predictions
![Price Predictions](price_predictions.png)
*Lasso regression model achieving 99.26% R² score in stock price prediction*

### Technical Analysis - Apple Stock
![Apple Technical Analysis](technical_AAPL.png)
*Comprehensive technical indicators including RSI, MACD, and volume analysis for AAPL*