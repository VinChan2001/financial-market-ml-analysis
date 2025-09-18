import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class FinancialDataLoader:
    def __init__(self):
        self.data_dir = 'data/raw'
        self.companies = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'NVDA': 'NVIDIA Corporation',
            'JPM': 'JPMorgan Chase & Co.',
            'JNJ': 'Johnson & Johnson',
            'V': 'Visa Inc.',
            'PG': 'Procter & Gamble'
        }

    def get_stock_data(self, symbols=None, period='2y'):
        if symbols is None:
            symbols = list(self.companies.keys())

        stock_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                info = ticker.info

                stock_data[symbol] = {
                    'price_data': hist,
                    'company_info': info,
                    'company_name': self.companies.get(symbol, symbol)
                }
                print(f"✓ Downloaded data for {symbol}")
            except Exception as e:
                print(f"✗ Error downloading {symbol}: {e}")

        return stock_data

    def create_financial_features(self, price_data):
        df = price_data.copy()

        # Technical indicators
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(5)
        df['Price_Change_10d'] = df['Close'].pct_change(10)

        # Volatility
        df['Volatility_10d'] = df['Price_Change'].rolling(window=10).std()
        df['Volatility_30d'] = df['Price_Change'].rolling(window=30).std()

        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

        # High-Low indicators
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

        return df

    def create_prediction_targets(self, df, prediction_days=5):
        # Future price movement (classification)
        df['Future_Return'] = df['Close'].shift(-prediction_days) / df['Close'] - 1
        df['Price_Direction'] = (df['Future_Return'] > 0).astype(int)  # 1 for up, 0 for down

        # Future price (regression)
        df['Future_Price'] = df['Close'].shift(-prediction_days)

        # Volatility prediction
        df['Future_Volatility'] = df['Price_Change'].shift(-prediction_days).rolling(window=5).std()

        return df

    def get_company_fundamentals(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            fundamentals = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'ebitda': info.get('ebitda', 0),
                'revenue': info.get('totalRevenue', 0),
                'gross_profit': info.get('grossProfits', 0),
                'net_income': info.get('netIncomeToCommon', 0),
                'total_debt': info.get('totalDebt', 0),
                'total_cash': info.get('totalCash', 0),
                'book_value': info.get('bookValue', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0),
                'beta': info.get('beta', 1.0),
                'dividend_yield': info.get('dividendYield', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }

            return fundamentals
        except Exception as e:
            print(f"Error getting fundamentals for {symbol}: {e}")
            return {}

    def create_sector_analysis_dataset(self):
        sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
            'Consumer': ['AMZN', 'TSLA', 'PG'],
            'Financial': ['JPM', 'V'],
            'Healthcare': ['JNJ']
        }

        sector_data = {}
        for sector, symbols in sectors.items():
            sector_performances = []
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='1y')
                    if not hist.empty:
                        performance = (hist['Close'][-1] / hist['Close'][0] - 1) * 100
                        sector_performances.append(performance)
                except:
                    continue

            if sector_performances:
                sector_data[sector] = {
                    'avg_performance': np.mean(sector_performances),
                    'volatility': np.std(sector_performances),
                    'companies': symbols
                }

        return sector_data

    def create_economic_dataset(self):
        """Create dataset for economic analysis and prediction"""

        # Get stock data
        stock_data = self.get_stock_data()

        combined_data = []

        for symbol, data in stock_data.items():
            if 'price_data' in data and not data['price_data'].empty:
                # Create features
                df_features = self.create_financial_features(data['price_data'])
                df_with_targets = self.create_prediction_targets(df_features)

                # Add company info
                df_with_targets['Symbol'] = symbol
                df_with_targets['Company'] = self.companies.get(symbol, symbol)

                # Add fundamentals
                fundamentals = self.get_company_fundamentals(symbol)
                for key, value in fundamentals.items():
                    df_with_targets[f'Fund_{key}'] = value

                combined_data.append(df_with_targets)

        if combined_data:
            final_dataset = pd.concat(combined_data, ignore_index=True)

            # Remove rows with NaN in target variables
            final_dataset = final_dataset.dropna(subset=['Price_Direction', 'Future_Price'])

            # Save dataset
            os.makedirs(self.data_dir, exist_ok=True)
            final_dataset.to_csv(os.path.join(self.data_dir, 'financial_dataset.csv'), index=False)

            return final_dataset
        else:
            print("No valid data collected")
            return pd.DataFrame()

    def get_market_indices(self):
        """Get major market indices for economic context"""
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^VIX': 'VIX (Volatility Index)'
        }

        index_data = {}
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2y')
                index_data[symbol] = {
                    'name': name,
                    'data': hist
                }
                print(f"✓ Downloaded {name} data")
            except Exception as e:
                print(f"✗ Error downloading {name}: {e}")

        return index_data

# Example usage and data generation
if __name__ == "__main__":
    loader = FinancialDataLoader()

    print("Creating comprehensive financial dataset...")
    dataset = loader.create_economic_dataset()

    if not dataset.empty:
        print(f"\nDataset created successfully!")
        print(f"Shape: {dataset.shape}")
        print(f"Companies: {dataset['Symbol'].unique()}")
        print(f"Date range: {dataset.index.min()} to {dataset.index.max()}")
        print(f"Features: {len([col for col in dataset.columns if col not in ['Symbol', 'Company']])}")

    # Create sector analysis
    print("\nAnalyzing sector performance...")
    sector_data = loader.create_sector_analysis_dataset()
    for sector, data in sector_data.items():
        print(f"{sector}: {data['avg_performance']:.2f}% (σ: {data['volatility']:.2f}%)")

    # Get market indices
    print("\nDownloading market indices...")
    indices = loader.get_market_indices()