import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.dates as mdates
from datetime import datetime
import os

class FinancialVisualizer:
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        plt.style.use('seaborn-v0_8')

    def plot_stock_price_trends(self, stock_data, symbols=None, save_path=None):
        """Plot stock price trends for multiple companies"""
        if symbols is None:
            symbols = list(stock_data.keys())[:5]  # Plot first 5 if not specified

        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()

        for i, symbol in enumerate(symbols[:4]):
            if symbol in stock_data and 'price_data' in stock_data[symbol]:
                data = stock_data[symbol]['price_data']
                company_name = stock_data[symbol].get('company_name', symbol)

                axes[i].plot(data.index, data['Close'], label='Close Price', color=self.colors[0], linewidth=2)
                axes[i].fill_between(data.index, data['Low'], data['High'], alpha=0.3, color=self.colors[1])

                if 'SMA_30' in data.columns:
                    axes[i].plot(data.index, data['SMA_30'], label='30-day SMA', color=self.colors[2], linestyle='--')

                axes[i].set_title(f'{company_name} ({symbol}) Stock Price')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel('Price ($)')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_technical_indicators(self, df, symbol, save_path=None):
        """Plot technical indicators for a specific stock"""
        symbol_data = df[df['Symbol'] == symbol].copy()

        if symbol_data.empty:
            print(f"No data found for symbol {symbol}")
            return

        fig, axes = plt.subplots(4, 1, figsize=(15, 16))

        # Price and moving averages
        axes[0].plot(symbol_data.index, symbol_data['Close'], label='Close Price', linewidth=2)
        if 'SMA_10' in symbol_data.columns:
            axes[0].plot(symbol_data.index, symbol_data['SMA_10'], label='10-day SMA', alpha=0.8)
        if 'SMA_30' in symbol_data.columns:
            axes[0].plot(symbol_data.index, symbol_data['SMA_30'], label='30-day SMA', alpha=0.8)
        axes[0].set_title(f'{symbol} - Price and Moving Averages')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # RSI
        if 'RSI' in symbol_data.columns:
            axes[1].plot(symbol_data.index, symbol_data['RSI'], color='purple', linewidth=2)
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
            axes[1].fill_between(symbol_data.index, 30, 70, alpha=0.1, color='gray')
            axes[1].set_title(f'{symbol} - RSI (Relative Strength Index)')
            axes[1].set_ylabel('RSI')
            axes[1].set_ylim(0, 100)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # MACD
        if 'MACD' in symbol_data.columns and 'MACD_Signal' in symbol_data.columns:
            axes[2].plot(symbol_data.index, symbol_data['MACD'], label='MACD', linewidth=2)
            axes[2].plot(symbol_data.index, symbol_data['MACD_Signal'], label='Signal Line', linewidth=2)
            axes[2].bar(symbol_data.index, symbol_data['MACD'] - symbol_data['MACD_Signal'],
                       label='Histogram', alpha=0.3)
            axes[2].set_title(f'{symbol} - MACD')
            axes[2].set_ylabel('MACD')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        # Volume
        axes[3].bar(symbol_data.index, symbol_data['Volume'], alpha=0.7, color='orange')
        if 'Volume_SMA' in symbol_data.columns:
            axes[3].plot(symbol_data.index, symbol_data['Volume_SMA'], color='red', linewidth=2, label='Volume SMA')
        axes[3].set_title(f'{symbol} - Volume')
        axes[3].set_ylabel('Volume')
        axes[3].set_xlabel('Date')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_financial_health_scores(self, health_df, save_path=None):
        """Plot financial health scores"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Health scores bar chart
        health_df_sorted = health_df.sort_values('Health_Score', ascending=True)
        bars = ax1.barh(health_df_sorted['Symbol'], health_df_sorted['Health_Score'])

        # Color bars based on rating
        colors = {'A': 'green', 'B': 'lightgreen', 'C': 'yellow', 'D': 'orange', 'F': 'red'}
        for bar, rating in zip(bars, health_df_sorted['Rating']):
            bar.set_color(colors.get(rating, 'gray'))

        ax1.set_xlabel('Health Score')
        ax1.set_title('Company Financial Health Scores')
        ax1.grid(True, alpha=0.3)

        # Add score labels
        for i, (score, symbol) in enumerate(zip(health_df_sorted['Health_Score'], health_df_sorted['Symbol'])):
            ax1.text(score + 1, i, f'{score:.1f}', va='center')

        # Rating distribution pie chart
        rating_counts = health_df['Rating'].value_counts()
        colors_pie = [colors.get(rating, 'gray') for rating in rating_counts.index]
        ax2.pie(rating_counts.values, labels=rating_counts.index, autopct='%1.1f%%',
                colors=colors_pie, startangle=90)
        ax2.set_title('Health Rating Distribution')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_sector_performance(self, sector_analysis, save_path=None):
        """Plot sector performance comparison"""
        sectors = list(sector_analysis.keys())
        returns = [data['avg_return'] for data in sector_analysis.values()]
        volatilities = [data['avg_volatility'] for data in sector_analysis.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Returns bar chart
        bars = ax1.bar(sectors, returns, color=self.colors[:len(sectors)])
        ax1.set_xlabel('Sector')
        ax1.set_ylabel('Average Return (%)')
        ax1.set_title('Sector Performance - Average Returns')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5 if height >= 0 else height - 1,
                    f'{ret:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')

        # Risk-Return scatter plot
        ax2.scatter(volatilities, returns, s=200, c=self.colors[:len(sectors)], alpha=0.7)

        for i, sector in enumerate(sectors):
            ax2.annotate(sector, (volatilities[i], returns[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)

        ax2.set_xlabel('Average Volatility')
        ax2.set_ylabel('Average Return (%)')
        ax2.set_title('Sector Risk-Return Profile')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_risk_assessment(self, risk_df, save_path=None):
        """Plot risk assessment results"""
        if risk_df.empty:
            print("No risk data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Risk scores
        risk_df_sorted = risk_df.sort_values('Risk_Score')
        bars = axes[0, 0].barh(risk_df_sorted['Symbol'], risk_df_sorted['Risk_Score'])

        # Color by risk level
        risk_colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'orange', 'Very High': 'red'}
        for bar, risk_level in zip(bars, risk_df_sorted['Risk_Level']):
            bar.set_color(risk_colors.get(risk_level, 'gray'))

        axes[0, 0].set_xlabel('Risk Score')
        axes[0, 0].set_title('Company Risk Scores')
        axes[0, 0].grid(True, alpha=0.3)

        # Beta vs Volatility
        axes[0, 1].scatter(risk_df['Beta'], risk_df['Volatility_30d'],
                          c=[risk_colors.get(level, 'gray') for level in risk_df['Risk_Level']],
                          s=100, alpha=0.7)
        axes[0, 1].set_xlabel('Beta')
        axes[0, 1].set_ylabel('30-day Volatility')
        axes[0, 1].set_title('Beta vs Volatility')
        axes[0, 1].grid(True, alpha=0.3)

        # Risk level distribution
        risk_counts = risk_df['Risk_Level'].value_counts()
        colors_pie = [risk_colors.get(level, 'gray') for level in risk_counts.index]
        axes[1, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                      colors=colors_pie, startangle=90)
        axes[1, 0].set_title('Risk Level Distribution')

        # RSI distribution
        if 'Current_RSI' in risk_df.columns:
            axes[1, 1].hist(risk_df['Current_RSI'], bins=10, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 1].axvline(x=30, color='g', linestyle='--', label='Oversold (30)')
            axes[1, 1].axvline(x=70, color='r', linestyle='--', label='Overbought (70)')
            axes[1, 1].set_xlabel('RSI')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('RSI Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_correlation_matrix(self, correlation_matrix, save_path=None):
        """Plot correlation matrix heatmap"""
        fig, ax = plt.subplots(figsize=(12, 10))

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)

        ax.set_title('Stock Returns Correlation Matrix')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_model_predictions(self, actual, predicted, model_name, problem_type='regression', save_path=None):
        """Plot model predictions vs actual values"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        if problem_type == 'regression':
            # Scatter plot of predictions vs actual
            axes[0].scatter(actual, predicted, alpha=0.6, color=self.colors[0])
            axes[0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
            axes[0].set_xlabel('Actual Values')
            axes[0].set_ylabel('Predicted Values')
            axes[0].set_title(f'{model_name} - Predictions vs Actual')
            axes[0].grid(True, alpha=0.3)

            # Residuals plot
            residuals = actual - predicted
            axes[1].scatter(predicted, residuals, alpha=0.6, color=self.colors[1])
            axes[1].axhline(y=0, color='r', linestyle='--')
            axes[1].set_xlabel('Predicted Values')
            axes[1].set_ylabel('Residuals')
            axes[1].set_title(f'{model_name} - Residual Plot')
            axes[1].grid(True, alpha=0.3)

        else:  # classification
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(actual, predicted)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
            axes[0].set_xlabel('Predicted Label')
            axes[0].set_ylabel('True Label')
            axes[0].set_title(f'{model_name} - Confusion Matrix')

            # Accuracy by class
            class_accuracy = cm.diagonal() / cm.sum(axis=1)
            axes[1].bar(range(len(class_accuracy)), class_accuracy, color=self.colors[:len(class_accuracy)])
            axes[1].set_xlabel('Class')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title(f'{model_name} - Accuracy by Class')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_financial_dashboard(self, economic_report, df, save_path=None):
        """Create comprehensive financial dashboard"""
        save_dir = save_path or 'reports/figures'
        os.makedirs(save_dir, exist_ok=True)

        print("Creating Financial Dashboard...")

        # 1. Financial Health Scores
        if 'financial_health' in economic_report:
            self.plot_financial_health_scores(
                economic_report['financial_health'],
                f'{save_dir}/financial_health_scores.png'
            )

        # 2. Sector Performance
        if 'sector_analysis' in economic_report:
            self.plot_sector_performance(
                economic_report['sector_analysis'],
                f'{save_dir}/sector_performance.png'
            )

        # 3. Risk Assessment
        if 'risk_assessment' in economic_report and not economic_report['risk_assessment'].empty:
            self.plot_risk_assessment(
                economic_report['risk_assessment'],
                f'{save_dir}/risk_assessment.png'
            )

        # 4. Portfolio Correlation
        if 'portfolio_analysis' in economic_report and 'correlation_matrix' in economic_report['portfolio_analysis']:
            self.plot_correlation_matrix(
                economic_report['portfolio_analysis']['correlation_matrix'],
                f'{save_dir}/correlation_matrix.png'
            )

        # 5. Technical indicators for top companies
        top_companies = df['Symbol'].unique()[:3]
        for symbol in top_companies:
            self.plot_technical_indicators(df, symbol, f'{save_dir}/technical_{symbol}.png')

        print(f"Dashboard saved to {save_dir}/")

    def plot_economic_indicators(self, market_indices_data, save_path=None):
        """Plot major economic indicators"""
        if not market_indices_data:
            print("No market indices data provided")
            return

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()

        for i, (symbol, data) in enumerate(market_indices_data.items()):
            if i >= 4 or 'data' not in data or data['data'].empty:
                continue

            index_data = data['data']
            name = data['name']

            axes[i].plot(index_data.index, index_data['Close'], linewidth=2, color=self.colors[i])
            axes[i].set_title(f'{name} ({symbol})')
            axes[i].set_ylabel('Index Value')
            axes[i].grid(True, alpha=0.3)

            # Calculate and display performance
            if len(index_data) > 1:
                performance = (index_data['Close'].iloc[-1] / index_data['Close'].iloc[0] - 1) * 100
                axes[i].text(0.02, 0.98, f'Performance: {performance:.1f}%',
                           transform=axes[i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()