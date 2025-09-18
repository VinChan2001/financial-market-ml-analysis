import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

class EconomicAnalyzer:
    def __init__(self):
        self.sector_mapping = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'NVDA': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary',
            'PG': 'Consumer Staples',
            'JPM': 'Financial Services',
            'V': 'Financial Services',
            'JNJ': 'Healthcare'
        }

    def calculate_financial_health_score(self, df):
        """Calculate financial health score for companies"""
        health_scores = []

        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol].iloc[-1]  # Latest data

            score_components = {}

            # Profitability metrics (30%)
            if 'Fund_pe_ratio' in df.columns and symbol_data['Fund_pe_ratio'] > 0:
                pe_score = min(100, max(0, 100 - (symbol_data['Fund_pe_ratio'] - 15) * 5))
                score_components['pe_score'] = pe_score * 0.1
            else:
                score_components['pe_score'] = 50

            if 'Fund_price_to_book' in df.columns and symbol_data['Fund_price_to_book'] > 0:
                pb_score = min(100, max(0, 100 - (symbol_data['Fund_price_to_book'] - 1) * 20))
                score_components['pb_score'] = pb_score * 0.1
            else:
                score_components['pb_score'] = 50

            # Revenue growth proxy (using price performance)
            price_performance = (symbol_data['Close'] / symbol_data['Open'] - 1) * 100
            growth_score = min(100, max(0, 50 + price_performance * 2))
            score_components['growth_score'] = growth_score * 0.1

            # Liquidity (20%)
            if 'Fund_total_cash' in df.columns and 'Fund_total_debt' in df.columns:
                cash = symbol_data.get('Fund_total_cash', 0)
                debt = symbol_data.get('Fund_total_debt', 1)
                liquidity_ratio = cash / max(debt, 1)
                liquidity_score = min(100, liquidity_ratio * 50)
                score_components['liquidity_score'] = liquidity_score * 0.2
            else:
                score_components['liquidity_score'] = 50

            # Market performance (30%)
            volatility = symbol_data.get('Volatility_30d', 0.02)
            volatility_score = max(0, 100 - volatility * 1000)
            score_components['volatility_score'] = volatility_score * 0.15

            rsi = symbol_data.get('RSI', 50)
            rsi_score = 100 - abs(rsi - 50) * 2  # Prefer RSI around 50
            score_components['rsi_score'] = rsi_score * 0.15

            # Size and stability (20%)
            if 'Fund_market_cap' in df.columns and symbol_data['Fund_market_cap'] > 0:
                market_cap_score = min(100, np.log10(symbol_data['Fund_market_cap']) * 10)
                score_components['market_cap_score'] = market_cap_score * 0.2
            else:
                score_components['market_cap_score'] = 50

            total_score = sum(score_components.values())

            health_scores.append({
                'Symbol': symbol,
                'Company': symbol_data['Company'],
                'Health_Score': total_score,
                'Score_Components': score_components,
                'Rating': self._get_health_rating(total_score)
            })

        return pd.DataFrame(health_scores)

    def _get_health_rating(self, score):
        """Convert numeric score to letter rating"""
        if score >= 80:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 60:
            return 'C'
        elif score >= 50:
            return 'D'
        else:
            return 'F'

    def sector_performance_analysis(self, df):
        """Analyze performance by sector"""
        sector_analysis = {}

        for symbol in df['Symbol'].unique():
            sector = self.sector_mapping.get(symbol, 'Other')
            symbol_data = df[df['Symbol'] == symbol]

            if len(symbol_data) < 2:
                continue

            # Calculate sector metrics
            price_change = (symbol_data['Close'].iloc[-1] / symbol_data['Close'].iloc[0] - 1) * 100
            avg_volume = symbol_data['Volume'].mean()
            volatility = symbol_data['Volatility_30d'].iloc[-1] if 'Volatility_30d' in symbol_data.columns else 0
            avg_rsi = symbol_data['RSI'].mean() if 'RSI' in symbol_data.columns else 50

            if sector not in sector_analysis:
                sector_analysis[sector] = {
                    'companies': [],
                    'price_changes': [],
                    'volumes': [],
                    'volatilities': [],
                    'rsi_values': []
                }

            sector_analysis[sector]['companies'].append(symbol)
            sector_analysis[sector]['price_changes'].append(price_change)
            sector_analysis[sector]['volumes'].append(avg_volume)
            sector_analysis[sector]['volatilities'].append(volatility)
            sector_analysis[sector]['rsi_values'].append(avg_rsi)

        # Calculate sector summary statistics
        sector_summary = {}
        for sector, data in sector_analysis.items():
            sector_summary[sector] = {
                'num_companies': len(data['companies']),
                'avg_return': np.mean(data['price_changes']),
                'return_std': np.std(data['price_changes']),
                'avg_volume': np.mean(data['volumes']),
                'avg_volatility': np.mean(data['volatilities']),
                'avg_rsi': np.mean(data['rsi_values']),
                'companies': data['companies']
            }

        return sector_summary

    def risk_assessment(self, df):
        """Perform comprehensive risk assessment"""
        risk_analysis = {}

        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol]

            if len(symbol_data) < 30:  # Need sufficient data
                continue

            # Market risk (Beta proxy using correlation with overall market)
            symbol_returns = symbol_data['Price_Change'].dropna()
            market_returns = df.groupby(df.index)['Price_Change'].mean()
            market_returns = market_returns.loc[symbol_returns.index]

            if len(symbol_returns) > 10 and len(market_returns) > 10:
                beta = np.cov(symbol_returns, market_returns)[0, 1] / np.var(market_returns)
            else:
                beta = 1.0

            # Volatility risk
            volatility_30d = symbol_data['Volatility_30d'].iloc[-1] if 'Volatility_30d' in symbol_data.columns else 0

            # Technical risk indicators
            current_rsi = symbol_data['RSI'].iloc[-1] if 'RSI' in symbol_data.columns else 50
            price_position = symbol_data['Close_Position'].iloc[-1] if 'Close_Position' in symbol_data.columns else 0.5

            # Financial risk (if fundamental data available)
            debt_to_equity = 0
            if 'Fund_total_debt' in symbol_data.columns and 'Fund_market_cap' in symbol_data.columns:
                debt = symbol_data['Fund_total_debt'].iloc[-1]
                equity = symbol_data['Fund_market_cap'].iloc[-1]
                if equity > 0:
                    debt_to_equity = debt / equity

            # Overall risk score (lower is better)
            risk_score = (
                min(abs(beta - 1) * 30, 30) +  # Beta risk (30 max)
                min(volatility_30d * 1000, 30) +  # Volatility risk (30 max)
                max(0, min((current_rsi - 70) * 2, 20)) +  # Overbought risk (20 max)
                max(0, min((30 - current_rsi) * 2, 20)) +  # Oversold risk (20 max)
                min(debt_to_equity * 50, 20)  # Financial leverage risk (20 max)
            )

            risk_analysis[symbol] = {
                'Symbol': symbol,
                'Beta': beta,
                'Volatility_30d': volatility_30d,
                'Current_RSI': current_rsi,
                'Price_Position': price_position,
                'Debt_to_Equity': debt_to_equity,
                'Risk_Score': risk_score,
                'Risk_Level': self._get_risk_level(risk_score)
            }

        return pd.DataFrame.from_dict(risk_analysis, orient='index')

    def _get_risk_level(self, score):
        """Convert risk score to risk level"""
        if score <= 20:
            return 'Low'
        elif score <= 40:
            return 'Medium'
        elif score <= 60:
            return 'High'
        else:
            return 'Very High'

    def portfolio_optimization_analysis(self, df):
        """Basic portfolio optimization analysis"""
        symbols = df['Symbol'].unique()
        returns_matrix = []

        # Calculate return matrix
        for symbol in symbols:
            symbol_data = df[df['Symbol'] == symbol]['Price_Change'].dropna()
            if len(symbol_data) >= 20:
                returns_matrix.append(symbol_data.values)

        if len(returns_matrix) < 2:
            return {"error": "Insufficient data for portfolio analysis"}

        # Align lengths
        min_length = min(len(returns) for returns in returns_matrix)
        returns_matrix = [returns[-min_length:] for returns in returns_matrix]

        returns_df = pd.DataFrame(returns_matrix).T
        returns_df.columns = symbols[:len(returns_matrix)]

        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()

        # Calculate portfolio metrics for equal weights
        weights = np.array([1/len(returns_df.columns)] * len(returns_df.columns))
        portfolio_return = np.sum(returns_df.mean() * weights) * 252  # Annualized
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov() * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

        return {
            'correlation_matrix': correlation_matrix,
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'individual_returns': returns_df.mean() * 252,
            'individual_volatilities': returns_df.std() * np.sqrt(252)
        }

    def market_anomaly_detection(self, df):
        """Detect market anomalies using Isolation Forest"""
        anomalies = {}

        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol]

            if len(symbol_data) < 50:
                continue

            # Features for anomaly detection
            features = ['Price_Change', 'Volume_Ratio', 'RSI', 'Volatility_10d']
            available_features = [f for f in features if f in symbol_data.columns]

            if len(available_features) < 2:
                continue

            X = symbol_data[available_features].dropna()

            if len(X) < 20:
                continue

            # Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(X)

            anomaly_dates = X.index[anomaly_labels == -1]
            anomaly_data = symbol_data.loc[anomaly_dates]

            anomalies[symbol] = {
                'anomaly_count': len(anomaly_dates),
                'anomaly_dates': anomaly_dates.tolist(),
                'anomaly_price_changes': anomaly_data['Price_Change'].tolist() if 'Price_Change' in anomaly_data.columns else []
            }

        return anomalies

    def economic_indicator_correlation(self, df, market_indices_data):
        """Analyze correlation with economic indicators"""
        correlations = {}

        # Calculate average market performance
        stock_performance = df.groupby(df.index)['Price_Change'].mean()

        for index_symbol, index_data in market_indices_data.items():
            if 'data' in index_data and not index_data['data'].empty:
                index_returns = index_data['data']['Close'].pct_change().dropna()

                # Align dates
                common_dates = stock_performance.index.intersection(index_returns.index)
                if len(common_dates) > 10:
                    aligned_stock = stock_performance.loc[common_dates]
                    aligned_index = index_returns.loc[common_dates]

                    correlation = aligned_stock.corr(aligned_index)
                    correlations[index_data['name']] = {
                        'correlation': correlation,
                        'data_points': len(common_dates)
                    }

        return correlations

    def generate_economic_report(self, df, market_indices_data=None):
        """Generate comprehensive economic analysis report"""
        print("Generating Economic Analysis Report...")

        report = {
            'financial_health': self.calculate_financial_health_score(df),
            'sector_analysis': self.sector_performance_analysis(df),
            'risk_assessment': self.risk_assessment(df),
            'portfolio_analysis': self.portfolio_optimization_analysis(df),
            'anomaly_detection': self.market_anomaly_detection(df)
        }

        if market_indices_data:
            report['economic_correlations'] = self.economic_indicator_correlation(df, market_indices_data)

        return report

    def get_investment_recommendations(self, economic_report):
        """Generate investment recommendations based on analysis"""
        recommendations = []

        # Health-based recommendations
        health_df = economic_report['financial_health']
        top_health = health_df.nlargest(3, 'Health_Score')

        recommendations.append("TOP FINANCIAL HEALTH PICKS:")
        for _, row in top_health.iterrows():
            recommendations.append(f"  {row['Symbol']} ({row['Company']}): {row['Health_Score']:.1f}/100 - Rating {row['Rating']}")

        # Risk-based recommendations
        if not economic_report['risk_assessment'].empty:
            risk_df = economic_report['risk_assessment']
            low_risk = risk_df[risk_df['Risk_Level'] == 'Low']

            if not low_risk.empty:
                recommendations.append("\nLOW RISK OPTIONS:")
                for _, row in low_risk.head(3).iterrows():
                    recommendations.append(f"  {row['Symbol']}: Risk Score {row['Risk_Score']:.1f}")

        # Sector recommendations
        sector_analysis = economic_report['sector_analysis']
        best_sector = max(sector_analysis.items(), key=lambda x: x[1]['avg_return'])

        recommendations.append(f"\nBEST PERFORMING SECTOR: {best_sector[0]}")
        recommendations.append(f"  Average Return: {best_sector[1]['avg_return']:.2f}%")
        recommendations.append(f"  Companies: {', '.join(best_sector[1]['companies'])}")

        return "\n".join(recommendations)