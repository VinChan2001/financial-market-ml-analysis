import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

class MLVisualizer:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50', '#9C27B0']

    def plot_data_distribution(self, df, target_col='target', save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].hist(df[target_col], bins=30, alpha=0.7, color=self.colors[0])
        axes[0, 0].set_title('Target Distribution')
        axes[0, 0].set_xlabel(target_col)
        axes[0, 0].set_ylabel('Frequency')

        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col)
        corr_matrix = df[numeric_cols[:10]].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
        axes[0, 1].set_title('Feature Correlation Matrix')

        axes[1, 0].boxplot([df[col] for col in numeric_cols[:5]], labels=numeric_cols[:5])
        axes[1, 0].set_title('Feature Distributions (Box Plot)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        df[numeric_cols[:5]].hist(bins=20, ax=axes[1, 1], alpha=0.7)
        axes[1, 1].set_title('Feature Histograms')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_model_comparison(self, model_results, problem_type='regression', save_path=None):
        model_names = list(model_results.keys())

        if problem_type == 'regression':
            train_scores = [model_results[name]['train_r2'] for name in model_names]
            test_scores = [model_results[name]['test_r2'] for name in model_names]
            metric = 'R² Score'
        else:
            train_scores = [model_results[name]['train_accuracy'] for name in model_names]
            test_scores = [model_results[name]['test_accuracy'] for name in model_names]
            metric = 'Accuracy'

        x = np.arange(len(model_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=self.figsize)
        bars1 = ax.bar(x - width/2, train_scores, width, label='Train', alpha=0.8, color=self.colors[0])
        bars2 = ax.bar(x + width/2, test_scores, width, label='Test', alpha=0.8, color=self.colors[1])

        ax.set_xlabel('Models')
        ax.set_ylabel(metric)
        ax.set_title(f'Model Performance Comparison - {metric}')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_predictions_vs_actual(self, y_true, y_pred, model_name, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.scatter(y_true, y_pred, alpha=0.6, color=self.colors[0])
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'{model_name} - Predictions vs Actual')

        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color=self.colors[1])
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{model_name} - Residual Plot')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path=None):
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'{model_name} - Confusion Matrix')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self, feature_importance, feature_names, model_name, top_n=10, save_path=None):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=True).tail(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(importance_df['feature'], importance_df['importance'], color=self.colors[0])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{model_name} - Top {top_n} Feature Importance')

        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.3f}',
                       xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(3, 0),
                       textcoords="offset points",
                       ha='left', va='center')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_comprehensive_report(self, model_results, df, X_test, y_test, problem_type='regression'):
        save_dir = 'reports/figures'
        os.makedirs(save_dir, exist_ok=True)

        print("Generating comprehensive visualization report...")

        self.plot_data_distribution(df, save_path=f'{save_dir}/data_distribution.png')

        self.plot_model_comparison(model_results, problem_type,
                                 save_path=f'{save_dir}/model_comparison.png')

        best_model_name = max(model_results.keys(),
                            key=lambda x: model_results[x]['test_r2'] if problem_type == 'regression'
                            else model_results[x]['test_accuracy'])

        best_predictions = model_results[best_model_name]['test_predictions']

        if problem_type == 'regression':
            self.plot_predictions_vs_actual(y_test, best_predictions, best_model_name,
                                           save_path=f'{save_dir}/best_model_predictions.png')
        else:
            self.plot_confusion_matrix(y_test, best_predictions, best_model_name,
                                     save_path=f'{save_dir}/best_model_confusion_matrix.png')

        best_model = model_results[best_model_name]['model']
        if hasattr(best_model, 'feature_importances_'):
            feature_names = [f'feature_{i}' for i in range(len(best_model.feature_importances_))]
            self.plot_feature_importance(best_model.feature_importances_, feature_names,
                                       best_model_name,
                                       save_path=f'{save_dir}/feature_importance.png')

        print(f"All visualizations saved to {save_dir}/")

    def plot_learning_curve(self, train_scores, val_scores, save_path=None):
        epochs = range(1, len(train_scores) + 1)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(epochs, train_scores, label='Training Score', color=self.colors[0])
        ax.plot(epochs, val_scores, label='Validation Score', color=self.colors[1])
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Score')
        ax.set_title('Learning Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()