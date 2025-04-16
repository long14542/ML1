import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve


def plot_trend_visualizations(df):
    """
    Create individual plots to evaluate trends in the data, each saved as a separate PNG
    """
    # Plot 1: Depression distribution by gender
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Gender', hue='Depression', data=df)
    plt.title('Depression Distribution by Gender')
    plt.xlabel('Gender (0: Male, 1: Female)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('trend_gender_depression.png')
    plt.close()

    # Plot 2: Depression distribution by sleep duration
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Sleep Duration', hue='Depression', data=df)
    plt.title('Depression Distribution by Sleep Duration')
    plt.xlabel('Sleep Duration (1: <5h, 2: 5-6h, 3: 7-8h, 4: >8h)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('trend_sleep_depression.png')
    plt.close()

    # Plot 3: Depression vs Academic Pressure
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Depression', y='Academic Pressure', data=df)
    plt.title('Academic Pressure vs Depression')
    plt.tight_layout()
    plt.savefig('trend_academic_pressure.png')
    plt.close()

    # Plot 4: Depression vs Study Satisfaction
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Depression', y='Study Satisfaction', data=df)
    plt.title('Study Satisfaction vs Depression')
    plt.tight_layout()
    plt.savefig('trend_study_satisfaction.png')
    plt.close()

    # Plot 5: Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('trend_correlation_heatmap.png')
    plt.close()

    # Additional plot: Age distribution by depression
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', hue='Depression', kde=True, bins=15)
    plt.title('Age Distribution by Depression Status')
    plt.tight_layout()
    plt.savefig('trend_age_distribution.png')
    plt.close()


def plot_model_performance(train_metrics, val_metrics, test_metrics, X_test, y_test, model):
    """
    Create individual plots to check model performance, each saved as a separate PNG
    """
    # Plot 1: ROC Curve
    plt.figure(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y_test, test_metrics['y_prob'])
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {test_metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('performance_roc_curve.png')
    plt.close()

    # Plot 2: Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    precision, recall, _ = precision_recall_curve(y_test, test_metrics['y_prob'])
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    plt.savefig('performance_precision_recall_curve.png')
    plt.close()

    # Plot 3: Confusion Matrix Heatmap
    plt.figure(figsize=(10, 6))
    cm = test_metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('performance_confusion_matrix.png')
    plt.close()

    # Plot 4: Feature Importance
    plt.figure(figsize=(12, 8))
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': np.abs(model.coef_[0])
    }).sort_values('Importance', ascending=False)

    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('performance_feature_importance.png')
    plt.close()

    # Comparison of metrics across datasets
    plt.figure(figsize=(12, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    train_values = [train_metrics[m] for m in metrics]
    val_values = [val_metrics[m] for m in metrics]
    test_values = [test_metrics[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.25

    plt.bar(x - width, train_values, width, label='Train')
    plt.bar(x, val_values, width, label='Validation')
    plt.bar(x + width, test_values, width, label='Test')

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig('performance_metrics_comparison.png')
    plt.close()
