import pandas as pd
import warnings
from data_preprocessing import load_and_preprocess_data
from model_training import train_logistic_regression, evaluate_model
from visualization import plot_trend_visualizations, plot_model_performance
from model_optimization import tune_hyperparameters, evaluate_model_stability


def main():
    # Tắt các cảnh báo không cần thiết
    warnings.filterwarnings("ignore", category=UserWarning)

    # Set the file path
    file_path = (r"C:\Users\admin\Downloads\Student Depression Dataset.csv")

    # Load and preprocess the data
    X_train, X_val, X_test, y_train, y_val, y_test, df = load_and_preprocess_data()

    # Create trend visualizations
    print("Creating trend visualizations...")
    plot_trend_visualizations(df)

    # Điều chỉnh hyperparameter để tìm cấu hình tối ưu
    print("\nOptimizing model hyperparameters...")
    best_model, best_params = tune_hyperparameters(X_train, y_train, X_val, y_val)

    # Đánh giá độ ổn định của mô hình thông qua cross-validation
    print("\nEvaluating model stability...")
    # Kết hợp tập train và validation để cross-validation
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])
    cv_results = evaluate_model_stability(X_combined, y_combined, best_params)

    # Train the logistic regression model with best parameters
    print("\nTraining final logistic regression model with best parameters...")
    model = train_logistic_regression(X_train, y_train, best_params)

    # Evaluate the model on all datasets
    print("\nEvaluating model performance...")
    train_metrics = evaluate_model(model, X_train, y_train, "Training")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")

    # Create performance visualizations
    print("\nCreating performance visualizations...")
    plot_model_performance(train_metrics, val_metrics, test_metrics, X_test, y_test, model)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
