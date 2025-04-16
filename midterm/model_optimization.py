import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Tìm hyperparameters tối ưu cho mô hình Logistic Regression
    sử dụng tập validation
    """
    print("Tuning hyperparameters...")

    # Định nghĩa các tổ hợp hyperparameter hợp lệ
    valid_params = [
        # L1 regularization
        {'C': c, 'penalty': 'l1', 'solver': solver, 'max_iter': 1000}
        for c in [0.001, 0.01, 0.1, 1, 10, 100]
        for solver in ['liblinear', 'saga']
    ]

    # L2 regularization
    valid_params.extend([
        {'C': c, 'penalty': 'l2', 'solver': solver, 'max_iter': 1000}
        for c in [0.001, 0.01, 0.1, 1, 10, 100]
        for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    ])

    # No regularization
    valid_params.extend([
        {'C': 1.0, 'penalty': None, 'solver': solver, 'max_iter': 1000}
        for solver in ['newton-cg', 'lbfgs', 'sag', 'saga']
    ])

    best_model = None
    best_val_score = 0
    best_params = None

    # Thử từng tổ hợp hyperparameters
    for i, params in enumerate(valid_params):
        try:
            # Huấn luyện mô hình với params hiện tại
            model = LogisticRegression(random_state=42, **params)
            model.fit(X_train, y_train)

            # Đánh giá trên tập validation
            y_val_pred = model.predict(X_val)
            val_score = f1_score(y_val, y_val_pred)

            # Chỉ in ra các kết quả tốt (F1 > 0.82) để giảm output
            if val_score > 0.82:
                print(f"Params {i + 1}/{len(valid_params)}: {params}, F1 Score: {val_score:.4f}")

            # Lưu mô hình tốt nhất
            if val_score > best_val_score:
                best_val_score = val_score
                best_model = model
                best_params = params
        except Exception:
            # Bỏ qua thông báo lỗi
            pass

    print("\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best validation F1 score: {best_val_score:.4f}")

    # Vẽ biểu đồ so sánh các giá trị C (với penalty và solver tốt nhất)
    plot_c_comparison(X_train, y_train, X_val, y_val, best_params)

    return best_model, best_params


def plot_c_comparison(X_train, y_train, X_val, y_val, best_params):
    """
    Vẽ biểu đồ so sánh hiệu suất với các giá trị C khác nhau
    """
    # Chỉ thực hiện nếu có penalty (không phải None)
    if best_params['penalty'] is None:
        return

    c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    train_scores = []
    val_scores = []

    for c in c_values:
        params = best_params.copy()
        params['C'] = c

        try:
            model = LogisticRegression(random_state=42, **params)
            model.fit(X_train, y_train)

            # Tính điểm trên tập training
            y_train_pred = model.predict(X_train)
            train_score = f1_score(y_train, y_train_pred)
            train_scores.append(train_score)

            # Tính điểm trên tập validation
            y_val_pred = model.predict(X_val)
            val_score = f1_score(y_val, y_val_pred)
            val_scores.append(val_score)
        except Exception:
            # Nếu có lỗi với giá trị C, thêm None
            train_scores.append(None)
            val_scores.append(None)

    # Lọc bỏ các giá trị None
    valid_indices = [i for i, (t, v) in enumerate(zip(train_scores, val_scores)) if t is not None and v is not None]
    valid_c = [c_values[i] for i in valid_indices]
    valid_train = [train_scores[i] for i in valid_indices]
    valid_val = [val_scores[i] for i in valid_indices]

    if len(valid_c) > 1:  # Chỉ vẽ nếu có ít nhất 2 điểm
        plt.figure(figsize=(10, 6))
        plt.semilogx(valid_c, valid_train, 'b-o', label='Training Score')
        plt.semilogx(valid_c, valid_val, 'r-o', label='Validation Score')
        plt.xlabel('C (Regularization parameter)')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs. Regularization Parameter')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('hyperparameter_c_comparison.png')
        plt.close()


def evaluate_model_stability(X, y, best_params, n_splits=5):
    """
    Đánh giá độ ổn định của mô hình thông qua cross-validation
    """
    print(f"\nEvaluating model stability with {n_splits}-fold cross-validation...")

    # Khởi tạo KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Khởi tạo mô hình với tham số tốt nhất
    model = LogisticRegression(random_state=42, **best_params)

    # Định nghĩa các metrics cần đánh giá
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    # Thực hiện cross-validation và in kết quả
    cv_results = {}
    for metric_name, scorer in scoring.items():
        # Tắt các cảnh báo từ scikit-learn trong quá trình cross_val_score
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(model, X, y, cv=kf, scoring=scorer)

        cv_results[metric_name] = scores
        print(f"{metric_name.capitalize()}: {scores.mean():.4f} (±{scores.std():.4f})")

    # Vẽ biểu đồ kết quả cross-validation
    plot_cv_results(cv_results, n_splits)

    return cv_results


def plot_cv_results(cv_results, n_splits):
    """
    Vẽ biểu đồ kết quả cross-validation
    """
    plt.figure(figsize=(12, 6))

    metrics = list(cv_results.keys())
    means = [cv_results[metric].mean() for metric in metrics]
    stds = [cv_results[metric].std() for metric in metrics]

    # Vẽ biểu đồ cột
    x = np.arange(len(metrics))
    width = 0.7

    plt.bar(x, means, width, yerr=stds, capsize=10,
            color='skyblue', edgecolor='black', alpha=0.8)

    # Thêm giá trị lên cột
    for i, v in enumerate(means):
        plt.text(i, v + stds[i] + 0.02, f"{v:.4f}", ha='center')

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'Cross-Validation Results ({n_splits}-fold)')
    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('cross_validation_results.png')
    plt.close()
