import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data():
    """
    Load and preprocess the student depression dataset
    """
    # Load the dataset
    df = pd.read_csv(r"C:\Users\admin\Downloads\Student Depression Dataset.csv")

    # Remove specified columns
    columns_to_remove = ['id', 'City', 'Profession', 'Degree',
                         'Have you ever had suicidal thoughts ?',
                         'Work Pressure', 'Job Satisfaction']

    df = df.drop(columns=columns_to_remove, errors='ignore')

    # Check for missing values before dropping
    print(f"Missing values before cleaning:\n{df.isnull().sum()}")

    # Handle missing values
    df = df.dropna()

    # Convert categorical variables to numerical
    # Sleep Duration
    sleep_mapping = {
        'Less than 5 hours': 1,
        '5-6 hours': 2,
        '7-8 hours': 3,
        'More than 8 hours': 4
    }
    df['Sleep Duration'] = df['Sleep Duration'].map(sleep_mapping)

    # Dietary Habits
    diet_mapping = {
        'Unhealthy': 0,
        'Moderate': 1,
        'Healthy': 2
    }
    df['Dietary Habits'] = df['Dietary Habits'].map(diet_mapping)

    # Family History of Mental Illness
    df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})

    # Gender
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    # Suppress detailed missing value reports after transformations
    missing_after = df.isnull().sum()
    if missing_after.sum() > 0:
        print(f"Total missing values after transformations: {missing_after.sum()}")

    # Drop any remaining rows with missing values
    df = df.dropna()

    # Prepare features and target
    X = df.drop('Depression', axis=1)
    y = df['Depression']

    # Split the data into training, validation, and test sets (60/20/20)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )  # 0.25 * 0.8 = 0.2 of the original data

    # Standardize numerical features
    # Chỉ chuẩn hoá các cột liên tục/ordinal phù hợp
    columns_to_scale = [
        'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction',
        'Sleep Duration', 'Work/Study Hours', 'Financial Stress',
        'Dietary Habits'
    ]

    scaler = StandardScaler()

    X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_val[columns_to_scale] = scaler.transform(X_val[columns_to_scale])
    X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    # Check for any NaN values in the processed data
    nan_count = np.isnan(X_train.values).sum()
    if nan_count > 0:
        print(f"NaN values in X after preprocessing: {nan_count}")

    return X_train, X_val, X_test, y_train, y_val, y_test, df
