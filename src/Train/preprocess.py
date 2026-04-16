from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from src.config import BASE_DIR, get_param


def prepare_data(
    target_column: str | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
):
    data_path = BASE_DIR / get_param("paths", "processed_data_path")

    target_column = target_column or get_param("training", "target_column")
    test_size = test_size if test_size is not None else get_param("training", "test_size")
    random_state = (
        random_state if random_state is not None else get_param("training", "random_state")
    )

    print("Loading processed data...")
    df = pd.read_csv(data_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    print("Features and target separated.")

    feature_names = X.columns.tolist()
    full_class_distribution = y.value_counts().to_dict()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print("Train-test split completed.")

    y_train_distribution_before_smote = y_train.value_counts().to_dict()
    y_test_distribution = y_test.value_counts().to_dict()

    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("SMOTE applied successfully.")

    y_train_distribution_after_smote = y_train.value_counts().to_dict()

    data_info = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "x_train_rows": X_train.shape[0],
        "x_train_cols": X_train.shape[1],
        "x_test_rows": X_test.shape[0],
        "x_test_cols": X_test.shape[1],
        "full_class_distribution": full_class_distribution,
        "y_train_distribution_before_smote": y_train_distribution_before_smote,
        "y_train_distribution_after_smote": y_train_distribution_after_smote,
        "y_test_distribution": y_test_distribution,
    }

    return X_train, X_test, y_train, y_test, data_info