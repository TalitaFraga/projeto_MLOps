from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")
DATA_PATH = BASE_DIR / os.getenv("PROCESSED_DATA_PATH")


def prepare_data(
    target_column: str | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
):
    target_column = target_column or os.getenv("TARGET_COLUMN")
    test_size = test_size if test_size is not None else float(os.getenv("TEST_SIZE"))
    random_state = random_state if random_state is not None else int(os.getenv("RANDOM_STATE"))

    print("Loading processed data...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    print("Features and target separated.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print("Train-test split completed.")

    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("SMOTE applied successfully.")

    return X_train, X_test, y_train, y_test